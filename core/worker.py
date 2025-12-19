import json
import traceback
import cv2
from concurrent.futures import ThreadPoolExecutor
from core.timing import TimingLogger
from detectors import (
    GroundingWireDetector,
    TapeDetector,
    TapeDetectorHailo,
    TapeDeviationDetector,
    WrongOrientation,
    MissingWiresDetector,
    TwistedWires,
)
from processors import ROICropper, YOLOROIMapper, WorkspaceExtractor
from utils.constants import (
    TAPE_DETECTOR_CONF_THRESHOLD,
    TAPE_CLASS_ID,
    LABEL_CLASS_ID,
    CONNECTOR_CLASS_ID,
    TAPE_DEVIATION_TOO_FAR,
    TAPE_DEVIATION_WRONG_LENGTH,
    WORKSPACE_EXTRACTOR_CONFIG,
    ERROR_CODES,
)


def _preprocess_roi(args):
    """
    Helper function to process a single ROI in a separate thread.

    Args:
        args (tuple): (workspace_image, roi_config, roi_name, cropper, target_size, zone_number)

    Returns:
        tuple: (roi_name, processed_image, roi_config, zone_number)
    """
    workspace_image, roi_config, roi_name, cropper, target_size, zone_number = args

    try:
        # Calculate bounds
        height, width = workspace_image.shape[:2]
        x1, y1, x2, y2 = cropper.calculate_roi_bounds(width, height, roi_config)

        # Clamp to image boundaries
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        # Validate bounds
        if x2 <= x1 or y2 <= y1:
            return roi_name, None, roi_config, zone_number

        # Crop the ROI
        # Note: Numpy slicing creates a view, but since we might resize or
        # pass to other threads/processes, and we want to release the original
        # workspace reference eventually, a copy might happen during resize anyway.
        cropped_roi = workspace_image[y1:y2, x1:x2]

        if cropped_roi.size == 0:
            return roi_name, None, roi_config, zone_number

        # Resize if needed (Heavy OpenCV operation -> Releases GIL)
        if target_size:
            processed_image = cv2.resize(cropped_roi, target_size)
        else:
            processed_image = cropped_roi

        return roi_name, processed_image, roi_config, zone_number

    except Exception as e:
        print(f"Error processing ROI {roi_name}: {e}")
        return roi_name, None, roi_config, zone_number


def _extract_workspace_helper(args):
    """
    Helper to extract workspace in a thread.
    Args: (image, extractor, zone_number)
    """
    image, extractor, zone_number = args
    try:
        ws = extractor.extract_workspace(image)
        return zone_number, ws
    except Exception as e:
        print(f"Error extracting workspace for zone {zone_number}: {e}")
        return zone_number, None


def _orientation_check_helper(args):
    """
    Helper to check orientation in a thread.
    Args: (image, detector, roi_name, zone_number)
    """
    image, detector, roi_name, zone_number = args
    try:
        is_wrong = detector.detect(image)
        return roi_name, zone_number, is_wrong
    except Exception as e:
        print(f"Error checking orientation for {roi_name}: {e}")
        return roi_name, zone_number, False


def _twisted_wires_check_helper(args):
    """
    Helper to check for twisted wires in a thread.
    Args: (workspace_rgb, detector, zone_number)
    """
    workspace_rgb, detector, zone_number = args
    try:
        is_twisted = detector.detect(workspace_rgb)
        return zone_number, is_twisted
    except Exception as e:
        print(f"Error checking twisted wires for zone {zone_number}: {e}")
        return zone_number, False


def worker_logic(command_queue, result_queue, config_paths):
    """
    Main logic for the worker process.

    Args:
        command_queue (multiprocessing.Queue): Queue for receiving commands.
        result_queue (multiprocessing.Queue): Queue for sending results.
        config_paths (dict): Dictionary containing paths to configuration files.
    """
    try:
        # --- Initialization Phase ---
        print("[Worker] Initializing...")

        # Load configurations
        with open(config_paths["roi_z1"], "r") as f:
            roi_data_z1 = json.load(f)
        with open(config_paths["roi_z2"], "r") as f:
            roi_data_z2 = json.load(f)
        with open(config_paths["positions"], "r") as f:
            positions = json.load(f)
        with open(config_paths["env"], "r") as f:
            env = json.load(f)    
        tape_detector = TapeDetectorHailo(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD) if env["use_hailo"] else TapeDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD)

        detectors = {
            "roi_cropper_z1": ROICropper(roi_data_z1),
            "roi_cropper_z2": ROICropper(roi_data_z2),
            "grounding_detector": GroundingWireDetector(),
            "tape_detector": tape_detector,
            "tape_deviation_detector": TapeDeviationDetector(positions),
            "yolo_roi_mapper": YOLOROIMapper(),
            "branch_wrong_orientation_detector": WrongOrientation(),
            "missing_wires_detector": MissingWiresDetector(),
            "twisted_wires_detector": TwistedWires(),
            "workspace_extractor": WorkspaceExtractor(WORKSPACE_EXTRACTOR_CONFIG),
        }

        # Helper to find ROI object
        def find_roi_object(cropper, category, roi_id):
            for roi in cropper.roi_objects.get(category.lower(), []):
                if roi.get("id") == roi_id:
                    return roi
            return None

        print("[Worker] Initialization complete. Ready.")

        # Initialize ThreadPoolExecutor
        # Max workers = 4 to match Raspberry Pi 5 core count
        with ThreadPoolExecutor(max_workers=4) as executor:

            # --- Loop Phase ---
            while True:
                command_data = command_queue.get()

                if command_data["command"] == "STOP":
                    print("[Worker] Stopping...")
                    break

                elif command_data["command"] == "TRIGGER":
                    images = command_data["images"]

                    try:
                        timer = TimingLogger()
                        error_codes = set()
                        timer.start("total_inspection_time")

                        # 1. Parallel Workspace Extraction
                        # Prepare tasks
                        ws_tasks = []
                        for i, img in enumerate(images):
                            zone_number = i + 1
                            ws_tasks.append(
                                (img, detectors["workspace_extractor"], zone_number)
                            )

                        timer.start("workspace_extraction_total")
                        ws_results = list(
                            executor.map(_extract_workspace_helper, ws_tasks)
                        )
                        timer.stop("workspace_extraction_total")

                        workspaces = []
                        for zone_number, ws in ws_results:
                            # Log individual times (approximate, since they ran in parallel)
                            # We can't easily time individual threads with the current TimingLogger structure
                            # without passing it in, but we have the total parallel time.
                            if ws is not None:
                                workspaces.append((zone_number, ws))

                        if not workspaces:
                            result_queue.put(
                                {
                                    "status": "DONE",
                                    "success": False,
                                    "error": "No workspace found",
                                }
                            )
                            continue

                        # 1.5. Twisted Wires Detection (on workspace images)
                        #timer.start("twisted_wires_detection_total")
                        #twisted_wires_tasks = []
                        #for zone_number, workspace in workspaces:
                        #    # Convert BGR to RGB for the detector
                        #    workspace_rgb = cv2.cvtColor(workspace, cv2.COLOR_BGR2RGB)
                        #    twisted_wires_tasks.append(
                        #        (
                        #            workspace_rgb,
                        #            detectors["twisted_wires_detector"],
                        #            zone_number,
                        #        )
                        #    )
#
 #                       twisted_wires_results = list(
  #                          executor.map(
   #                             _twisted_wires_check_helper, twisted_wires_tasks
    #                        )
      #                  )
     #                   timer.stop("twisted_wires_detection_total")
#
 #                       for zone_number, is_twisted in twisted_wires_results:
  #                          if is_twisted:
   #                             error_codes.add(ERROR_CODES["TWISTED_WIRES"])

                        # 2. Batch Preparation & Parallel Processing
                        tape_batch_images = []
                        tape_batch_metadata = []

                        roi_map_per_zone = {}
                        roi_processing_tasks = []

                        for zone_number, workspace in workspaces:
                            if zone_number == 1:
                                cropper = detectors["roi_cropper_z1"]
                                roi_data = roi_data_z1
                            else:
                                cropper = detectors["roi_cropper_z2"]
                                roi_data = roi_data_z2

                            roi_map_per_zone[zone_number] = {
                                "workspace": workspace,
                                "roi_data": roi_data,
                                "cropper": cropper,
                            }

                            # Iterate over all ROI categories and objects to build the task list
                            for category_name, roi_list in cropper.roi_objects.items():
                                for roi_config in roi_list:
                                    roi_id = roi_config.get("id", 0)
                                    # Create unique key
                                    roi_name = f"{category_name.upper()}_{roi_id:03d}"

                                    # Determine target size
                                    target_size = None
                                    if roi_name.startswith(
                                        "TAPE"
                                    ) or roi_name.startswith("LABEL"):
                                        target_size = (640, 640)

                                    # Add zone_number to task args if needed, or just track it
                                    # We need to know which zone this result belongs to.
                                    # Let's add zone_number to the return value of _preprocess_roi?
                                    # Or just rely on the order? executor.map preserves order.
                                    # But we are mixing zones in the loop? No, we are looping zones.
                                    # Let's collect ALL tasks from ALL zones first to maximize parallelism.
                                    roi_processing_tasks.append(
                                        (
                                            workspace,
                                            roi_config,
                                            roi_name,
                                            cropper,
                                            target_size,
                                            zone_number,
                                        )
                                    )

                        timer.start("roi_processing_total")

                        # We need to update _preprocess_roi to handle zone_number if we pass it
                        # Or just wrap it. Let's update _preprocess_roi signature slightly or wrap the args.
                        # Actually, _preprocess_roi takes `args` tuple. I can just unpack 5 args and ignore the 6th if I want,
                        # or update it. Let's update `_preprocess_roi` to return zone_number too.

                        # Redefining _preprocess_roi locally or updating the top one?
                        # I'll update the top one in this file replacement.

                        # Execute tasks in parallel
                        roi_results = list(
                            executor.map(_preprocess_roi, roi_processing_tasks)
                        )

                        timer.stop("roi_processing_total")

                        # Process results
                        for result in roi_results:
                            # Unpack based on updated _preprocess_roi return
                            roi_name, roi_image, roi_config, zone_number = result
                            if roi_image is None:
                                continue

                            category, roi_id_str = roi_name.split("_")
                            roi_id = int(roi_id_str)

                            if roi_name.startswith("GROUNDING"):
                                timer.start(
                                    f"grounding_detector_{roi_name}_z{zone_number}"
                                )
                                if not detectors["grounding_detector"].is_present(
                                    roi_image
                                ):
                                    error_codes.add(ERROR_CODES["GROUNDING_MISSING"])
                                timer.stop(
                                    f"grounding_detector_{roi_name}_z{zone_number}"
                                )
                                timer.add(
                                    "grounding_detector_total",
                                    timer.timings.get(
                                        f"grounding_detector_{roi_name}_z{zone_number}",
                                        0,
                                    ),
                                )

                            elif roi_name.startswith("WIRES"):
                                timer.start(f"wires_detector_{roi_name}_z{zone_number}")
                                expected_colors = roi_config.get(
                                    "expected_colors", None
                                )

                                if zone_number == 2 and "wires" in roi_data_z2:
                                    wires_config = roi_data_z2["wires"]
                                    if (
                                        isinstance(wires_config, dict)
                                        and "color_ranges" in wires_config
                                    ):
                                        detectors[
                                            "missing_wires_detector"
                                        ].set_color_ranges_from_dict(
                                            wires_config["color_ranges"]
                                        )

                                if not detectors["missing_wires_detector"].is_present(
                                    roi_image, expected_colors
                                ):
                                    error_codes.add(ERROR_CODES["WIRES_MISSING"])
                                timer.stop(f"wires_detector_{roi_name}_z{zone_number}")
                                timer.add(
                                    "missing_wires_detector_total",
                                    timer.timings.get(
                                        f"wires_detector_{roi_name}_z{zone_number}", 0
                                    ),
                                )

                            elif roi_name.startswith("TAPE") or roi_name.startswith(
                                "LABEL"
                            )  or roi_name.startswith("CONNECTORS") :
                                tape_batch_images.append(roi_image)
                                tape_batch_metadata.append(
                                    {
                                        "zone": zone_number,
                                        "type": (
                                            "TAPE"
                                            if roi_name.startswith("TAPE")
                                            else "LABEL" if roi_name.startswith("LABEL") else "CONNECTORS"
                                        ),
                                        "id": roi_id,
                                        "name": roi_name,
                                    }
                                )

                        # 3. Batch Inference (TapeDetector)
                        if tape_batch_images:
                            timer.start("tape_detector_total")
                            tape_results = detectors["tape_detector"].predict_batch(
                                tape_batch_images
                            )
                            total_tape_time = timer.stop("tape_detector_total")

                            # Attribute time proportionally
                            avg_time = total_tape_time / len(tape_batch_images)
                            for meta in tape_batch_metadata:
                                roi_name = meta["name"]
                                zone = meta["zone"]
                                timer.add(f"tape_detector_{roi_name}_z{zone}", avg_time)

                            annotations_per_zone = {z: {} for z, _ in workspaces}

                            for result, meta in zip(tape_results, tape_batch_metadata):
                                zone = meta["zone"]
                                roi_type = meta["type"]
                                roi_id = meta["id"]

                                detected_classes = (
                                    result.boxes.cls.tolist()
                                    if result.boxes is not None
                                    else []
                                )

                                if roi_type == "TAPE":
                                    if TAPE_CLASS_ID not in detected_classes:
                                        print(roi_id, detected_classes)
                                        error_codes.add(
                                            ERROR_CODES["TAPE_NOT_DETECTED"]
                                        )
                                        continue

                                    if roi_id == 2 and CONNECTOR_CLASS_ID in detected_classes:
                                        error_codes.add(
                                            ERROR_CODES["WRONG_ORIENTATION"]
                                        )
                                        
                                        
                                            
                                    #if roi_id not in annotations_per_zone[zone]:
                                    #    annotations_per_zone[zone][roi_id] = []

                                    for box_data in result.boxes:
                                        x_center, y_center, width, height = (
                                            box_data.xywhn[0]
                                        )
                                        #annotations_per_zone[zone][roi_id].append(
                                        #    [
                                        #        int(box_data.cls[0]),
                                        #        x_center.item(),
                                        #        y_center.item(),
                                        #        width.item(),
                                         #       height.item(),
                                         #   ]
                                        #)

                                        try:
                                            correct = detectors[
                                                "tape_deviation_detector"
                                            ].is_tape_correct(roi_id, x_center, width)
                                            if correct == TAPE_DEVIATION_TOO_FAR:
                                                error_codes.add(
                                                    ERROR_CODES["TAPE_TOO_FAR"]
                                                )
                                            elif correct == TAPE_DEVIATION_WRONG_LENGTH:
                                                error_codes.add(
                                                    ERROR_CODES["TAPE_WRONG_LENGTH"]
                                                )
                                        except (ValueError, IndexError):
                                            pass

                                elif roi_type == "LABEL":
                                    if LABEL_CLASS_ID not in detected_classes:
                                        error_codes.add(
                                            ERROR_CODES["LABEL_NOT_DETECTED"]
                                        )
                                elif roi_type == "CONNECTORS":
                                    # CONNECTOR_CLASS_ID
                                    if CONNECTOR_CLASS_ID not in detected_classes:
                                        error_codes.add(
                                            ERROR_CODES["WRONG_ORIENTATION"]
                                        )

                        # 4. Orientation Check (Parallelized)
                        #orientation_tasks = []

                        #timer.start("yolo_roi_mapper_total")
                        #for zone_number, data in roi_map_per_zone.items():
                           # workspace = data["workspace"]
                           # roi_data = data["roi_data"]
                            #annotations = annotations_per_zone.get(zone_number, {})

                            # This part is fast enough to keep sequential or parallelize?
                            # It crops images based on YOLO results.
                           # new_rois_images, _ = detectors[
                           #     "yolo_roi_mapper"
                            #].get_images(workspace, annotations, roi_data)

                            #if new_rois_images:
                             #   for r_name, r_image in new_rois_images.items():
                            #        orientation_tasks.append(
                             #           (
                             #               r_image,
                             #               detectors[
                              #                  "branch_wrong_orientation_detector"
                              #              ],
                               #             r_name,
                                #            zone_number,
                               #         )
                              #      )
                        #timer.stop("yolo_roi_mapper_total")

                        #if orientation_tasks:
                         #   timer.start("branch_wrong_orientation_detector_total")
                          #  orientation_results = list(
                          #      executor.map(
                         #           _orientation_check_helper, orientation_tasks
                          #      )
                          #  )
                          #  timer.stop("branch_wrong_orientation_detector_total")
#
                          #  for r_name, zone_number, is_wrong in orientation_results:
                          #      if is_wrong:
                          #          error_codes.add(ERROR_CODES["WRONG_ORIENTATION"])
                                # Log individual times? We only have total now.
                                # timer.add(f"orientation_detector_{r_name}_z{zone_number}", ...)

                        timer.stop("total_inspection_time")

                        # Print the report
                        timer.print_report()

                        # Final Result
                        if error_codes:
                            combined_code = "".join(sorted(error_codes))
                            result_queue.put(
                                {
                                    "status": "DONE",
                                    "success": False,
                                    "error": combined_code,
                                }
                            )
                        else:
                            result_queue.put(
                                {"status": "DONE", "success": True, "error": ""}
                            )

                    except Exception as e:
                        traceback.print_exc()
                        result_queue.put(
                            {
                                "status": "DONE",
                                "success": False,
                                "error": f"Worker Error: {str(e)}",
                            }
                        )

    except Exception as e:
        traceback.print_exc()
        print(f"[Worker] Fatal Error: {e}")
