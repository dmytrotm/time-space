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
    MissingWiresDetector,
    
    MultiClassYoloDetector,
    MultiClassHailoDetector
)
from processors import ROICropper, YOLOROIMapper, WorkspaceExtractor, aruco_factory
from configs.config import (
    TAPE_DETECTOR_CONF_THRESHOLD,
    TAPE_CLASS_ID,
    LABEL_CLASS_ID,
    CONNECTOR_CLASS_ID,
    TAPE_DEVIATION_TOO_FAR,
    TAPE_DEVIATION_WRONG_LENGTH,
    WORKSPACE_EXTRACTOR_CONFIG,
    ERROR_CODES,
    ZONES_DICT_WS1,
    ZONES_DICT_WS2
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
        height, width = workspace_image.shape[:2]
        x1, y1, x2, y2 = cropper.calculate_roi_bounds(width, height, roi_config)

        # Clamp to image boundaries
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return roi_name, None, roi_config, zone_number

        cropped_roi = workspace_image[y1:y2, x1:x2]

        if cropped_roi.size == 0:
            return roi_name, None, roi_config, zone_number

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
    Args: (expected_zone, image, extractor)
    """
    expected_zone, image, extractor = args
    try:
        zone_id, ws_img = extractor.extract_workspace(image)
        return expected_zone, zone_id, ws_img
    except Exception as e:
        print(f"Error extracting workspace: {e}")
        return expected_zone, -1, None


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
        print("[Worker] Initializing...")

        with open(config_paths["roi_z1"], "r") as f:
            roi_data_z1 = json.load(f)
        with open(config_paths["roi_z2"], "r") as f:
            roi_data_z2 = json.load(f)
        with open(config_paths["positions"], "r") as f:
            positions = json.load(f)
        with open(config_paths["env"], "r") as f:
            env = json.load(f)    

        yolo_detectors = MultiClassHailoDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD) if env["use_hailo"] else MultiClassYoloDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD)
        #tape_detector = TapeDetectorHailo(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD) if env["use_hailo"] else TapeDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD)

        detectors = {
            "roi_cropper_z1": ROICropper(roi_data_z1),
            "roi_cropper_z2": ROICropper(roi_data_z2),
            "grounding_detector": GroundingWireDetector(),
            "yolo_detector": yolo_detectors,
            "tape_deviation_detector": TapeDeviationDetector(positions),
            "missing_wires_detector": MissingWiresDetector(),
            "workspace_extractor_ws1": WorkspaceExtractor(aruco_factory(), ZONES_DICT_WS1),
            "workspace_extractor_ws2": WorkspaceExtractor(aruco_factory(), ZONES_DICT_WS2),
        }

        def find_roi_object(cropper, category, roi_id):
            for roi in cropper.roi_objects.get(category.lower(), []):
                if roi.get("id") == roi_id:
                    return roi
            return None

        print("[Worker] Initialization complete. Ready.")


        with ThreadPoolExecutor(max_workers=4) as executor:


            while True:
                command_data = command_queue.get()

                if command_data["command"] == "STOP":
                    print("[Worker] Stopping...")
                    break

                elif command_data["command"] == "TRIGGER":
                    images = command_data["images"]
                    workspace_id = command_data.get("workspace_id", 1)
                    save_errors = command_data.get("save_errors", False)

                    try:
                        timer = TimingLogger()
                        error_codes = set()
                        error_images = {} if save_errors else None

                        timer.start("total_inspection_time")

                        ws_tasks = []
                        ext_key = f"workspace_extractor_ws{workspace_id}"
                        extractor = detectors.get(ext_key, detectors["workspace_extractor_ws1"])
                        
                        for i, img in enumerate(images):
                            expected_zone = i + 1
                            ws_tasks.append(
                                (expected_zone, img, extractor)
                            )

                        timer.start("workspace_extraction_total")
                        # _extract_workspace_helper returns (expected_zone, actual_zone_id, image)
                        ws_results = list(
                            executor.map(_extract_workspace_helper, ws_tasks)
                        )
                        timer.stop("workspace_extraction_total")

                        workspaces = []
                        missing_zones = []
                        for expected_zone, actual_zone, ws in ws_results:
                            if ws is None or actual_zone == -1:
                                missing_zones.append(expected_zone)
                            else:
                                workspaces.append((actual_zone, ws))
                            
                        if missing_zones:
                            zones_str = " and ".join(str(z) for z in missing_zones)
                            
                            result_data = {
                                "status": "DONE",
                                "workspace_id": workspace_id,
                                "success": False,
                                "error": f"No workspace found\nfor Zone {zones_str}",
                            }
                            
                            if save_errors:
                                err_images = {}
                                for expected_zone, _, img in ws_results:
                                    if expected_zone in missing_zones and img is not None:
                                        err_images[f"zone{expected_zone}_failed_crop"] = img
                                for expected_zone, raw_img, _ in ws_tasks:
                                    if expected_zone in missing_zones and raw_img is not None:
                                        err_images[f"zone{expected_zone}_raw"] = raw_img
                                result_data["error_images"] = err_images

                            result_queue.put(result_data)
                            continue

                        if len(workspaces) != 2:
                            result_queue.put(
                                {
                                    "status": "DONE",
                                    "workspace_id": workspace_id,
                                    "success": False,
                                    "error": "Failed to extract\nboth zones",
                                }
                            )
                            continue

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

                            for category_name, roi_list in cropper.roi_objects.items():
                                for roi_config in roi_list:
                                    roi_id = roi_config.get("id", 0)
                                    roi_name = f"{category_name.upper()}_{roi_id:03d}"

                                    target_size = None
#TODO Set the right size
                                    if roi_name.startswith(
                                        "TAPE"
                                    ) or roi_name.startswith("LABEL"):
                                        target_size = (576, 576)
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

                        roi_results = list(
                            executor.map(_preprocess_roi, roi_processing_tasks)
                        )

                        timer.stop("roi_processing_total")

                        for result in roi_results:
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
                                    if save_errors:
                                        error_images[roi_name] = roi_image
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
                            #TODO Make it with a model
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
                                    error_codes.add(ERROR_CODES["CONNECTOR_MISSING"])
                                    if save_errors:
                                        error_images[roi_name] = roi_image

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

                        if tape_batch_images:
                            timer.start("tape_detector_total")
                            tape_results = detectors["yolo_detector"].predict_batch(
                                tape_batch_images, tape_batch_metadata
                            )
                            total_tape_time = timer.stop("tape_detector_total")

                            avg_time = total_tape_time / len(tape_batch_images)
                            for meta in tape_batch_metadata:
                                roi_name = meta["name"]
                                zone = meta["zone"]
                                timer.add(f"tape_detector_{roi_name}_z{zone}", avg_time)

                            for result, meta, roi_image in zip(tape_results, tape_batch_metadata, tape_batch_images):
                                zone = meta["zone"]
                                roi_type = meta["type"]
                                roi_id = meta["id"]
                                roi_name = meta["name"]

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
                                        if save_errors:
                                            error_images[roi_name] = roi_image

                                        continue

                                    
                                    for box_data in result.boxes:
                                        x_center, y_center, width, height = (
                                            box_data.xywhn[0]
                                        )

                                        try:
                                            correct = detectors[
                                                "tape_deviation_detector"
                                            ].is_tape_correct(roi_id, x_center, width)
                                            if correct == TAPE_DEVIATION_TOO_FAR:
                                                error_codes.add(
                                                    ERROR_CODES["TAPE_TOO_FAR"]
                                                )
                                                if save_errors:
                                                    error_images[roi_name] = roi_image

                                            elif correct == TAPE_DEVIATION_WRONG_LENGTH:
                                                error_codes.add(
                                                    ERROR_CODES["TAPE_WRONG_LENGTH"]
                                                )
                                                if save_errors:
                                                    error_images[roi_name] = roi_image

                                        except (ValueError, IndexError):
                                            pass

                                elif roi_type == "LABEL":
                                    if LABEL_CLASS_ID not in detected_classes:
                                        error_codes.add(
                                            ERROR_CODES["LABEL_NOT_DETECTED"]
                                        )
                                        if save_errors:
                                            error_images[roi_name] = roi_image

                                elif roi_type == "CONNECTORS":
                                    if CONNECTOR_CLASS_ID not in detected_classes:
                                        error_codes.add(
                                            ERROR_CODES["CONNECTOR_MISSING"]
                                        )
                                        if save_errors:
                                            error_images[roi_name] = roi_image

                        
                        timer.stop("total_inspection_time")

                        timer.print_report()

                        if error_codes:
                            combined_code = "".join(sorted(error_codes))
                            result_data = {
                                "status": "DONE",
                                "workspace_id": workspace_id,
                                "success": False,
                                "error": combined_code,
                            }
                            if save_errors and error_images:
                                result_data["error_images"] = error_images
                            result_queue.put(result_data)
                            
                        else:
                            result_queue.put(
                                {"status": "DONE", "workspace_id": workspace_id, "success": True, "error": ""}
                            )

                    except Exception as e:
                        traceback.print_exc()
                        result_queue.put(
                            {
                                "status": "DONE",
                                "workspace_id": workspace_id,
                                "success": False,
                                "error": f"Worker Error: {str(e)}",
                            }
                        )

    except Exception as e:
        traceback.print_exc()
        print(f"[Worker] Fatal Error: {e}")
