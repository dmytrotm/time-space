import multiprocessing
import time
import json
import traceback
import cv2
from concurrent.futures import ThreadPoolExecutor
from core.timing import TimingLogger
from detectors import (
    GroundingWireDetector,
    TapeDetector,
    TapeDeviationDetector,
    WrongOrientation,
    MissingWiresDetector,
)
from processors import ROICropper, YOLOROIMapper, WorkspaceExtractor
from utils.constants import (
    TAPE_DETECTOR_CONF_THRESHOLD,
    TAPE_CLASS_ID,
    LABEL_CLASS_ID,
    TAPE_DEVIATION_TOO_FAR,
    TAPE_DEVIATION_WRONG_LENGTH,
    WORKSPACE_EXTRACTOR_CONFIG,
    ERROR_CODES,
)

def _preprocess_roi(args):
    """
    Helper function to process a single ROI in a separate thread.
    
    Args:
        args (tuple): (workspace_image, roi_config, roi_name, cropper, target_size)
        
    Returns:
        tuple: (roi_name, processed_image, roi_config)
    """
    workspace_image, roi_config, roi_name, cropper, target_size = args
    
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
            return roi_name, None, roi_config

        # Crop the ROI
        # Note: Numpy slicing creates a view, but since we might resize or 
        # pass to other threads/processes, and we want to release the original 
        # workspace reference eventually, a copy might happen during resize anyway.
        cropped_roi = workspace_image[y1:y2, x1:x2]
        
        if cropped_roi.size == 0:
            return roi_name, None, roi_config
            
        # Resize if needed (Heavy OpenCV operation -> Releases GIL)
        if target_size:
            processed_image = cv2.resize(cropped_roi, target_size)
        else:
            processed_image = cropped_roi
            
        return roi_name, processed_image, roi_config
        
    except Exception as e:
        print(f"Error processing ROI {roi_name}: {e}")
        return roi_name, None, roi_config

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
            
        # Initialize Detectors & Tools
        # Note: Heavy models (YOLO) are loaded here, inside the worker process.
        detectors = {
            "roi_cropper_z1": ROICropper(roi_data_z1),
            "roi_cropper_z2": ROICropper(roi_data_z2),
            "grounding_detector": GroundingWireDetector(),
            "tape_detector": TapeDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD),
            "tape_deviation_detector": TapeDeviationDetector(positions),
            "yolo_roi_mapper": YOLOROIMapper(),
            "branch_wrong_orientation_detector": WrongOrientation(),
            "missing_wires_detector": MissingWiresDetector(),
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
                        
                        # 1. Extract Workspaces
                        workspaces = []
                        for i, img in enumerate(images):
                            zone_number = i + 1
                            timer.start(f"workspace_extraction_zone_{zone_number}")
                            ws = detectors["workspace_extractor"].extract_workspace(img)
                            timer.stop(f"workspace_extraction_zone_{zone_number}")
                            
                            if ws is not None:
                                workspaces.append((zone_number, ws))
                        
                        if not workspaces:
                             result_queue.put({"status": "DONE", "success": False, "error": "No workspace found"})
                             continue

                        # 2. Batch Preparation & Parallel Processing
                        tape_batch_images = []
                        tape_batch_metadata = []
                        
                        roi_map_per_zone = {}
                        
                        for zone_number, workspace in workspaces:
                            if zone_number == 1:
                                cropper = detectors["roi_cropper_z1"]
                                roi_data = roi_data_z1
                            else:
                                cropper = detectors["roi_cropper_z2"]
                                roi_data = roi_data_z2
                            
                            roi_map_per_zone[zone_number] = {"workspace": workspace, "roi_data": roi_data, "cropper": cropper}
                            
                            # Prepare tasks for parallel execution
                            tasks = []
                            
                            # Iterate over all ROI categories and objects to build the task list
                            for category_name, roi_list in cropper.roi_objects.items():
                                for roi_config in roi_list:
                                    roi_id = roi_config.get("id", 0)
                                    # Create unique key
                                    roi_name = f"{category_name.upper()}_{roi_id:03d}"
                                    
                                    # Determine target size
                                    # Tape and Label usually need resizing for YOLO (e.g. 640x640)
                                    # Grounding and Wires might use different logic or raw crops.
                                    # For this optimization, we assume Tape/Label go to 640x640.
                                    target_size = None
                                    if roi_name.startswith("TAPE") or roi_name.startswith("LABEL"):
                                        target_size = (640, 640)
                                    
                                    tasks.append((workspace, roi_config, roi_name, cropper, target_size))

                            timer.start(f"roi_processing_zone_{zone_number}")
                            
                            # Execute tasks in parallel
                            # This distributes the cropping and resizing across 4 cores
                            results = list(executor.map(_preprocess_roi, tasks))
                            
                            timer.stop(f"roi_processing_zone_{zone_number}")
                            
                            # Process results
                            for roi_name, roi_image, roi_config in results:
                                if roi_image is None:
                                    continue
                                    
                                category, roi_id_str = roi_name.split("_")
                                roi_id = int(roi_id_str)
                                
                                if roi_name.startswith("GROUNDING"):
                                    timer.start(f"grounding_detector_{roi_name}_z{zone_number}")
                                    if not detectors["grounding_detector"].is_present(roi_image):
                                        error_codes.add(ERROR_CODES["GROUNDING_MISSING"])
                                    timer.stop(f"grounding_detector_{roi_name}_z{zone_number}")
                                    timer.add("grounding_detector_total", timer.timings.get(f"grounding_detector_{roi_name}_z{zone_number}", 0))
                                        
                                elif roi_name.startswith("WIRES"):
                                    timer.start(f"wires_detector_{roi_name}_z{zone_number}")
                                    # roi_config is passed back, so we can use it directly if needed, 
                                    # but existing logic used find_roi_object. 
                                    # roi_config IS the object from cropper.roi_objects, so we can use it.
                                    expected_colors = roi_config.get("expected_colors", None)
                                    
                                    if zone_number == 2 and "wires" in roi_data_z2:
                                         wires_config = roi_data_z2["wires"]
                                         if isinstance(wires_config, dict) and "color_ranges" in wires_config:
                                             detectors["missing_wires_detector"].set_color_ranges_from_dict(wires_config["color_ranges"])

                                    if not detectors["missing_wires_detector"].is_present(roi_image, expected_colors):
                                        error_codes.add(ERROR_CODES["WIRES_MISSING"])
                                    timer.stop(f"wires_detector_{roi_name}_z{zone_number}")
                                    timer.add("missing_wires_detector_total", timer.timings.get(f"wires_detector_{roi_name}_z{zone_number}", 0))

                                elif roi_name.startswith("TAPE") or roi_name.startswith("LABEL"):
                                    tape_batch_images.append(roi_image)
                                    tape_batch_metadata.append({
                                        "zone": zone_number,
                                        "type": "TAPE" if roi_name.startswith("TAPE") else "LABEL",
                                        "id": roi_id,
                                        "name": roi_name
                                    })

                        # 3. Batch Inference (TapeDetector)
                        if tape_batch_images:
                            timer.start("tape_detector_total")
                            tape_results = detectors["tape_detector"].predict_batch(tape_batch_images)
                            total_tape_time = timer.stop("tape_detector_total")
                            
                            # Attribute time proportionally to each ROI for logging purposes
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
                                
                                detected_classes = result.boxes.cls.tolist() if result.boxes is not None else []
                                
                                if roi_type == "TAPE":
                                    if TAPE_CLASS_ID not in detected_classes:
                                        error_codes.add(ERROR_CODES["TAPE_NOT_DETECTED"])
                                        continue
                                    
                                    if roi_id not in annotations_per_zone[zone]:
                                        annotations_per_zone[zone][roi_id] = []
                                        
                                    for box_data in result.boxes:
                                        x_center, y_center, width, height = box_data.xywhn[0]
                                        annotations_per_zone[zone][roi_id].append([
                                            int(box_data.cls[0]),
                                            x_center.item(),
                                            y_center.item(),
                                            width.item(),
                                            height.item(),
                                        ])
                                        
                                        try:
                                            correct = detectors["tape_deviation_detector"].is_tape_correct(
                                                roi_id, x_center, width
                                            )
                                            if correct == TAPE_DEVIATION_TOO_FAR:
                                                error_codes.add(ERROR_CODES["TAPE_TOO_FAR"])
                                            elif correct == TAPE_DEVIATION_WRONG_LENGTH:
                                                error_codes.add(ERROR_CODES["TAPE_WRONG_LENGTH"])
                                        except (ValueError, IndexError):
                                            pass
                                            
                                elif roi_type == "LABEL":
                                    if LABEL_CLASS_ID not in detected_classes:
                                        error_codes.add(ERROR_CODES["LABEL_NOT_DETECTED"])

                        # 4. Orientation Check
                        for zone_number, data in roi_map_per_zone.items():
                            workspace = data["workspace"]
                            roi_data = data["roi_data"]
                            annotations = annotations_per_zone.get(zone_number, {})
                            
                            timer.start(f"yolo_roi_mapper_zone_{zone_number}")
                            new_rois_images, _ = detectors["yolo_roi_mapper"].get_images(
                                workspace, annotations, roi_data
                            )
                            timer.stop(f"yolo_roi_mapper_zone_{zone_number}")
                            timer.add("yolo_roi_mapper_total", timer.timings.get(f"yolo_roi_mapper_zone_{zone_number}", 0))
                            
                            if new_rois_images:
                                for r_name, r_image in new_rois_images.items():
                                    timer.start(f"orientation_detector_{r_name}_z{zone_number}")
                                    if detectors["branch_wrong_orientation_detector"].detect(r_image):
                                        error_codes.add(ERROR_CODES["WRONG_ORIENTATION"])
                                    timer.stop(f"orientation_detector_{r_name}_z{zone_number}")
                                    timer.add("branch_wrong_orientation_detector_total", timer.timings.get(f"orientation_detector_{r_name}_z{zone_number}", 0))

                        timer.stop("total_inspection_time")
                        
                        # Print the report
                        timer.print_report()

                        # Final Result
                        if error_codes:
                            combined_code = "".join(sorted(error_codes))
                            result_queue.put({"status": "DONE", "success": False, "error": combined_code})
                        else:
                            result_queue.put({"status": "DONE", "success": True, "error": ""})
                            
                    except Exception as e:
                        traceback.print_exc()
                        result_queue.put({"status": "DONE", "success": False, "error": f"Worker Error: {str(e)}"})

    except Exception as e:
        traceback.print_exc()
        print(f"[Worker] Fatal Error: {e}")
