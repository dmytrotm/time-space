import multiprocessing
import time
import json
import traceback
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
        
        # --- Loop Phase ---
        while True:
            command_data = command_queue.get()
            
            if command_data["command"] == "STOP":
                print("[Worker] Stopping...")
                break
            
            elif command_data["command"] == "TRIGGER":
                images = command_data["images"] # Expecting list of numpy arrays or paths? 
                # Ideally paths to avoid pickling huge arrays, but for now let's assume 
                # the ImageServer in main process might pass arrays. 
                # If passing arrays via Queue is too slow, we should pass paths or shared memory.
                # For this refactor, let's assume we receive the images directly or paths.
                # Given the prompt implies "ImageServer" takes photos, let's assume we get the actual image data 
                # OR we should move ImageServer to worker? 
                # The prompt says: "The Manager... spawns... worker... Data Structures... Use Queue for sending commands (e.g. 'TRIGGER', 'image_path'...)"
                # So we should probably pass paths or handle image capture in main and pass data.
                # Let's support passing the image objects for now as per the existing flow, 
                # but be aware of pickling overhead. If `images` are numpy arrays, they are picklable.
                
                # Wait, the prompt example says: `{"command": "TRIGGER", "image_path": "..."}`
                # But `ImageServer` in `main.py` returns numpy images. 
                # To follow the prompt strictly and optimize, we should probably pass paths if they are saved, 
                # OR just pass the numpy arrays if they are in memory. 
                # Let's stick to the existing `main.py` flow where `cameras.take_photos()` returns images.
                # We will assume `images` in command_data are the list of numpy images.
                
                try:
                    start_total = time.time()
                    error_codes = set()
                    
                    # 1. Extract Workspaces (Batching per zone if possible, but usually sequential per image)
                    workspaces = []
                    for i, img in enumerate(images):
                        ws = detectors["workspace_extractor"].extract_workspace(img)
                        if ws is not None:
                            workspaces.append((i + 1, ws)) # (zone_number, workspace_image)
                    
                    if not workspaces:
                         result_queue.put({"status": "DONE", "success": False, "error": "No workspace found"})
                         continue

                    # 2. Batch Preparation
                    # We will collect all ROIs that need detection into batches
                    
                    # Batches for TapeDetector (Tape, Label)
                    tape_batch_images = []
                    tape_batch_metadata = [] # Stores (zone_number, roi_type, roi_id, specific_roi_name)
                    
                    # Data for other detectors (Grounding, Wires - currently CPU/OpenCV based, maybe not full batching yet)
                    # But we can still structure it.
                    
                    # Data for Orientation (needs YOLO ROI Mapper first)
                    
                    # First pass: Crop and collect
                    roi_map_per_zone = {} # Store cropped ROIs for later use (e.g. orientation)
                    
                    for zone_number, workspace in workspaces:
                        if zone_number == 1:
                            cropper = detectors["roi_cropper_z1"]
                            roi_data = roi_data_z1
                        else:
                            cropper = detectors["roi_cropper_z2"]
                            roi_data = roi_data_z2
                            
                        rois = cropper.crop(workspace)
                        roi_map_per_zone[zone_number] = {"workspace": workspace, "rois": rois, "roi_data": roi_data, "cropper": cropper}
                        
                        for roi_name, roi_image in rois.items():
                            if roi_image is None or roi_image.size == 0:
                                continue
                                
                            category, roi_id_str = roi_name.split("_")
                            roi_id = int(roi_id_str)
                            
                            # Grounding - CPU/OpenCV (Fast enough to do inline or collect?)
                            # Let's do inline for now as it's not a heavy DL model
                            if roi_name.startswith("GROUNDING"):
                                if not detectors["grounding_detector"].is_present(roi_image):
                                    error_codes.add(ERROR_CODES["GROUNDING_MISSING"])
                                    
                            # Wires - CPU/OpenCV
                            elif roi_name.startswith("WIRES"):
                                roi_obj = find_roi_object(cropper, category, roi_id)
                                expected_colors = roi_obj.get("expected_colors", None) if roi_obj else None
                                
                                # Update color ranges if needed (optimization: do this once or check if changed)
                                if zone_number == 2 and "wires" in roi_data_z2:
                                     wires_config = roi_data_z2["wires"]
                                     if isinstance(wires_config, dict) and "color_ranges" in wires_config:
                                         detectors["missing_wires_detector"].set_color_ranges_from_dict(wires_config["color_ranges"])

                                if not detectors["missing_wires_detector"].is_present(roi_image, expected_colors):
                                    error_codes.add(ERROR_CODES["WIRES_MISSING"])

                            # Tape & Label - Collect for Batch Inference
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
                        # predict_batch should return a list of results
                        tape_results = detectors["tape_detector"].predict_batch(tape_batch_images)
                        
                        # Process results
                        annotations_per_zone = {z: {} for z, _ in workspaces}
                        
                        for result, meta in zip(tape_results, tape_batch_metadata):
                            zone = meta["zone"]
                            roi_type = meta["type"]
                            roi_id = meta["id"]
                            roi_name = meta["name"]
                            
                            detected_classes = result.boxes.cls.tolist() if result.boxes is not None else []
                            
                            if roi_type == "TAPE":
                                if TAPE_CLASS_ID not in detected_classes:
                                    error_codes.add(ERROR_CODES["TAPE_NOT_DETECTED"])
                                    continue
                                
                                # Collect annotations for Orientation check later
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
                                    
                                    # Deviation Check
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

                    # 4. Orientation Check (Needs results from Tape Detection)
                    # This part is a bit complex because `yolo_roi_mapper` needs the full workspace and annotations
                    # to re-crop or analyze orientation.
                    
                    for zone_number, data in roi_map_per_zone.items():
                        workspace = data["workspace"]
                        roi_data = data["roi_data"]
                        annotations = annotations_per_zone.get(zone_number, {})
                        
                        # Get images for orientation check
                        new_rois_images, _ = detectors["yolo_roi_mapper"].get_images(
                            workspace, annotations, roi_data
                        )
                        
                        if new_rois_images:
                            # Batch these too if possible, or loop
                            # WrongOrientation detector is likely a simple classifier or logic
                            for r_name, r_image in new_rois_images.items():
                                if detectors["branch_wrong_orientation_detector"].detect(r_image):
                                    error_codes.add(ERROR_CODES["WRONG_ORIENTATION"])

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
