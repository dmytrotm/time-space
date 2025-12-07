from utils import (
    ImageServer,
    WorkspaceExtractor,
    ROICropper,
    Visualizer,
    YOLOROIMapper,
    UIManager,
)
from utils.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    RED,
    ORANGE,
    CONFIG_ROI_Z1_PATH,
    CONFIG_ROI_Z2_PATH,
    CONFIG_POSITIONS_PATH,
    TAPE_DETECTOR_CONF_THRESHOLD,
    TAPE_CLASS_ID,
    LABEL_CLASS_ID,
    TAPE_DEVIATION_TOO_FAR,
    TAPE_DEVIATION_WRONG_LENGTH,
    WORKSPACE_EXTRACTOR_CONFIG,
)
from detectors import (
    GroundingWireDetector,
    TapeDetector,
    TapeDeviationDetector,
    WrongOrientation,
)
import cv2
import json
import time


def load_configurations():
    """Load all configuration files"""
    with open(CONFIG_ROI_Z1_PATH, "r") as f:
        roi_data_z1 = json.load(f)
    with open(CONFIG_ROI_Z2_PATH, "r") as f:
        roi_data_z2 = json.load(f)
    with open(CONFIG_POSITIONS_PATH, "r") as f:
        positions = json.load(f)
    return roi_data_z1, roi_data_z2, positions


def initialize_detectors(roi_data_z1, roi_data_z2, positions):
    """Initialize all detector and processing objects"""
    return {
        "roi_cropper_z1": ROICropper(roi_data_z1),
        "roi_cropper_z2": ROICropper(roi_data_z2),
        "grounding_detector": GroundingWireDetector(),
        "tape_detector": TapeDetector(conf_threshold=TAPE_DETECTOR_CONF_THRESHOLD),
        "tape_deviation_detector": TapeDeviationDetector(positions),
        "yolo_roi_mapper": YOLOROIMapper(),
        "branch_wrong_orientation_detector": WrongOrientation(),
    }


def find_roi_object(roi_cropper, category, roi_id):
    """Find ROI object from cropper"""
    for roi in roi_cropper.roi_objects.get(category.lower(), []):
        if roi.get("id") == roi_id:
            return roi
    return None


def create_verification_function(cameras, extractor, roi_data_z1, roi_data_z2, detectors):
    """
    Create verification function with closure over dependencies.
    Returns a function that executes inspection and returns (bool, str).
    """
    def verification_function():
        """
        Execute inspection cycle.
        Returns (success: bool, error_message: str)
        """
        try:
            timings = {
                "image_capture": 0,
                "grounding_detector_time": 0,
                "tape_detector_time": 0,
                "yolo_roi_mapper_time": 0,
                "branch_wrong_orientation_detector_time": 0,
            }
            
            # Capture images
            start_time = time.time()
            images = cameras.take_photos()
            timings["image_capture"] = time.time() - start_time
            
            if not images:
                return (False, "No images captured")
            
            # Process all zones
            errors = []
            for i, image in enumerate(images):
                zone_number = i + 1
                
                # Extract workspace
                start_time = time.time()
                workspace = extractor.extract_workspace(image)
                timings[f"workspace_extraction_zone_{zone_number}"] = time.time() - start_time
                
                if workspace is None:
                    errors.append(f"Failed to extract workspace for Zone {zone_number}")
                    continue
                
                # Select ROI data and cropper for this zone
                if zone_number == 1:
                    roi_cropper = detectors["roi_cropper_z1"]
                    roi_data = roi_data_z1
                else:
                    roi_cropper = detectors["roi_cropper_z2"]
                    roi_data = roi_data_z2
                
                # Crop ROIs
                start_time = time.time()
                rois = roi_cropper.crop(workspace)
                timings[f"roi_cropping_zone_{zone_number}"] = time.time() - start_time
                
                annotations = {}
                
                # Process all ROIs
                for roi_name, roi_image in rois.items():
                    if roi_image is None or roi_image.size == 0:
                        continue
                    
                    category, roi_id_str = roi_name.split("_")
                    roi_id = int(roi_id_str)
                    
                    roi_object = find_roi_object(roi_cropper, category, roi_id)
                    if roi_object is None:
                        continue
                    
                    # Process based on ROI type
                    if roi_name.startswith("GROUNDING"):
                        start_time = time.time()
                        is_present = detectors["grounding_detector"].is_present(roi_image)
                        timings["grounding_detector_time"] += time.time() - start_time
                        if not is_present:
                            errors.append(f"Zone {zone_number}: Grounding Missing in {roi_name}")
                    
                    elif roi_name.startswith("TAPE"):
                        start_time = time.time()
                        results = detectors["tape_detector"].detect(roi_image)
                        timings["tape_detector_time"] += time.time() - start_time
                        detected_classes = (
                            results[0].boxes.cls.tolist() if results[0].boxes is not None else []
                        )
                        
                        tape_id = int(roi_name.split("_")[-1])
                        annotations[tape_id] = []
                        
                        if TAPE_CLASS_ID not in detected_classes:
                            errors.append(f"Zone {zone_number}: TAPE not detected in {roi_name}")
                            continue
                        
                        for box_data in results[0].boxes:
                            x_center, y_center, width, height = box_data.xywhn[0]
                            annotations[tape_id].append([
                                int(box_data.cls[0]),
                                x_center.item(),
                                y_center.item(),
                                width.item(),
                                height.item(),
                            ])
                            
                            try:
                                index = int(roi_name.split("_")[-1])
                                correct = detectors["tape_deviation_detector"].is_tape_correct(
                                    index, x_center, width
                                )
                                
                                if correct == TAPE_DEVIATION_TOO_FAR:
                                    errors.append(f"Zone {zone_number}: TAPE too far in {roi_name}")
                                elif correct == TAPE_DEVIATION_WRONG_LENGTH:
                                    errors.append(f"Zone {zone_number}: TAPE wrong length in {roi_name}")
                            except (ValueError, IndexError):
                                pass
                    
                    elif roi_name.startswith("LABEL"):
                        start_time = time.time()
                        results = detectors["tape_detector"].detect(roi_image)
                        timings["tape_detector_time"] += time.time() - start_time
                        detected_classes = (
                            results[0].boxes.cls.tolist() if results[0].boxes is not None else []
                        )
                        
                        if LABEL_CLASS_ID not in detected_classes:
                            errors.append(f"Zone {zone_number}: LABEL not detected in {roi_name}")
                
                # Process orientation detection
                start_time = time.time()
                new_rois_images, new_rois_json = detectors["yolo_roi_mapper"].get_images(
                    workspace, annotations, roi_data
                )
                timings["yolo_roi_mapper_time"] += time.time() - start_time
                
                if new_rois_images:
                    for roi_name, roi_image in new_rois_images.items():
                        start_time = time.time()
                        is_wrong_orientation = detectors["branch_wrong_orientation_detector"].detect(
                            roi_image
                        )
                        timings["branch_wrong_orientation_detector_time"] += time.time() - start_time
                        
                        if is_wrong_orientation:
                            roi_id = int(roi_name.split("_")[-1])
                            errors.append(f"Zone {zone_number}: Wrong orientation detected in branch {roi_id}")
            
            # Return success or failure
            if errors:
                # Return first error (or combine them)
                return (False, errors[0])
            else:
                return (True, "")
                
        except Exception as e:
            return (False, f"Exception: {str(e)}")
    
    return verification_function


if __name__ == "__main__":
    """Main entry point"""

    DEFAULT_Z1_IMAGE_PATH = "Z1_0_1.png"
    DEFAULT_Z2_IMAGE_PATH = "Z2_0_1.png"

    try:
        # Load configurations
        roi_data_z1, roi_data_z2, positions = load_configurations()

        # Initialize components
        cameras = ImageServer(DEFAULT_Z1_IMAGE_PATH, DEFAULT_Z2_IMAGE_PATH)
        extractor = WorkspaceExtractor(WORKSPACE_EXTRACTOR_CONFIG)
        detectors = initialize_detectors(roi_data_z1, roi_data_z2, positions)

        # Create verification function
        verification_func = create_verification_function(
            cameras, extractor, roi_data_z1, roi_data_z2, detectors
        )

        # Create and run UI
        ui_manager = UIManager(verification_func, WINDOW_WIDTH, WINDOW_HEIGHT)
        ui_manager.main_loop()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
