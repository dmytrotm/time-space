from utils import TakePhotos, WorkspaceExtractor, ROICropper
from detectors import GroundingWireDetector, TapeDetector, TapeDeviationDetector
from utils.yolo_roi_mapper import YOLOROIMapper
import cv2
import json
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load ROI configurations
    with open("configs/rois_z1.json", 'r') as f:
        roi_data_z1 = json.load(f)
    with open("configs/rois_z2.json", 'r') as f:
        roi_data_z2 = json.load(f)
    with open("configs/positions.json", 'r') as f:
        positions = json.load(f)

    # Create instances of the tools
    cameras = TakePhotos(
        "dataset/Test_Case4/Z1_0_4.png",
        "dataset/Test_Case4/Z2_0_4.png"
    )
    images = cameras.serve_photos()

    extractor = WorkspaceExtractor("configs/custom_markers.yaml")
    roi_cropper_z1 = ROICropper(roi_data_z1)
    roi_cropper_z2 = ROICropper(roi_data_z2)
    grounding_detector = GroundingWireDetector()
    tape_detector = TapeDetector(conf_threshold=0.8)
    tape_deviation_detector = TapeDeviationDetector(positions)
    yolo_roi_mapper = YOLOROIMapper()

    if not images:
        logging.info("No images were loaded. Exiting.")
    else:
        logging.info("Displaying images. Press any key to close all windows.")
        for i, image in enumerate(images):
            zone_number = i + 1
            
            workspace = extractor.extract_workspace(image)
            
            if workspace is not None:
                # cv2.imshow(f"Workspace Zone {zone_number}", workspace)

                # Select the correct ROI cropper for the zone
                if zone_number == 1:
                    rois = roi_cropper_z1.crop(workspace)
                else:
                    rois = roi_cropper_z2.crop(workspace)

                # Process and display each ROI
                annotations = {}
                for roi_name, roi_image in rois.items():
                    if roi_image is not None and roi_image.size > 0:
                        # Special handling for wires with id 1 in zone 2
                        if roi_name.startswith("GROUNDING"):
                            is_present = grounding_detector.is_present(roi_image) 
                            if is_present:
                                logging.info(f"\033[92mGrounding wire is present\033[0m")
                            else:
                                logging.warning(f"\033[91mGrounding wire is missing\033[0m")
                        
                        else:
                            # Apply TapeDetector and visualize results
                            results = tape_detector.detect(roi_image)
                            annotated_roi = results[0].plot()
                            # cv2.imshow(f"Tape Detections in {roi_name}", annotated_roi)

                            # Check if the correct object is detected
                            detected_classes = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
                            
                            if roi_name.startswith("TAPE"):
                                tape_id = int(roi_name.split('_')[-1])
                                annotations[tape_id] = []
                                if 1 in detected_classes:
                                    logging.info(f"\033[92mOK: Tape detected in {roi_name}\033[0m")
                                    for box in results[0].boxes:
                                        x_center, y_center, width, height = box.xywhn[0]
                                        annotations[tape_id].append([int(box.cls[0]), x_center, y_center, width, height])
                                        # Extract index from roi_name (e.g., 'TAPE_1' -> 1)
                                        try:
                                            index = int(roi_name.split('_')[-1])
                                            correct = tape_deviation_detector.is_tape_correct(index, x_center, width)
                                            if correct == -1:
                                                logging.warning(f"\033[91m    -- FAIL: Tape is too far\033[0m")
                                            elif correct == 1:
                                                logging.warning(f"\033[91m    -- FAIL: Tape is too long or too short\033[0m")
                                            elif correct == 0:
                                                logging.info(f"\033[92m    -- OK: Tape fine\033[0m")
                                        except (ValueError, IndexError):
                                            logging.error(f"\033[91mCould not determine tape index from ROI name: {roi_name}\033[0m")
                                else:
                                    logging.warning(f"\033[91mFAIL: Tape NOT detected in {roi_name}\033[0m")
                                    
                            elif roi_name.startswith("LABEL"):
                                if 0 in detected_classes:
                                    logging.info(f"\033[92mOK: Label detected in {roi_name}\033[0m")
                                else:
                                    logging.warning(f"\033[91mFAIL: Label NOT detected in {roi_name}\033[0m")
                    else:
                        logging.warning(f"Warning: ROI {roi_name} from Zone {zone_number} is empty or invalid.")

                if zone_number == 1:
                    new_rois = yolo_roi_mapper.get_images(workspace, annotations, roi_data_z1)
                else:
                    new_rois = yolo_roi_mapper.get_images(workspace, annotations, roi_data_z2)
                for roi_name, roi_image in new_rois.items():
                    cv2.imshow(f"New ROI {roi_name}", roi_image)

            else:
                logging.warning(f"Warning: Workspace for Zone {zone_number} could not be extracted.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()