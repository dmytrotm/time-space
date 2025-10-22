from utils import ImageServer, WorkspaceExtractor, ROICropper, Visualizer
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
    cameras = ImageServer(
        "dataset/Test_Case7/Z1_0_2.png",
        "dataset/Test_Case7/Z2_0_2.png"
    )
    images = cameras.take_photos()

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
                    roi_cropper = roi_cropper_z1
                    roi_data = roi_data_z1
                else:
                    rois = roi_cropper_z2.crop(workspace)
                    roi_cropper = roi_cropper_z2
                    roi_data = roi_data_z2

                visualizer = Visualizer(workspace)
                # Process and display each ROI
                annotations = {}
                for roi_name, roi_image in rois.items():
                    if roi_image is not None and roi_image.size > 0:
                        category, roi_id_str = roi_name.split('_')
                        roi_id = int(roi_id_str)
                        
                        roi_object = None
                        # Find the roi_object from the roi_cropper
                        for roi in roi_cropper.roi_objects.get(category.lower(), []):
                            if roi.get('id') == roi_id:
                                roi_object = roi
                                break
                        
                        if roi_object is None:
                            logging.error(f"Could not find ROI object for {roi_name}")
                            continue

                        if roi_name.startswith("GROUNDING"):
                            is_present = grounding_detector.is_present(roi_image)
                            if is_present:
                                visualizer.draw_roi(roi_object, Visualizer.GREEN, "Grounding OK")
                            else:
                                visualizer.draw_roi(roi_object, Visualizer.RED, "Grounding Missing")

                        else:
                            # Apply TapeDetector and visualize results
                            results = tape_detector.detect(roi_image)

                            # Check if the correct object is detected
                            detected_classes = results[0].boxes.cls.tolist() if results[0].boxes is not None else []

                            if roi_name.startswith("TAPE"):
                                tape_id = int(roi_name.split('_')[-1])
                                annotations[tape_id] = []
                                if 1 in detected_classes:
                                    for box_data in results[0].boxes:
                                        x_center, y_center, width, height = box_data.xywhn[0]
                                        annotations[tape_id].append([int(box_data.cls[0]), x_center, y_center, width, height])
                                        # Extract index from roi_name (e.g., 'TAPE_1' -> 1)
                                        try:
                                            index = int(roi_name.split('_')[-1])
                                            correct = tape_deviation_detector.is_tape_correct(index, x_center, width)
                                            if correct == -1:
                                                visualizer.draw_roi(roi_object, Visualizer.ORANGE, f"FAIL in {roi_name}: Too far")
                                            elif correct == 1:
                                                visualizer.draw_roi(roi_object, Visualizer.ORANGE, f"FAIL in {roi_name}: Wrong length")
                                            elif correct == 0:
                                                visualizer.draw_roi(roi_object, Visualizer.GREEN, f"Tape OK in {roi_name}")
                                        except (ValueError, IndexError):
                                            logging.error(f"Could not determine tape index from ROI name: {roi_name}")
                                else:
                                    visualizer.draw_roi(roi_object, Visualizer.RED, f"FAIL in {roi_name}: Not detected")

                            elif roi_name.startswith("LABEL"):
                                if 0 in detected_classes:
                                    visualizer.draw_roi(roi_object, Visualizer.GREEN, f"Label OK in {roi_name}")
                                else:
                                    visualizer.draw_roi(roi_object, Visualizer.RED, f"FAIL in {roi_name}: Not detected")
                    else:
                        logging.warning(f"Warning: ROI {roi_name} from Zone {zone_number} is empty or invalid.")

                cv2.imshow(f"Zone {zone_number} Visualizations", visualizer.get_image())

                # Add YOLO mapper to generate new ROIs based on detections
                new_rois = yolo_roi_mapper.get_images(workspace, annotations, roi_data)
                if new_rois:
                    for roi_name, roi_image in new_rois.items():
                        cv2.imshow(f"New ROI {roi_name}", roi_image)

            else:
                logging.error(f"Workspace for Zone {zone_number} could not be extracted.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()