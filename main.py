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
    LOG_SEPARATOR,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    WORKSPACE_EXTRACTOR_CONFIG,
    KEY_MAPPING,
)
from detectors import (
    GroundingWireDetector,
    TapeDetector,
    TapeDeviationDetector,
    WrongOrientation,
)
import cv2
import json
import logging


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


def process_grounding_roi(roi_image, roi_object, grounding_detector, visualizer):
    """Process grounding wire detection"""
    if not grounding_detector.is_present(roi_image):
        visualizer.draw_roi(roi_object, RED, "Grounding Missing")


def process_tape_roi(
    roi_name,
    roi_image,
    roi_object,
    tape_detector,
    tape_deviation_detector,
    visualizer,
    annotations,
):
    """Process tape detection and deviation check"""
    results = tape_detector.detect(roi_image)
    detected_classes = (
        results[0].boxes.cls.tolist() if results[0].boxes is not None else []
    )

    tape_id = int(roi_name.split("_")[-1])
    annotations[tape_id] = []

    if TAPE_CLASS_ID not in detected_classes:
        visualizer.draw_roi(roi_object, RED, f"FAIL in {roi_name}: Not detected")
        return

    for box_data in results[0].boxes:
        x_center, y_center, width, height = box_data.xywhn[0]
        annotations[tape_id].append(
            [
                int(box_data.cls[0]),
                x_center.item(),
                y_center.item(),
                width.item(),
                height.item(),
            ]
        )

        try:
            index = int(roi_name.split("_")[-1])
            correct = tape_deviation_detector.is_tape_correct(index, x_center, width)

            if correct == TAPE_DEVIATION_TOO_FAR:
                visualizer.draw_roi(roi_object, ORANGE, f"FAIL in {roi_name}: Too far")
            elif correct == TAPE_DEVIATION_WRONG_LENGTH:
                visualizer.draw_roi(
                    roi_object, ORANGE, f"FAIL in {roi_name}: Wrong length"
                )
        except (ValueError, IndexError):
            logging.error(f"Could not determine tape index from ROI name: {roi_name}")


def process_label_roi(roi_name, roi_image, roi_object, tape_detector, visualizer):
    """Process label detection"""
    results = tape_detector.detect(roi_image)
    detected_classes = (
        results[0].boxes.cls.tolist() if results[0].boxes is not None else []
    )

    if LABEL_CLASS_ID not in detected_classes:
        visualizer.draw_roi(roi_object, RED, f"FAIL in {roi_name}: Not detected")


def process_single_roi(
    roi_name, roi_image, roi_cropper, detectors, visualizer, annotations
):
    """Process a single ROI based on its type"""
    if roi_image is None or roi_image.size == 0:
        logging.warning(f"Warning: ROI {roi_name} is empty or invalid.")
        return

    category, roi_id_str = roi_name.split("_")
    roi_id = int(roi_id_str)

    roi_object = find_roi_object(roi_cropper, category, roi_id)
    if roi_object is None:
        logging.error(f"Could not find ROI object for {roi_name}")
        return

    if roi_name.startswith("GROUNDING"):
        process_grounding_roi(
            roi_image, roi_object, detectors["grounding_detector"], visualizer
        )
    elif roi_name.startswith("TAPE"):
        process_tape_roi(
            roi_name,
            roi_image,
            roi_object,
            detectors["tape_detector"],
            detectors["tape_deviation_detector"],
            visualizer,
            annotations,
        )
    elif roi_name.startswith("LABEL"):
        process_label_roi(
            roi_name, roi_image, roi_object, detectors["tape_detector"], visualizer
        )


def process_orientation_detection(
    workspace, annotations, roi_data, detectors, visualizer
):
    """Process orientation detection for branches"""
    new_rois_images, new_rois_json = detectors["yolo_roi_mapper"].get_images(
        workspace, annotations, roi_data
    )

    if not new_rois_images:
        return

    for roi_name, roi_image in new_rois_images.items():
        is_wrong_orientation = detectors["branch_wrong_orientation_detector"].detect(
            roi_image
        )
        roi_id = int(roi_name.split("_")[-1])
        new_roi_data = next(
            (roi for roi in new_rois_json["orientation"] if roi["id"] == roi_id), None
        )

        if is_wrong_orientation:
            visualizer.draw_roi_center(new_roi_data, RED, f"Wrong Orientation {roi_id}")


def process_zone(image, zone_number, extractor, roi_data_z1, roi_data_z2, detectors):
    """Process a single zone (Z1 or Z2) and return the visualization image"""
    workspace = extractor.extract_workspace(image)
    if workspace is None:
        logging.error(f"Workspace for Zone {zone_number} could not be extracted.")
        return None

    # Select ROI data and cropper for this zone
    if zone_number == 1:
        roi_cropper = detectors["roi_cropper_z1"]
        roi_data = roi_data_z1
    else:
        roi_cropper = detectors["roi_cropper_z2"]
        roi_data = roi_data_z2

    rois = roi_cropper.crop(workspace)
    visualizer = Visualizer(workspace)
    annotations = {}

    # Process all ROIs
    for roi_name, roi_image in rois.items():
        process_single_roi(
            roi_name, roi_image, roi_cropper, detectors, visualizer, annotations
        )

    # Process orientation detection
    process_orientation_detection(
        workspace, annotations, roi_data, detectors, visualizer
    )

    return visualizer.get_image()


def run_inspection_cycle(
    cameras, extractor, roi_data_z1, roi_data_z2, detectors, ui_manager
):
    """Run a single inspection cycle with step-by-step visualization"""
    logging.info(LOG_SEPARATOR)
    logging.info("Starting new inspection cycle...")
    logging.info(LOG_SEPARATOR)

    ui_manager.hide_instruction_window()

    images = cameras.take_photos()
    if not images:
        logging.warning("No images were loaded. Returning to idle state.")
        ui_manager.show_main_instructions()
        return "continue"

    ui_manager.show_loading_screen()

    # Process all zones first
    zone_images = []
    for i, image in enumerate(images):
        zone_number = i + 1
        logging.info(f"Processing Zone {zone_number}...")
        zone_viz = process_zone(
            image, zone_number, extractor, roi_data_z1, roi_data_z2, detectors
        )
        if zone_viz is not None:
            zone_images.append((zone_number, zone_viz))
            logging.info(f"Zone {zone_number} processed successfully")
        else:
            logging.warning(f"Zone {zone_number} processing failed")

    ui_manager.hide_loading_screen()

    if not zone_images:
        logging.error("No zones were successfully processed.")
        ui_manager.show_main_instructions()
        return "continue"

    logging.info(f"All zones processed. Total zones: {len(zone_images)}")

    current_index = 0

    while True:
        zone_num, zone_img = zone_images[current_index]
        ui_manager.show_zone_visualization(zone_num, zone_img, len(zone_images))

        action = ui_manager.wait_for_action()

        if action == "exit":
            logging.info("Exit requested during inspection.")
            ui_manager.cleanup()
            ui_manager.show_main_instructions()
            return "exit"
        elif action == "finish":
            logging.info("Inspection cycle completed.")
            ui_manager.cleanup()
            ui_manager.show_main_instructions()
            return "continue"
        elif action == "next":
            current_index = (current_index + 1) % len(zone_images)
            logging.info(f"Moving to Zone {zone_images[current_index][0]}")
        elif action == "previous":
            current_index = (current_index - 1) % len(zone_images)
            logging.info(f"Moving to Zone {zone_images[current_index][0]}")


if __name__ == "__main__":
    """Main entry point"""

    DEFAULT_Z1_IMAGE_PATH = "Z1_0_1.png"
    DEFAULT_Z2_IMAGE_PATH = "Z2_0_1.png"

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    logging.info(LOG_SEPARATOR)
    logging.info("Starting Inspection System")
    logging.info(LOG_SEPARATOR)

    try:
        # Load configurations
        logging.info("Loading configurations...")
        roi_data_z1, roi_data_z2, positions = load_configurations()
        logging.info("Configurations loaded successfully")

        # Initialize UI Manager
        logging.info("Initializing UI Manager...")
        ui_manager = UIManager(WINDOW_WIDTH, WINDOW_HEIGHT, KEY_MAPPING)
        logging.info("UI Manager ready")

        # Initialize components
        logging.info("Initializing image server...")
        cameras = ImageServer(DEFAULT_Z1_IMAGE_PATH, DEFAULT_Z2_IMAGE_PATH)
        logging.info("Image server initialized")

        logging.info("Initializing workspace extractor...")
        extractor = WorkspaceExtractor(WORKSPACE_EXTRACTOR_CONFIG)
        logging.info("Workspace extractor initialized")

        logging.info("Initializing detectors...")
        detectors = initialize_detectors(roi_data_z1, roi_data_z2, positions)
        logging.info("All detectors initialized successfully")

        logging.info(LOG_SEPARATOR)
        logging.info("System ready! Press ENTER to start inspection or ESC to exit")
        logging.info(LOG_SEPARATOR)

        # Main processing loop
        while True:
            action = ui_manager.wait_for_action()

            if action == "exit":
                logging.info("Exit command received. Shutting down...")
                break

            if action == "start":
                result = run_inspection_cycle(
                    cameras,
                    extractor,
                    roi_data_z1,
                    roi_data_z2,
                    detectors,
                    ui_manager,
                )
                if result == "exit":
                    logging.info("Exit requested during cycle. Shutting down...")
                    break

    except KeyboardInterrupt:
        logging.info("\nKeyboard interrupt received. Shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Cleanup
        logging.info("Cleaning up...")
        ui_manager.cleanup()
        logging.info("Shutdown complete")
        logging.info(LOG_SEPARATOR)
