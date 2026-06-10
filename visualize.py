from utils import Visualizer
from processors import ROICropper, WorkspaceExtractor
from core import ImageServer
from configs.config import GREEN
import cv2
import json

if __name__ == "__main__":
    # Load ROI configurations
    with open("configs/rois_z1.json", "r") as f:
        roi_data_z1 = json.load(f)
    with open("configs/rois_z2.json", "r") as f:
        roi_data_z2 = json.load(f)

    # Create instances of the tools
    cameras = ImageServer(
        "dataset/Test_Case2/Z1_0_2.png", "dataset/Test_Case3/Z2_0_2.png"
    )
    results = cameras.take_photos()

    extractor = WorkspaceExtractor()
    roi_cropper_z1 = ROICropper(roi_data_z1)
    roi_cropper_z2 = ROICropper(roi_data_z2)

    if not results:
        print("No images were loaded. Exiting.")
    else:
        for i, (image, zone_id) in enumerate(results):
            print(f"Processing image {i+1}, Detected Zone: {zone_id}")

            workspace = extractor.extract_workspace(image)

            if workspace is not None:
                # Select the correct ROI cropper for the zone
                if zone_id == 1:
                    roi_cropper = roi_cropper_z1
                elif zone_id == 2:
                    roi_cropper = roi_cropper_z2
                else:
                    print(f"Unknown zone {zone_id} for image {i+1}. Skipping ROI visualization.")
                    continue

                # Print available categories for debugging
                print(f"\nZone {zone_id} - Available categories:")
                for category in roi_cropper.roi_objects.keys():
                    count = len(roi_cropper.roi_objects[category])
                    print(f"  - '{category}' ({count} ROIs)")

                visualizer = Visualizer(workspace)

                # Draw only wire ROIs in green
                visualizer.draw_rois_by_category(roi_cropper, "wires", color=GREEN)

                cv2.imshow(f"Zone {zone_id} Visualizations", visualizer.get_image())
            else:
                print(f"Workspace for Zone {zone_id} could not be extracted.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()