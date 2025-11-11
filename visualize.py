from utils import ImageServer, WorkspaceExtractor, ROICropper, Visualizer
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
        "dataset/Test_Case3/Z1_0_2.png", "dataset/Test_Case3/Z2_0_2.png"
    )
    images = cameras.take_photos()

    extractor = WorkspaceExtractor("configs/custom_markers.yaml")
    roi_cropper_z1 = ROICropper(roi_data_z1)
    roi_cropper_z2 = ROICropper(roi_data_z2)

    if not images:
        print("No images were loaded. Exiting.")
    else:
        for i, image in enumerate(images):
            zone_number = i + 1

            workspace = extractor.extract_workspace(image)

            if workspace is not None:
                # Select the correct ROI cropper for the zone
                roi_cropper = roi_cropper_z1 if zone_number == 1 else roi_cropper_z2

                visualizer = Visualizer(workspace)

                # Draw all ROIs from the cropper in green
                visualizer.draw_rois(roi_cropper, color=Visualizer.GREEN)

                cv2.imshow(f"Zone {zone_number} Visualizations", visualizer.get_image())
            else:
                print(f"Workspace for Zone {zone_number} could not be extracted.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()