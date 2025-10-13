import cv2
import json
from utils.roi_cropper import ROICropper
from utils.yolo_roi_mapper import YOLOROIMapper

def main():
    # 1. Load original image and ROI data
    original_img_path = "Z2_0_4.png"
    original_image = cv2.imread(original_img_path)
    if original_image is None:
        print(f"Error: Could not load image {original_img_path}")
        return

    positions_json_path = "rois_z2.json"
    try:
        with open(positions_json_path, 'r') as f:
            roi_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ROI data: {e}")
        return

    # 2. Use ROICropper to get cropped images
    try:
        cropper = ROICropper(roi_data)
        cropped_images = cropper.crop(original_image)

        # Display the cropped images
        print(f"Successfully cropped {len(cropped_images)} ROIs.")
        for key, cropped_img in cropped_images.items():
            cv2.imshow(f"Cropped ROI: {key}", cropped_img)
        print("Press any key to close the cropped ROI windows.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during cropping: {e}")
        return

    # 3. Mocked YOLO annotations for the detected objects in the ROIs
    # In a real scenario, you would run an object detection model on the `cropped_images`
    multiple_tapes_annotations = {
        1: [[float(x) for x in "1 0.45625 0.48203125 0.28515625 0.1203125".split()]],
        2: [[float(x) for x in "1 0.34375 0.48671875 0.23984375 0.1265625".split()]],
        3: [[float(x) for x in "1 0.33671875 0.49375 0.25390625 0.125".split()]],
        4: [[float(x) for x in "1 0.3390625 0.4203125 0.26640625 0.10703125".split()]],
        5: [],
    }

    # 4. Use YOLOROIMapper to map annotations and visualize
    class_names = ["label", "tape"]
    mapper = YOLOROIMapper(class_names)

    result_json = mapper.process_roi_mapping(
        original_img_path=original_img_path,
        roi_annotations=multiple_tapes_annotations,
        positions_json_path=positions_json_path,
    )

    if result_json and result_json["orientation"]:
        print(f"Created JSON with {len(result_json['orientation'])} ROI structures")
        print(json.dumps(result_json, indent=2))
        
        # Visualize all ROIs on the original image
        mapper.visualize_rois_on_image(
            original_img_path,
            result_json,
            positions_json_path,
            multiple_tapes_annotations
        )
    else:
        print("No ROI structures were created")
        # Still visualize the old ROIs and detected objects
        mapper.visualize_rois_on_image(
            original_img_path,
            {"orientation": []},
            positions_json_path,
            multiple_tapes_annotations
        )

if __name__ == "__main__":
    main()
