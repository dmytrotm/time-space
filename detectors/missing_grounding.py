import cv2
import os
import numpy as np
import pandas as pd


class GroundingWireDetector:
    def __init__(self, threshold: float = 0.0005):
        self.threshold = threshold

    def percentage_of_grounding_wire(self, image: cv2.UMat, return_mask=False):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green_yellow1 = np.array([20, 50, 50])
        upper_green_yellow1 = np.array([40, 255, 255])

        lower_green_yellow2 = np.array([40, 50, 50])
        upper_green_yellow2 = np.array([80, 255, 255])

        mask1 = cv2.inRange(hsv, lower_green_yellow1, upper_green_yellow1)
        mask2 = cv2.inRange(hsv, lower_green_yellow2, upper_green_yellow2)

        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        total_pixels = mask.shape[0] * mask.shape[1]
        white_pixels = np.sum(mask) / 255

        percentage = white_pixels / total_pixels

        if return_mask:
            return (mask, percentage)
        else:
            return percentage

    def is_present(self, image: cv2.UMat):
        p = self.percentage_of_grounding_wire(image)
        return p > self.threshold

    def visualize_detection(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return

        mask, percentage = self.percentage_of_grounding_wire(img, return_mask=True)

        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
        overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)

        output_path = image_path.replace(".", "_detection.")
        cv2.imwrite(output_path, overlay)


def process_folder(
    folder_path: str, output_csv: str = "result.csv", threshold: float = 0.0005
):
    detector = GroundingWireDetector(threshold=threshold)
    result_list = []

    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' not found!")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(folder_path, filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not load image: {img_path}")
                continue

            p = detector.percentage_of_grounding_wire(img)
            is_missing = detector.is_grounding_missing(img)

            result_list.append(
                {"Filename": filename, "%": p, "Result": int(is_missing)}
            )

    if not result_list:
        return

    result = pd.DataFrame(result_list)

    result.to_csv(output_csv, index=False)

    sample_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:3]
    for sample_file in sample_files:
        detector.visualize_detection(os.path.join(folder_path, sample_file))


if __name__ == "__main__":
    folder = "all_rois"
    process_folder(folder, threshold=0.0005)