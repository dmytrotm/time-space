import cv2
import os
import numpy as np
import pandas as pd
import json
import logging

# Default path to color ranges config
DEFAULT_COLOR_RANGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'configs', 
    'wire_color_ranges.json'
)

class GroundingWireDetector:
    def __init__(self, threshold: float = 0.0005, color_ranges_path=None):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        if color_ranges_path is None:
            color_ranges_path = DEFAULT_COLOR_RANGES_PATH
            
        self.color_ranges = self._load_color_ranges(color_ranges_path)

    def _load_color_ranges(self, path):
        """
        Load color ranges from JSON configuration file.
        """
        try:
            with open(path, 'r') as f:
                ranges_dict = json.load(f)
            
            # Convert lists to numpy arrays
            color_ranges = {}
            for color, ranges_list in ranges_dict.items():
                color_ranges[color] = []
                for range_dict in ranges_list:
                    color_ranges[color].append({
                        'lower': np.array(range_dict['lower']),
                        'upper': np.array(range_dict['upper'])
                    })
            
            return color_ranges
            
        except Exception as e:
            self.logger.error(f"Failed to load color ranges from {path}: {e}")
            return {}

    def percentage_of_grounding_wire(self, image: cv2.UMat, return_mask=False):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        combined_mask = None
        
        # Use 'green-yellow' from config if available, otherwise fallback or error
        if 'green-yellow' in self.color_ranges:
            for range_dict in self.color_ranges['green-yellow']:
                lower = range_dict['lower']
                upper = range_dict['upper']
                mask = cv2.inRange(hsv, lower, upper)

                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
        else:
            # Fallback if config is missing or invalid (though we expect it to be there)
            # This matches the original hardcoded values
            lower_green_yellow1 = np.array([20, 50, 50])
            upper_green_yellow1 = np.array([40, 255, 255])
            lower_green_yellow2 = np.array([40, 50, 50])
            upper_green_yellow2 = np.array([80, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_green_yellow1, upper_green_yellow1)
            mask2 = cv2.inRange(hsv, lower_green_yellow2, upper_green_yellow2)
            combined_mask = cv2.bitwise_or(mask1, mask2)

        if combined_mask is None:
             combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
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
    
    # Added for compatibility with process_folder which calls is_grounding_missing
    def is_grounding_missing(self, image: cv2.UMat):
        return not self.is_present(image)

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