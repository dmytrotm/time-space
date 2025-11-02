"""
Missing Wires Detector

Детектор відсутніх дротів на основі HSV кольорової сегментації.
Виявляє чорні, сині та коричневі дроти.
"""

import cv2
import numpy as np
import os
import logging

__all__ = ['MissingWiresDetector', 'Config']

class Config:
    """
    Configuration class for storing color ranges and ROI definitions.
    """
    
    def __init__(self):
        """
        Initialize configuration with predefined color ranges and ROIs.
        """
        self.color_ranges = {
            'black': [
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 75])}
            ],
            'blue': [
                {'lower': np.array([90, 40, 40]), 'upper': np.array([130, 255, 255])}
            ],
            'brown': [
                # {'lower': np.array([0, 30, 20]), 'upper': np.array([20, 200, 150])},
                {'lower': np.array([160, 30, 20]), 'upper': np.array([180, 200, 210])}
            ]
        }

        # ROI definitions in percentages (0.0-1.0) of image dimensions
        # Format: (x_min%, y_min%, x_max%, y_max%)
        # Based on reference image size: 3073x2146
        self.roi_definitions = {
            'blue_brown': (0.0, 0.582, 0.124, 0.699),      # Left side: blue and brown wires
            'black': (0.244, 0.233, 0.309, 0.466),         # Center: black wire
            # 'blue': (0.456, 0.559, 0.586, 0.652),        # Right side: additional blue wire
            # 'black_brown': (0.651, 0.349, 0.732, 0.606)  # Far right: black and brown
        }
        
        # Map ROI names to the colors they should detect
        self.roi_color_mapping = {
            'blue_brown': ['blue', 'brown'],    # This ROI should only detect blue and brown wires
            'black': ['black'],                 # This ROI should only detect black wire
            # 'blue': ['brown'],                  # This ROI should only detect brown wire
            # 'black_brown': ['brown']            # This ROI should only detect brown wire
        }

    def get_color_ranges(self):
        """Return color ranges dictionary."""
        return self.color_ranges

    def get_roi_definitions(self):
        """Return ROI definitions dictionary."""
        return self.roi_definitions
    
    def get_roi_color_mapping(self):
        """Return ROI to color mapping dictionary."""
        return self.roi_color_mapping


class MissingWiresDetector:
    """
    Detector for missing wires using HSV segmentation and contour analysis.
    Detects black, blue, and brown wires in images.
    """
    
    def __init__(self, config=None):
        """
        Initialize detector with configuration.

        Args:
            config: Config instance with color ranges and ROI definitions (optional)
        """
        if config is None:
            config = Config()
        
        self.config = config
        self.color_ranges = self.config.get_color_ranges()
        self.roi_definitions = self.config.get_roi_definitions()
        self.roi_color_mapping = self.config.get_roi_color_mapping()
        self.logger = logging.getLogger(__name__)

    def detect(self, image, roi_name=None):
        """
        Detect wires of predefined colors in image or specific ROI.

        Args:
            image: Input image (BGR format)
            roi_name: ROI name to process (optional, None = whole image)

        Returns:
            dict: Color names mapped to detection status (bool)
        """
        detected_status = {color: False for color in self.color_ranges.keys()}

        if image is None:
            self.logger.error("Input image is None")
            return detected_status

        processed_image = image

        # Crop image to ROI if roi_name is provided and valid
        if roi_name and roi_name in self.roi_definitions:
            roi = self.roi_definitions[roi_name]
            height, width = image.shape[:2]
            
            # Convert percentage coordinates to pixel coordinates
            x_min_pct, y_min_pct, x_max_pct, y_max_pct = roi
            x_min = int(x_min_pct * width)
            y_min = int(y_min_pct * height)
            x_max = int(x_max_pct * width)
            y_max = int(y_max_pct * height)

            # Ensure ROI is within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            if x_min >= x_max or y_min >= y_max:
                self.logger.warning(f"Invalid ROI coordinates {roi} for image size ({width}, {height})")
                return detected_status

            processed_image = image[y_min:y_max, x_min:x_max]
        elif roi_name:
            self.logger.warning(f"ROI '{roi_name}' not found in config")
            return detected_status

        hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        found_colors_set = set()
        
        # Determine which colors to detect in this ROI
        colors_to_detect = self.color_ranges.keys()  # Default: detect all colors
        if roi_name and roi_name in self.roi_color_mapping:
            colors_to_detect = self.roi_color_mapping[roi_name]  # Only detect specific colors for this ROI

        for color, ranges_list in self.color_ranges.items():
            # Skip colors that shouldn't be detected in this ROI
            if color not in colors_to_detect:
                continue
                
            combined_mask = None
            for range_dict in ranges_list:
                lower = range_dict['lower']
                upper = range_dict['upper']
                mask = cv2.inRange(hsv, lower, upper)

                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations to reduce noise
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            # Contour analysis for wire-like shapes
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(w, h) / (min(w, h) + 1e-5)
                area = cv2.contourArea(c)

                # Check for wire-like contours (high aspect ratio, sufficient area)
                # aspect > 2.0: allows slightly less elongated shapes (more tolerant)
                if aspect > 2.0 and area > 150:
                    found_colors_set.add(color)
                    self.logger.debug(f"{color}: Wire detected (aspect={aspect:.2f}, area={area:.0f})")
                    break

        # Update the detected_status dictionary
        for color in found_colors_set:
            detected_status[color] = True

        return detected_status

    def detect_across_rois(self, image):
        """
        Detect colors across all defined ROIs.
        Color is considered present if detected in at least one ROI.

        Args:
            image: Input image (BGR format)

        Returns:
            dict: Color names mapped to detection status across all ROIs
        """
        if image is None:
            self.logger.error("Input image is None")
            return {color: False for color in self.color_ranges.keys()}

        overall_detection_status = {color: False for color in self.color_ranges.keys()}

        for roi_name in self.roi_definitions.keys():
            detected_in_roi = self.detect(image, roi_name=roi_name)

            for color, is_present in detected_in_roi.items():
                if is_present:
                    overall_detection_status[color] = True

        return overall_detection_status

    def get_missing_wires(self, image, use_rois=True):
        """
        Get list of missing wire colors.

        Args:
            image: Input image (BGR format)
            use_rois: If True, detect across all ROIs; if False, detect on whole image

        Returns:
            list: List of missing wire color names
        """
        if use_rois:
            detected = self.detect_across_rois(image)
        else:
            detected = self.detect(image)
        
        missing = [color for color, present in detected.items() if not present]
        return missing


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            print(f"Testing on: {image_path}\n")
            detector = MissingWiresDetector()
            img = cv2.imread(image_path)
            if img is not None:
                detected = detector.detect_across_rois(img)
                missing = detector.get_missing_wires(img)
                
                print("\n" + "="*70)
                print("RESULTS:")
                for color, present in detected.items():
                    status = "✓ PRESENT" if present else "✗ MISSING"
                    print(f"  {color.upper()}: {status}")
                
                if missing:
                    print(f"\n⚠️  Missing wires: {', '.join(missing)}")
                else:
                    print("\n✅ All wires present!")
                print("="*70 + "\n")
        else:
            print(f"Error: Image not found: {image_path}\n")
