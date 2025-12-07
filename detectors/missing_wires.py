"""
Missing Wires Detector

Детектор відсутніх дротів на основі HSV кольорової сегментації.
Виявляє чорні, сині та коричневі дроти.
"""

import cv2
import numpy as np
import logging
import json
import os

__all__ = ['MissingWiresDetector']

# Default path to color ranges config
DEFAULT_COLOR_RANGES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'configs', 
    'wire_color_ranges.json'
)


class MissingWiresDetector:
    """
    Detector for missing wires using HSV segmentation and contour analysis.
    Detects black, blue, and brown wires in images.
    """
    
    def __init__(self, color_ranges_path=None):
        """
        Initialize detector with color ranges from JSON config.

        Args:
            color_ranges_path: Path to JSON file with color ranges (optional)
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Then load color ranges (which may use logger)
        if color_ranges_path is None:
            color_ranges_path = DEFAULT_COLOR_RANGES_PATH
        
        self.color_ranges = self._load_color_ranges(color_ranges_path)


    def _load_color_ranges(self, path):
        """
        Load color ranges from JSON configuration file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            dict: Color ranges with numpy arrays
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
            # Return empty dict as fallback
            return {}

    def set_color_ranges_from_dict(self, ranges_dict):
        """
        Set color ranges from a dictionary (loaded from config).
        
        Args:
            ranges_dict: Dictionary with color ranges
        """
        color_ranges = {}
        for color, ranges_list in ranges_dict.items():
            color_ranges[color] = []
            for range_dict in ranges_list:
                color_ranges[color].append({
                    'lower': np.array(range_dict['lower']),
                    'upper': np.array(range_dict['upper'])
                })
        
        self.color_ranges = color_ranges


    def detect(self, image, colors_to_detect=None):
        """
        Detect wires of specified colors in image.

        Args:
            image: Input image (BGR format)
            colors_to_detect: List of color names to detect (optional, None = all colors)

        Returns:
            dict: Color names mapped to detection status (bool)
        """
        detected_status = {color: False for color in self.color_ranges.keys()}

        if image is None or image.size == 0:
            self.logger.error("Input image is None or empty")
            return detected_status

        # Determine which colors to detect
        if colors_to_detect is None:
            colors_to_detect = self.color_ranges.keys()

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        found_colors_set = set()

        for color in colors_to_detect:
            if color not in self.color_ranges:
                self.logger.warning(f"Color '{color}' not in configured ranges")
                continue
                
            combined_mask = None
            for range_dict in self.color_ranges[color]:
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
                if aspect > 2.0 and area > 150:
                    found_colors_set.add(color)
                    self.logger.debug(f"{color}: Wire detected (aspect={aspect:.2f}, area={area:.0f})")
                    break

        # Update the detected_status dictionary
        for color in found_colors_set:
            detected_status[color] = True

        return detected_status

    def is_present(self, image, expected_colors=None):
        """
        Check if all expected wire colors are present in the image.
        Similar to GroundingWireDetector.is_present() for unified interface.

        Args:
            image: Input image (BGR format), pre-cropped to ROI
            expected_colors: List of color names expected in this ROI (e.g., ['blue', 'brown'])
                           If None, checks for any wire color

        Returns:
            bool: True if all expected colors are present, False otherwise
        """
        if expected_colors is None:
            # If no specific colors expected, check if any wire is present
            expected_colors = list(self.color_ranges.keys())
            # Return True if at least one color is detected
            detected = self.detect(image, expected_colors)
            return any(detected.values())
        
        # Check if all expected colors are present
        detected = self.detect(image, expected_colors)
        
        for color in expected_colors:
            if color not in detected or not detected[color]:
                self.logger.info(f"Missing expected wire color: {color}")
                return False
        
        return True

    def get_missing_wires(self, image, expected_colors=None):
        """
        Get list of missing wire colors from expected set.

        Args:
            image: Input image (BGR format)
            expected_colors: List of expected color names (optional, None = all colors)

        Returns:
            list: List of missing wire color names
        """
        if expected_colors is None:
            expected_colors = list(self.color_ranges.keys())
        
        detected = self.detect(image, expected_colors)
        missing = [color for color in expected_colors if not detected.get(color, False)]
        return missing
