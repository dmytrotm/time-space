import cv2
import numpy as np
from utils.constants import GREEN, RED, ORANGE

class Visualizer:
    """
    A class to handle visualization of ROIs and detections on an image.
    """
    
    def __init__(self, image):
        self.image = image.copy()
    
    def draw_bounding_box(self, box, color, label=""):
        """
        Draws a bounding box on the image.
        :param box: A tuple of (x, y, w, h) for the bounding box.
        :param color: The color of the bounding box.
        :param label: The label for the bounding box.
        """
        x, y, w, h = box
        cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 3)
        if label:
            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            text_x = x + (w - text_width) // 2
            text_y = y + h + 50
            cv2.putText(self.image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    
    def draw_roi_rectangle(self, roi, color, label=""):
        """
        Draws a rectangle ROI (start/end points) on the image.
        :param roi: A dictionary with 'start' and 'end' points.
        :param color: The color of the bounding box.
        :param label: The label for the bounding box.
        """
        height, width = self.image.shape[:2]
        
        start_x = round(roi["start"]["x"] * width)
        start_y = round(roi["start"]["y"] * height)
        end_x = round(roi["end"]["x"] * width)
        end_y = round(roi["end"]["y"] * height)
        
        x1 = min(start_x, end_x)
        x2 = max(start_x, end_x)
        y1 = min(start_y, end_y)
        y2 = max(start_y, end_y)
        
        cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 3)
        
        if label:
            # Get text size to center it
            box_width = x2 - x1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            text_x = x1 + (box_width - text_width) // 2
            text_y = y2 + 50
            cv2.putText(self.image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    
    def draw_roi_center(self, roi, color, label=""):
        """
        Draws a center-based square ROI on the image.
        :param roi: A dictionary with 'center' and 'relative_half_size'.
        :param color: The color of the bounding box.
        :param label: The label for the bounding box.
        """
        height, width = self.image.shape[:2]
        
        # Convert relative center to pixel coordinates
        center_x = round(roi["center"]["x"] * width)
        center_y = round(roi["center"]["y"] * height)
        
        # Calculate half size in pixels
        reference_dim = min(width, height)
        half_size_pixels = round(roi["relative_half_size"] * reference_dim)
        half_size_pixels = max(1, half_size_pixels)
        
        # Calculate bounds
        x1 = center_x - half_size_pixels
        y1 = center_y - half_size_pixels
        x2 = center_x + half_size_pixels
        y2 = center_y + half_size_pixels
        
        # Draw rectangle
        cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 3)
        
        if label:
            # Get text size to center it
            box_width = x2 - x1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
            text_x = x1 + (box_width - text_width) // 2
            text_y = y2 + 50
            cv2.putText(self.image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    
    def draw_roi(self, roi, color, label=""):
        """
        Draws a single ROI on the image (universal function).
        Automatically detects ROI type and uses appropriate drawing method.
        :param roi: A dictionary representing the ROI.
        :param color: The color of the bounding box.
        :param label: The label for the bounding box.
        """
        roi_type = roi.get('_type')
        
        if roi_type == 'center':
            self.draw_roi_center(roi, color, label)
        elif roi_type == 'rectangle':
            self.draw_roi_rectangle(roi, color, label)
        else:
            # Fallback: try to detect type on the fly
            if 'center' in roi and 'relative_half_size' in roi:
                self.draw_roi_center(roi, color, label)
            elif 'start' in roi and 'end' in roi:
                self.draw_roi_rectangle(roi, color, label)
            else:
                print(f"Warning: Unknown ROI format for {label}")
    
    def draw_rois(self, roi_cropper, color=None, label_prefix="", 
                  color_by_category=False):
        """
        Draws all ROIs from an ROICropper instance on the image.
        :param roi_cropper: An instance of the ROICropper class.
        :param color: The color of the bounding boxes (if None and color_by_category=False, uses GREEN).
        :param label_prefix: A prefix for the ROI labels.
        :param color_by_category: If True, uses different colors for different categories.
        """
        # Default colors for different categories
        category_colors = {
            0: GREEN,
            1: RED,
            2: ORANGE,
        }
        
        if color is None and not color_by_category:
            color = GREEN
        
        for idx, (category_name, roi_list) in enumerate(roi_cropper.roi_objects.items()):
            # Choose color
            if color_by_category:
                current_color = category_colors.get(idx % len(category_colors), GREEN)
            else:
                current_color = color
            
            for roi in roi_list:
                roi_id = roi.get('id', 0)
                roi_type = roi.get('_type', 'unknown')
                
                # Create label with type indicator
                type_indicator = "■" if roi_type == 'center' else "▭"
                label = f"{label_prefix}{category_name.upper()}_{roi_id:03d} {type_indicator}"
                
                self.draw_roi(roi, current_color, label)
    
    def draw_rois_by_category(self, roi_cropper, category_name, color, label_prefix=""):
        """
        Draws ROIs from a specific category only.
        :param roi_cropper: An instance of the ROICropper class.
        :param category_name: The name of the category to draw.
        :param color: The color of the bounding boxes.
        :param label_prefix: A prefix for the ROI labels.
        """
        if category_name not in roi_cropper.roi_objects:
            print(f"Warning: Category '{category_name}' not found in ROICropper")
            return
        
        roi_list = roi_cropper.roi_objects[category_name]
        for roi in roi_list:
            roi_id = roi.get('id', 0)
            roi_type = roi.get('_type', 'unknown')
            
            type_indicator = "■" if roi_type == 'center' else "▭"
            label = f"{label_prefix}{category_name.upper()}_{roi_id:03d} {type_indicator}"
            
            self.draw_roi(roi, color, label)
    
    def add_legend(self, position=(10, 30), font_scale=1.0, thickness=5):
        """
        Adds a legend explaining the ROI type indicators.
        :param position: Starting position (x, y) for the legend.
        :param font_scale: Font scale for the text.
        :param thickness: Thickness of the text.
        """
        x, y = position
        line_height = 40
        
        cv2.putText(self.image, "ROI Types:", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, GREEN, thickness)
        
        cv2.putText(self.image, "  Square (center-based)", (x, y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, GREEN, thickness - 1)
        
        cv2.putText(self.image, "  Rectangle (start-end)", (x, y + line_height * 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, GREEN, thickness - 1)
    
    def get_image(self):
        """
        Returns the image with visualizations.
        """
        return self.image