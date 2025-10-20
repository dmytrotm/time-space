import numpy as np
import logging

class ROICropper:
    def __init__(self, roi_data):
        self.roi_objects = {}
        self.total_rois = 0
        self.logger = logging.getLogger(__name__)
        self.process_rois(roi_data)

    def process_rois(self, roi_data):
        if not isinstance(roi_data, dict):
            raise TypeError("ROI data must be a dictionary.")

        for key, value in roi_data.items():
            if isinstance(value, list):
                if value and all(self.is_valid_roi_object(obj) for obj in value):
                    self.roi_objects[key] = value
                    self.total_rois += len(value)
                    self.logger.info(f"Loaded {len(value)} {key}")
            elif isinstance(value, dict) and self.is_valid_roi_object(value):
                self.roi_objects[key] = [value]
                self.total_rois += 1
                self.logger.info(f"Loaded 1 {key}")
            else:
                self.logger.warning(f"Skipping '{key}': not a valid ROI object or list of ROI objects")

        if self.total_rois == 0:
            raise ValueError("No valid ROI objects found in the data")

        self.logger.info(f"Total ROIs loaded: {self.total_rois}")

    def is_valid_roi_object(self, obj):
        if not isinstance(obj, dict):
            return False
        
        if 'start' not in obj or 'end' not in obj:
            return False
        
        start = obj['start']
        end = obj['end']
        
        if not (isinstance(start, dict) and isinstance(end, dict)):
            return False
        
        required_coords = ['x', 'y']
        if not all(coord in start and coord in end for coord in required_coords):
            return False
        
        try:
            float(start['x'])
            float(start['y'])
            float(end['x'])
            float(end['y'])
            return True
        except (ValueError, TypeError):
            return False

    def calculate_roi_bounds(self, image_width, image_height, roi):
        start_x = int(roi["start"]["x"] * image_width)
        start_y = int(roi["start"]["y"] * image_height)
        end_x = int(roi["end"]["x"] * image_width)
        end_y = int(roi["end"]["y"] * image_height)
        
        x1, x2 = min(start_x, end_x), max(start_x, end_x)
        y1, y2 = min(start_y, end_y), max(start_y, end_y)
        
        return x1, y1, x2, y2

    def crop(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        height, width = image.shape[:2]
        cropped_images = {}

        for category_name, roi_list in self.roi_objects.items():
            for roi in roi_list:
                roi_id = roi.get("id", 0)
                x1, y1, x2, y2 = self.calculate_roi_bounds(width, height, roi)

                x1 = max(0, min(x1, width))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height))
                y2 = max(0, min(y2, height))

                if x2 <= x1 or y2 <= y1:
                    self.logger.warning(f"Invalid bounds for {category_name} {roi_id}")
                    continue

                cropped_roi = image[y1:y2, x1:x2]

                if cropped_roi.size == 0:
                    self.logger.warning(f"Empty crop for {category_name} {roi_id}")
                    continue
                
                key = f"{category_name.upper()}_{roi_id:03d}"
                cropped_images[key] = cropped_roi

        return cropped_images