import numpy as np
import logging


class ROICropper:
    def __init__(self, roi_data):
        self.roi_objects = {}
        self.total_rois = 0
        self.logger = logging.getLogger(__name__)
        self.process_rois(roi_data)

    def _to_float(self, value):
        """Convert tensor or any numeric type to Python float"""
        if hasattr(value, "item"):  # PyTorch tensor or numpy scalar
            return float(value.item())
        return float(value)

    def process_rois(self, roi_data):
        """Process ROI data and categorize by type"""
        if not isinstance(roi_data, dict):
            raise TypeError("ROI data must be a dictionary.")

        for key, value in roi_data.items():
            if isinstance(value, list):
                valid_rois = []
                for obj in value:
                    roi_type = self.detect_roi_type(obj)
                    if roi_type:
                        valid_rois.append({**obj, "_type": roi_type})

                if valid_rois:
                    self.roi_objects[key] = valid_rois
                    self.total_rois += len(valid_rois)
                    self.logger.info(f"Loaded {len(valid_rois)} {key}")
                else:
                    self.logger.warning(f"Skipping '{key}': no valid ROI objects found")

            elif isinstance(value, dict):
                roi_type = self.detect_roi_type(value)
                if roi_type:
                    self.roi_objects[key] = [{**value, "_type": roi_type}]
                    self.total_rois += 1
                    self.logger.info(f"Loaded 1 {key}")
                else:
                    self.logger.warning(f"Skipping '{key}': not a valid ROI object")

        if self.total_rois == 0:
            raise ValueError("No valid ROI objects found in the data")

        self.logger.info(f"Total ROIs loaded: {self.total_rois}")

    def detect_roi_type(self, obj):
        """Detect ROI type: 'center' (square) or 'rectangle'"""
        if not isinstance(obj, dict):
            return None

        # Check for center-based ROI (square)
        if "center" in obj and "relative_half_size" in obj:
            center = obj["center"]
            if isinstance(center, dict) and "x" in center and "y" in center:
                try:
                    self._to_float(center["x"])
                    self._to_float(center["y"])
                    self._to_float(obj["relative_half_size"])
                    return "center"
                except (ValueError, TypeError, AttributeError):
                    return None

        # Check for rectangle ROI (start/end points)
        if "start" in obj and "end" in obj:
            start = obj["start"]
            end = obj["end"]
            if isinstance(start, dict) and isinstance(end, dict):
                if "x" in start and "y" in start and "x" in end and "y" in end:
                    try:
                        self._to_float(start["x"])
                        self._to_float(start["y"])
                        self._to_float(end["x"])
                        self._to_float(end["y"])
                        return "rectangle"
                    except (ValueError, TypeError, AttributeError):
                        return None

        return None

    def calculate_roi_bounds_center(self, image_width, image_height, roi):
        """Calculate bounds for center-based (square) ROI"""
        # Convert to Python float first
        center_x = self._to_float(roi["center"]["x"]) * image_width
        center_y = self._to_float(roi["center"]["y"]) * image_height

        # Calculate half size in pixels based on minimum dimension
        reference_dim = min(image_width, image_height)
        half_size_pixels = self._to_float(roi["relative_half_size"]) * reference_dim
        half_size_pixels = max(1, int(round(half_size_pixels)))

        # Calculate bounds (now round() will work)
        center_x = int(round(center_x))
        center_y = int(round(center_y))

        x1 = center_x - half_size_pixels
        y1 = center_y - half_size_pixels
        x2 = center_x + half_size_pixels
        y2 = center_y + half_size_pixels

        return x1, y1, x2, y2

    def calculate_roi_bounds_rectangle(self, image_width, image_height, roi):
        """Calculate bounds for rectangle ROI (start/end points)"""
        # Convert to Python float first
        start_x = self._to_float(roi["start"]["x"]) * image_width
        start_y = self._to_float(roi["start"]["y"]) * image_height
        end_x = self._to_float(roi["end"]["x"]) * image_width
        end_y = self._to_float(roi["end"]["y"]) * image_height

        # Now round() will work
        start_x = int(round(start_x))
        start_y = int(round(start_y))
        end_x = int(round(end_x))
        end_y = int(round(end_y))

        x1 = min(start_x, end_x)
        x2 = max(start_x, end_x)
        y1 = min(start_y, end_y)
        y2 = max(start_y, end_y)

        return x1, y1, x2, y2

    def calculate_roi_bounds(self, image_width, image_height, roi):
        """Universal function to calculate ROI bounds based on type"""
        roi_type = roi.get("_type")

        if roi_type == "center":
            return self.calculate_roi_bounds_center(image_width, image_height, roi)
        elif roi_type == "rectangle":
            return self.calculate_roi_bounds_rectangle(image_width, image_height, roi)
        else:
            raise ValueError(f"Unknown ROI type: {roi_type}")

    def crop(self, image):
        """Crop all ROIs from the image"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")

        height, width = image.shape[:2]
        cropped_images = {}

        for category_name, roi_list in self.roi_objects.items():
            for roi in roi_list:
                roi_id = roi.get("id", 0)
                roi_type = roi.get("_type", "unknown")

                try:
                    x1, y1, x2, y2 = self.calculate_roi_bounds(width, height, roi)

                    # Clamp to image boundaries
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))

                    # Validate bounds
                    if x2 <= x1 or y2 <= y1:
                        self.logger.warning(
                            f"Invalid bounds for {category_name} {roi_id} ({roi_type}): "
                            f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
                        )
                        continue

                    # Crop the ROI
                    cropped_roi = image[y1:y2, x1:x2]

                    if cropped_roi.size == 0:
                        self.logger.warning(f"Empty crop for {category_name} {roi_id}")
                        continue

                    # Create unique key
                    key = f"{category_name.upper()}_{roi_id:03d}"
                    cropped_images[key] = cropped_roi

                    self.logger.debug(
                        f"Cropped {key} ({roi_type}): {cropped_roi.shape[1]}x{cropped_roi.shape[0]}"
                    )

                except Exception as e:
                    self.logger.error(f"Error cropping {category_name} {roi_id}: {e}")
                    continue

        return cropped_images
