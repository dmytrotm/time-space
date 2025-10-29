import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import logging

from utils.roi_cropper import ROICropper

RELATIVE_HALF_SIZE = 0.052

class YOLOROIMapper:

    def __init__(self, class_names=["label", "tape"]):
        self.class_names = class_names or []
        self.colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
        ]
        self.logger = logging.getLogger(__name__)

    def load_positions_json(self, json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"JSON file {json_path} not found")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON format in {json_path}")
            return None

    def get_tape_roi_coordinates(self, positions_data, tape_id, original_size):
        """Get tape ROI coordinates - supports both rectangle and center-based formats"""
        if not positions_data or "tapes" not in positions_data:
            self.logger.error("No tapes data found in JSON")
            return None, None

        target_tape = None
        for tape in positions_data["tapes"]:
            if tape["id"] == tape_id:
                target_tape = tape
                break

        if target_tape is None:
            self.logger.error(f"Tape with ID {tape_id} not found")
            available_ids = [tape["id"] for tape in positions_data["tapes"]]
            self.logger.info(f"Available tape IDs: {available_ids}")
            return None, None

        orig_w, orig_h = original_size

        # Check if it's center-based (square) or rectangle format
        if "center" in target_tape and "relative_half_size" in target_tape:
            # Center-based square format
            center_x_px = int(target_tape["center"]["x"] * orig_w)
            center_y_px = int(target_tape["center"]["y"] * orig_h)
            
            reference_dim = min(orig_w, orig_h)
            half_size_px = int(target_tape["relative_half_size"] * reference_dim)
            
            roi_x = center_x_px - half_size_px
            roi_y = center_y_px - half_size_px
            roi_w = 2 * half_size_px
            roi_h = 2 * half_size_px
            
        elif "start" in target_tape and "end" in target_tape:
            # Rectangle format
            start_x_px = int(target_tape["start"]["x"] * orig_w)
            start_y_px = int(target_tape["start"]["y"] * orig_h)
            end_x_px = int(target_tape["end"]["x"] * orig_w)
            end_y_px = int(target_tape["end"]["y"] * orig_h)

            roi_x = min(start_x_px, end_x_px)
            roi_y = min(start_y_px, end_y_px)
            roi_w = abs(end_x_px - start_x_px)
            roi_h = abs(end_y_px - start_y_px)
        else:
            self.logger.error(f"Tape {tape_id} has invalid format")
            return None, None

        # Add padding
        padding = 10
        roi_x = max(0, roi_x - padding)
        roi_y = max(0, roi_y - padding)
        roi_w = min(orig_w - roi_x, roi_w + 2 * padding)
        roi_h = min(orig_h - roi_y, roi_h + 2 * padding)
       

        self.logger.info(
            f"Tape {tape_id} ROI: position=({roi_x}, {roi_y}), size=({roi_w}x{roi_h})"
        )

        return (roi_x, roi_y), (roi_w, roi_h)

    def load_yolo_annotations_from_file(self, annotation_path):
        annotations = []
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append(
                            [class_id, x_center, y_center, width, height]
                        )
        else:
            self.logger.warning(f"Annotation file {annotation_path} not found")

        return annotations

    def map_roi_annotations_to_original(
        self,
        roi_annotations,
        roi_position,
        roi_size,
        original_size,
        tape_id,
    ):
        """Map ROI annotations to original image coordinates in square (center-based) format
        The right border of the square will be at the center of the detected object"""
        if roi_position is None or roi_position[0] is None:
            self.logger.warning("Invalid ROI position")
            return []

        mapped_rois = []
        roi_x, roi_y = roi_position
        roi_w, roi_h = roi_size
        orig_w, orig_h = original_size

        for ann in roi_annotations:
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = ann

            # Convert normalized ROI coordinates to pixels within ROI
            x_center_roi_px = x_center_norm * roi_w
            y_center_roi_px = y_center_norm * roi_h

            # Convert to original image pixel coordinates
            x_center_orig_px = roi_x + x_center_roi_px
            y_center_orig_px = roi_y + y_center_roi_px

            # Normalize to original image dimensions
            x_center_orig_norm = x_center_orig_px / orig_w
            y_center_orig_norm = y_center_orig_px / orig_h

            square_center_x = x_center_orig_norm - 0.016
            
            square_center_y = y_center_orig_norm

            roi = {
                "id": tape_id,
                "center": {
                    "x": square_center_x,
                    "y": square_center_y,
                },
                "relative_half_size": RELATIVE_HALF_SIZE
            }

            mapped_rois.append(roi)

        return mapped_rois

    def process_roi_mapping(
        self,
        original_img,
        roi_annotations,
        positions_data,
    ):
        if original_img is None:
            self.logger.error(f"Could not load image")
            return {"orientation": []}

        original_size = (
            original_img.shape[1],
            original_img.shape[0],
        )
        self.logger.info(f"Original image size: {original_size}")

        if positions_data is None:
            return {"orientation": []}

        all_rois = []

        for tape_id, annotations in roi_annotations.items():
            self.logger.info(f"\n--- Processing tape {tape_id} ---")

            roi_position, roi_size = self.get_tape_roi_coordinates(
                positions_data, tape_id, original_size
            )

            if roi_position is None:
                self.logger.warning(f"Skipping tape {tape_id} due to invalid ROI")
                continue

            if not annotations:
                self.logger.info(f"No annotations provided for tape {tape_id}")
                continue

            self.logger.info(f"Processing {len(annotations)} annotations for tape {tape_id}")

            mapped_rois = self.map_roi_annotations_to_original(
                annotations, roi_position, roi_size, original_size, tape_id
            )

            if mapped_rois:
                all_rois.extend(mapped_rois)

        self.logger.info(f"\n=== PROCESSING COMPLETE ===")
        self.logger.info(f"Total ROI structures created: {len(all_rois)}")

        return {"orientation": all_rois}

    def visualize_rois_on_image(self, original_img, result_json, positions_json_path, roi_annotations=None):
        """Visualize ROIs - supports both square and rectangle formats"""
        original_size = (original_img.shape[1], original_img.shape[0])
        orig_w, orig_h = original_size

        # Draw tape ROIs from positions data
        positions_data = self.load_positions_json(positions_json_path)
        if positions_data and 'tapes' in positions_data:
            for tape in positions_data['tapes']:
                # Check format and draw accordingly
                if 'center' in tape and 'relative_half_size' in tape:
                    # Square format
                    center_x_px = int(tape['center']['x'] * orig_w)
                    center_y_px = int(tape['center']['y'] * orig_h)
                    
                    reference_dim = min(orig_w, orig_h)
                    half_size_px = int(tape['relative_half_size'] * reference_dim)
                    
                    start_x_px = center_x_px - half_size_px
                    start_y_px = center_y_px - half_size_px
                    end_x_px = center_x_px + half_size_px
                    end_y_px = center_y_px + half_size_px
                    
                    # Draw center point
                    cv2.circle(original_img, (center_x_px, center_y_px), 5, (255, 0, 0), -1)
                    
                elif 'start' in tape and 'end' in tape:
                    # Rectangle format
                    start_x_px = int(tape['start']['x'] * orig_w)
                    start_y_px = int(tape['start']['y'] * orig_h)
                    end_x_px = int(tape['end']['x'] * orig_w)
                    end_y_px = int(tape['end']['y'] * orig_h)
                else:
                    continue
                
                cv2.rectangle(original_img, (start_x_px, start_y_px), (end_x_px, end_y_px), (255, 0, 0), 2)
                label = f"Tape ID: {tape.get('id', 'N/A')}"
                cv2.putText(original_img, label, (start_x_px, start_y_px - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Draw YOLO detections on ROIs
        if roi_annotations:
            for tape_id, annotations in roi_annotations.items():
                if not annotations:
                    continue

                roi_position, roi_size = self.get_tape_roi_coordinates(positions_data, tape_id, original_size)
                if roi_position is None:
                    continue

                roi_x, roi_y = roi_position
                roi_w, roi_h = roi_size

                for ann in annotations:
                    class_id_float, x_center_norm, y_center_norm, width_norm, height_norm = ann
                    class_id = int(class_id_float)

                    x_center_roi_px = int(x_center_norm * roi_w)
                    y_center_roi_px = int(y_center_norm * roi_h)
                    width_roi_px = int(width_norm * roi_w)
                    height_roi_px = int(height_norm * roi_h)

                    x_center_orig_px = roi_x + x_center_roi_px
                    y_center_orig_px = roi_y + y_center_roi_px

                    start_x = x_center_orig_px - width_roi_px // 2
                    start_y = y_center_orig_px - height_roi_px // 2
                    end_x = x_center_orig_px + width_roi_px // 2
                    end_y = y_center_orig_px + height_roi_px // 2

                    cv2.rectangle(original_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    
                    # Draw vertical line at detection center
                    cv2.line(original_img, (x_center_orig_px, start_y), 
                            (x_center_orig_px, end_y), (0, 255, 255), 2)

        # Draw final orientation ROIs (square format)
        if "orientation" not in result_json or not result_json["orientation"]:
            self.logger.info("No orientation ROIs to visualize.")
            plt.figure(figsize=(15, 15))
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title("ROIs on Original Image")
            plt.axis("off")
            plt.show()
            return

        reference_dim = min(orig_w, orig_h)
        
        for roi in result_json["orientation"]:
            # Square format with center and relative_half_size
            center_x_px = int(roi["center"]["x"] * orig_w)
            center_y_px = int(roi["center"]["y"] * orig_h)
            half_size_px = int(roi["relative_half_size"] * reference_dim)
            
            start_x = center_x_px - half_size_px
            start_y = center_y_px - half_size_px
            end_x = center_x_px + half_size_px
            end_y = center_y_px + half_size_px

            class_id = roi.get("id", 0) % len(self.colors)
            color = self.colors[class_id]

            # Draw square ROI
            cv2.rectangle(original_img, (start_x, start_y), (end_x, end_y), color, 2)
            
            # Draw center point
            cv2.circle(original_img, (center_x_px, center_y_px), 4, color, -1)
            
            # Draw vertical line at right edge (should align with detection center)
            cv2.line(original_img, (end_x, start_y), (end_x, end_y), color, 3)

            label = f"Orient ID: {roi.get('id', 'N/A')}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                original_img, (start_x, start_y - h - 10), (start_x + w, start_y), color, -1
            )
            cv2.putText(
                original_img,
                label,
                (start_x, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title("ROIs on Original Image")
        plt.axis("off")
        plt.show()

    def get_images(self, workspace, roi_annotations, positions_data):
        new_rois_json = self.process_roi_mapping(workspace, roi_annotations, positions_data)
        
        # Check if there are any ROIs to process
        if not new_rois_json.get("orientation"):
            self.logger.info("No orientation ROIs found, skipping cropping")
            return None
        
        cropper = ROICropper(new_rois_json)
        new_rois_images = cropper.crop(workspace)
        return new_rois_images, new_rois_json