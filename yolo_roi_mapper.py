import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class YOLOROIMapper:

    def __init__(self, class_names=["label", "tape"]):
        """
        Initialize the ROI mapper

        Args:
            class_names (list): List of class names corresponding to class IDs
        """
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

    def load_positions_json(self, json_path):
        """
        Load positions from JSON file

        Args:
            json_path (str): Path to JSON file with tape/wire positions

        Returns:
            dict: Parsed JSON data
        """
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file {json_path} not found")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_path}")
            return None

    def get_tape_roi_coordinates(self, positions_data, tape_id, original_size):
        """
        Get ROI coordinates for a specific tape from JSON data

        Args:
            positions_data (dict): Parsed JSON data
            tape_id (int): ID of the tape to use as ROI
            original_size (tuple): (width, height) of original image

        Returns:
            tuple: (roi_position, roi_size) or (None, None) if tape not found
        """
        if not positions_data or "tapes" not in positions_data:
            print("Error: No tapes data found in JSON")
            return None, None

        target_tape = None
        for tape in positions_data["tapes"]:
            if tape["id"] == tape_id:
                target_tape = tape
                break

        if target_tape is None:
            print(f"Error: Tape with ID {tape_id} not found")
            available_ids = [tape["id"] for tape in positions_data["tapes"]]
            print(f"Available tape IDs: {available_ids}")
            return None, None

        orig_w, orig_h = original_size

        start_x_px = int(target_tape["start"]["x"] * orig_w)
        start_y_px = int(target_tape["start"]["y"] * orig_h)
        end_x_px = int(target_tape["end"]["x"] * orig_w)
        end_y_px = int(target_tape["end"]["y"] * orig_h)

        roi_x = min(start_x_px, end_x_px)
        roi_y = min(start_y_px, end_y_px)
        roi_w = abs(end_x_px - start_x_px)
        roi_h = abs(end_y_px - start_y_px)

        padding = 10
        roi_x = max(0, roi_x - padding)
        roi_y = max(0, roi_y - padding)
        roi_w = min(orig_w - roi_x, roi_w + 2 * padding)
        roi_h = min(orig_h - roi_y, roi_h + 2 * padding)

        print(
            f"Tape {tape_id} ROI: position=({roi_x}, {roi_y}), size=({roi_w}x{roi_h})"
        )

        return (roi_x, roi_y), (roi_w, roi_h)

    def load_yolo_annotations_from_file(self, annotation_path):
        """
        Load YOLO format annotations from file (kept as separate method for compatibility)

        Args:
            annotation_path (str): Path to YOLO annotation file

        Returns:
            list: List of [class_id, x_center, y_center, width, height] (normalized)
        """
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
            print(f"Warning: Annotation file {annotation_path} not found")

        return annotations

    def map_roi_annotations_to_original(
        self,
        roi_annotations,
        roi_position,
        roi_size,
        original_size,
        tape_id,
    ):
        """
        Map YOLO annotations from ROI coordinates to original image coordinates and create JSON structure

        Args:
            roi_annotations (list): List of YOLO annotations in ROI coordinates
            roi_position (tuple): (x, y) position of ROI in original image
            roi_size (tuple): (width, height) of ROI
            original_size (tuple): (width, height) of original image
            tape_id (int): ID of the tape (used as ROI ID)

        Returns:
            list: List of ROI structures for this specific tape
        """
        if roi_position is None or roi_position[0] is None:
            print("Warning: Invalid ROI position")
            return []

        mapped_rois = []
        roi_x, roi_y = roi_position
        roi_w, roi_h = roi_size
        orig_w, orig_h = original_size

        for ann in roi_annotations:
            class_id, x_center_norm, y_center_norm, width_norm, height_norm = ann

            x_center_roi_px = x_center_norm * roi_w
            y_center_roi_px = y_center_norm * roi_h

            x_center_orig_px = roi_x + x_center_roi_px
            y_center_orig_px = roi_y + y_center_roi_px

            x_center_orig_norm = x_center_orig_px / orig_w
            y_center_orig_norm = y_center_orig_px / orig_h

            new_roi_width = 0.032
            new_roi_height = 0.04266

            # The right border of the new ROI is at the center of the detected object
            new_roi_end_x = x_center_orig_norm
            new_roi_start_x = new_roi_end_x - new_roi_width

            # The y-center of the new ROI is at the y-center of the detected object
            new_roi_start_y = y_center_orig_norm - new_roi_height / 2
            new_roi_end_y = y_center_orig_norm + new_roi_height / 2

            roi = {
                "id": tape_id,
                "start": {
                    "x": new_roi_start_x,
                    "y": new_roi_start_y,
                },
                "end": {
                    "x": new_roi_end_x,
                    "y": new_roi_end_y,
                },
            }

            mapped_rois.append(roi)

        return mapped_rois

    def process_roi_mapping(
        self,
        original_img_path,
        roi_annotations,
        positions_json_path,
    ):
        """
        Complete process to map ROI annotations to original image and return JSON
        Now only accepts dictionary format for consistency

        Args:
            original_img_path (str): Path to original image
            roi_annotations (dict): Dictionary with tape_id as key and list of annotations as value
                                  e.g., {1: [[1, 0.5, 0.5, 0.2, 0.2]], 2: [[1, 0.3, 0.3, 0.1, 0.1]]}
            positions_json_path (str): Path to JSON file with tape positions

        Returns:
            dict: JSON object with all mapped ROI structures from all tapes
        """
        if not isinstance(roi_annotations, dict):
            print(
                "Error: roi_annotations must be a dictionary with tape_id as key and annotations list as value"
            )
            print(
                "Example: {1: [[1, 0.5, 0.5, 0.2, 0.2]], 2: [[1, 0.3, 0.3, 0.1, 0.1]]}"
            )
            return {"orientation": []}

        if not os.path.exists(original_img_path):
            print(f"Error: Original image {original_img_path} not found")
            return {"orientation": []}

        original_img = cv2.imread(original_img_path)
        if original_img is None:
            print(f"Error: Could not load image {original_img_path}")
            return {"orientation": []}

        original_size = (
            original_img.shape[1],
            original_img.shape[0],
        )
        print(f"Original image size: {original_size}")

        positions_data = self.load_positions_json(positions_json_path)
        if positions_data is None:
            return {"orientation": []}

        all_rois = []

        for tape_id, annotations in roi_annotations.items():
            print(f"\n--- Processing tape {tape_id} ---")

            roi_position, roi_size = self.get_tape_roi_coordinates(
                positions_data, tape_id, original_size
            )

            if roi_position is None:
                print(f"Skipping tape {tape_id} due to invalid ROI")
                continue

            if not annotations:
                print(f"No annotations provided for tape {tape_id}")
                continue

            print(f"Processing {len(annotations)} annotations for tape {tape_id}")

            mapped_rois = self.map_roi_annotations_to_original(
                annotations, roi_position, roi_size, original_size, tape_id
            )

            if mapped_rois:
                all_rois.extend(mapped_rois)

        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total ROI structures created: {len(all_rois)}")

        return {"orientation": all_rois}

    def visualize_rois_on_image(self, original_img_path, result_json, positions_json_path, roi_annotations=None):
        """
        Visualize the ROIs on the original image.

        Args:
            original_img_path (str): Path to the original image.
            result_json (dict): The JSON object with ROI structures.
            positions_json_path (str): Path to the JSON file with old tape positions.
            roi_annotations (dict): Dictionary with tape_id as key and list of annotations as value.
        """
        if not os.path.exists(original_img_path):
            print(f"Error: Original image {original_img_path} not found")
            return

        img = cv2.imread(original_img_path)
        if img is None:
            print(f"Error: Could not load image {original_img_path}")
            return

        original_size = (img.shape[1], img.shape[0])
        orig_w, orig_h = original_size

        # Draw old ROIs
        positions_data = self.load_positions_json(positions_json_path)
        if positions_data and 'tapes' in positions_data:
            for tape in positions_data['tapes']:
                start_x_px = int(tape['start']['x'] * orig_w)
                start_y_px = int(tape['start']['y'] * orig_h)
                end_x_px = int(tape['end']['x'] * orig_w)
                end_y_px = int(tape['end']['y'] * orig_h)
                cv2.rectangle(img, (start_x_px, start_y_px), (end_x_px, end_y_px), (255, 0, 0), 2) # Blue for old ROIs
                label = f"ID: {tape.get('id', 'N/A')}"
                cv2.putText(img, label, (start_x_px, start_y_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        # Draw detected objects
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

                    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2) # Green for detected objects


        if "orientation" not in result_json or not result_json["orientation"]:
            print("No ROIs to visualize.")
            # Still show the image with old ROIs if any
            plt.figure(figsize=(15, 15))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("ROIs on Original Image")
            plt.axis("off")
            plt.show()
            return

        for roi in result_json["orientation"]:
            start_x = int(roi["start"]["x"] * orig_w)
            start_y = int(roi["start"]["y"] * orig_h)
            end_x = int(roi["end"]["x"] * orig_w)
            end_y = int(roi["end"]["y"] * orig_h)

            class_id = roi.get("id", 0) % len(self.colors)
            color = self.colors[class_id]

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)

            label = f"ID: {roi.get('id', 'N/A')}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img, (start_x, start_y - h - 10), (start_x + w, start_y), color, -1
            )
            cv2.putText(
                img,
                label,
                (start_x, start_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("ROIs on Original Image")
        plt.axis("off")
        plt.show()