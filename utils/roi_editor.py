import cv2
import numpy as np
import json
import os
import argparse


class InteractiveROIEditor:
    def __init__(self, image_path, roi_file="rois.json"):
        self.image_path = image_path
        self.roi_file = roi_file
        self.original_image = None
        self.display_image = None
        self.categories = ["tapes", "grounding", "wires", "connectors", "label"]
        self.current_category = "tapes"
        self.rois_dict = {cat: [] for cat in self.categories}
        self.drawing = False
        self.start_point = None
        self.end_point = None

        # Category mapping for shapes
        self.category_shapes = {
            "tapes": "square",
            "grounding": "rectangle",
            "wires": "rectangle",
            "connectors": "square",
            "label": "square",
        }

        # Colors for each category
        self.category_colors = {
            "tapes": (0, 255, 0),  # Green
            "grounding": (255, 0, 0),  # Blue
            "wires": (0, 0, 255),  # Red
            "connectors": (255, 0, 255),  # Magenta
            "label": (0, 255, 255),  # Cyan
        }
        self.text_color = (255, 255, 255)  # White for text
        self.preview_color = (255, 255, 0)  # Yellow for preview

        self.load_image()
        self.load_rois()

    def load_image(self):
        """Load the image"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {self.image_path}")

        self.height, self.width = self.original_image.shape[:2]
        self.display_image = self.original_image.copy()

        print(f"Image loaded: {self.width}x{self.height}")
        print("\nControls:")
        print("  - Click and drag to draw an ROI")
        print(f"  - Keys 1-{len(self.categories)}: Switch category:")
        for i, cat in enumerate(self.categories, 1):
            print(f"    {i}: {cat} ({self.category_shapes[cat]})")
        print("  - Press 's' to save ROIs")
        print("  - Press 'c' to clear current category ROIs")
        print("  - Press 'd' to delete last ROI in current category")
        print("  - Press 'q' or ESC to quit")

    def load_rois(self):
        """Load existing ROIs from JSON file"""
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, "r") as f:
                    data = json.load(f)
                    # Support both new categorical format and legacy flat 'rois' format
                    loaded_any = False
                    for cat in self.categories:
                        if cat in data:
                            val = data[cat]
                            if isinstance(val, list):
                                self.rois_dict[cat] = val
                            else:
                                # Handle single object case (e.g. 'label' in rois_z1.json)
                                self.rois_dict[cat] = [val]
                            loaded_any = True

                    if not loaded_any and "rois" in data:
                        # Fallback to old format, putting everything in 'tapes' or first category
                        self.rois_dict[self.categories[0]] = data["rois"]
                        print(f"Loaded legacy ROIs into {self.categories[0]}")

                count = sum(len(rois) for rois in self.rois_dict.values())
                print(f"Loaded {count} existing ROIs from {self.roi_file}")
            except Exception as e:
                print(f"Error loading ROIs: {e}")

    def save_rois(self):
        """Save ROIs to JSON file"""
        data = {
            "image_size": {"width": self.width, "height": self.height},
        }
        for cat, rois in self.rois_dict.items():
            if not rois:
                continue
            if cat == "label" and len(rois) == 1:
                # Save as single object for backward compatibility if it's 'label' and only 1
                data[cat] = rois[0]
            else:
                data[cat] = rois

        try:
            with open(self.roi_file, "w") as f:
                json.dump(data, f, indent=2)
            count = sum(len(rois) for rois in self.rois_dict.values())
            print(f"Saved {count} ROIs to {self.roi_file}")
        except Exception as e:
            print(f"Error saving ROIs: {e}")

    def pixel_to_relative(self, x, y):
        """Convert pixel coordinates to relative coordinates (0-1)"""
        return x / self.width, y / self.height

    def relative_to_pixel(self, rel_x, rel_y):
        """Convert relative coordinates to pixel coordinates"""
        return int(rel_x * self.width), int(rel_y * self.height)

    def add_roi(self, start_pixel, end_pixel):
        """Add a new ROI"""
        x1, y1 = start_pixel
        x2, y2 = end_pixel

        # Calculate max ID in current category
        existing_ids = [roi.get("id", -1) for roi in self.rois_dict[self.current_category]]
        next_id = max(existing_ids) + 1 if existing_ids else 0

        shape = self.category_shapes[self.current_category]

        if shape == "square":
            # Calculate center and half-size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            half_size_px = max(abs(x2 - x1), abs(y2 - y1)) / 2

            rel_center_x, rel_center_y = self.pixel_to_relative(center_x, center_y)
            rel_half_size = half_size_px / self.width  # Relative to width

            roi = {
                "id": next_id,
                "center": {"x": rel_center_x, "y": rel_center_y},
                "relative_half_size": rel_half_size,
            }
        else:  # rectangle
            start_x = min(x1, x2)
            start_y = min(y1, y2)
            end_x = max(x1, x2)
            end_y = max(y1, y2)

            rel_start_x, rel_start_y = self.pixel_to_relative(start_x, start_y)
            rel_end_x, rel_end_y = self.pixel_to_relative(end_x, end_y)

            roi = {
                "id": next_id,
                "start": {"x": rel_start_x, "y": rel_start_y},
                "end": {"x": rel_end_x, "y": rel_end_y},
            }

        self.rois_dict[self.current_category].append(roi)
        print(f"Added {self.current_category} ROI {roi['id']}")

    def draw_rois(self):
        """Draw all ROIs on the display image"""
        self.display_image = self.original_image.copy()

        for cat, rois in self.rois_dict.items():
            color = self.category_colors[cat]
            thickness = 3 if cat == self.current_category else 1
            shape = self.category_shapes[cat]

            for roi in rois:
                if shape == "square":
                    cx, cy = self.relative_to_pixel(roi["center"]["x"], roi["center"]["y"])
                    # Use width for relative half size to match add_roi
                    half_px = int(roi["relative_half_size"] * self.width)
                    start_pt = (cx - half_px, cy - half_px)
                    end_pt = (cx + half_px, cy + half_px)
                else:
                    start_pt = self.relative_to_pixel(roi["start"]["x"], roi["start"]["y"])
                    end_pt = self.relative_to_pixel(roi["end"]["x"], roi["end"]["y"])

                cv2.rectangle(self.display_image, start_pt, end_pt, color, thickness)
                
                label = f"{cat} {roi['id']}"
                if "name" in roi:
                    label += f": {roi['name']}"
                    
                cv2.putText(
                    self.display_image,
                    label,
                    (start_pt[0], start_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        # Draw current category label
        overlay_text = f"Current Category: {self.current_category} ({self.category_shapes[self.current_category]})"
        cv2.putText(
            self.display_image,
            overlay_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            self.category_colors[self.current_category],
            2,
        )

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                temp_image = self.display_image.copy()
                
                if self.category_shapes[self.current_category] == "square":
                    # For square, we visualize it as dynamic square from start to end
                    cx = (self.start_point[0] + self.end_point[0]) // 2
                    cy = (self.start_point[1] + self.end_point[1]) // 2
                    half_px = max(abs(x - self.start_point[0]), abs(y - self.start_point[1])) // 2
                    cv2.rectangle(
                        temp_image, 
                        (cx - half_px, cy - half_px), 
                        (cx + half_px, cy + half_px), 
                        self.preview_color, 2
                    )
                else:
                    cv2.rectangle(
                        temp_image, self.start_point, self.end_point, self.preview_color, 2
                    )
                cv2.imshow("Interactive ROI Editor", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                self.add_roi(self.start_point, self.end_point)
                self.draw_rois()
                cv2.imshow("Interactive ROI Editor", self.display_image)

    def run(self):
        """Main loop"""
        cv2.namedWindow("Interactive ROI Editor", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Interactive ROI Editor", self.mouse_callback)

        self.draw_rois()

        while True:
            cv2.imshow("Interactive ROI Editor", self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q or ESC
                break
            elif key == ord("s"):  # Save
                self.save_rois()
            elif key >= ord("1") and key <= ord(str(len(self.categories))):
                idx = int(chr(key)) - 1
                self.current_category = self.categories[idx]
                print(f"Switched category to: {self.current_category}")
                self.draw_rois()
            elif key == ord("c"):  # Clear current category ROIs
                self.rois_dict[self.current_category] = []
                self.draw_rois()
                print(f"Cleared all ROIs in {self.current_category}")
            elif key == ord("d"):  # Delete last ROI in current category
                if self.rois_dict[self.current_category]:
                    deleted_roi = self.rois_dict[self.current_category].pop()
                    print(f"Deleted {self.current_category} ROI {deleted_roi['id']}")
                    self.draw_rois()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI Editor for rectangular ROIs."
    )
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--rois", default="rois_interactive.json", help="Path to ROIs JSON file"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    try:
        editor = InteractiveROIEditor(args.image, args.rois)
        editor.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # To run from command line:
    # python utils/roi_editor.py --image /Users/dmytro/Desktop/tns_code_16.3/time-space/zones/2/workspace_1773743484.png --rois /Users/dmytro/Desktop/tns_code_16.3/time-space/configs/rois_z2.json
    main()
