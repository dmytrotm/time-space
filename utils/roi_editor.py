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
        self.rois = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.roi_counter = 1

        # Colors
        self.roi_color = (0, 255, 0)  # Green for ROI boxes
        self.text_color = (255, 255, 255)  # White for text
        self.preview_color = (255, 255, 0) # Yellow for preview

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
        print("Controls:")
        print("  - Click and drag to draw an ROI")
        print("  - Press 's' to save ROIs")
        print("  - Press 'c' to clear all ROIs")
        print("  - Press 'd' to delete last ROI")
        print("  - Press 'q' or ESC to quit")

    def load_rois(self):
        """Load existing ROIs from JSON file"""
        if os.path.exists(self.roi_file):
            try:
                with open(self.roi_file, 'r') as f:
                    data = json.load(f)
                    self.rois = data.get('rois', [])
                    if self.rois:
                        self.roi_counter = max(roi.get('id', 0) for roi in self.rois) + 1
                    else:
                        self.roi_counter = 1
                print(f"Loaded {len(self.rois)} existing ROIs from {self.roi_file}")
            except Exception as e:
                print(f"Error loading ROIs: {e}")
                self.rois = []

    def save_rois(self):
        """Save ROIs to JSON file"""
        data = {
            "image_path": self.image_path,
            "image_size": {"width": self.width, "height": self.height},
            "rois": self.rois
        }

        try:
            with open(self.roi_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.rois)} ROIs to {self.roi_file}")
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
        # Ensure start_point is top-left and end_point is bottom-right
        x1, y1 = start_pixel
        x2, y2 = end_pixel
        start_x = min(x1, x2)
        start_y = min(y1, y2)
        end_x = max(x1, x2)
        end_y = max(y1, y2)

        rel_start_x, rel_start_y = self.pixel_to_relative(start_x, start_y)
        rel_end_x, rel_end_y = self.pixel_to_relative(end_x, end_y)

        roi = {
            "id": self.roi_counter,
            "start": {"x": rel_start_x, "y": rel_start_y},
            "end": {"x": rel_end_x, "y": rel_end_y}
        }

        self.rois.append(roi)
        self.roi_counter += 1
        print(f"Added ROI {roi['id']}")

    def draw_rois(self):
        """Draw all ROIs on the display image"""
        self.display_image = self.original_image.copy()

        for roi in self.rois:
            start_x, start_y = self.relative_to_pixel(roi["start"]["x"], roi["start"]["y"])
            end_x, end_y = self.relative_to_pixel(roi["end"]["x"], roi["end"]["y"])

            cv2.rectangle(self.display_image, (start_x, start_y), (end_x, end_y), self.roi_color, 2)
            cv2.putText(self.display_image, f"ROI {roi['id']}",
                       (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

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
                cv2.rectangle(temp_image, self.start_point, self.end_point, self.preview_color, 2)
                cv2.imshow('Interactive ROI Editor', temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                self.add_roi(self.start_point, self.end_point)
                self.draw_rois()
                cv2.imshow('Interactive ROI Editor', self.display_image)


    def run(self):
        """Main loop"""
        cv2.namedWindow('Interactive ROI Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Interactive ROI Editor', self.mouse_callback)

        self.draw_rois()

        while True:
            cv2.imshow('Interactive ROI Editor', self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Save
                self.save_rois()
            elif key == ord('c'):  # Clear all ROIs
                self.rois = []
                self.roi_counter = 1
                self.draw_rois()
                print("Cleared all ROIs")
            elif key == ord('d'):  # Delete last ROI
                if self.rois:
                    deleted_roi = self.rois.pop()
                    print(f"Deleted ROI {deleted_roi['id']}")
                    self.draw_rois()

        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Interactive ROI Editor for rectangular ROIs.')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--rois', default='rois_interactive.json', help='Path to ROIs JSON file')

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
    # python interactive_roi_editor.py --image Z2_2_1.png --rois rois_interactive.json
    main()