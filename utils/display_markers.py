import cv2
import argparse
import sys
import os
import numpy as np

# Add the project root to the python path so we can import from processors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.workspace_extractor import WorkspaceExtractor
from processors.aruco_detector import aruco_factory


def display_markers():

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    ret, image = cap.read()
    
    for i in range(4):
        ret, image = cap.read()

    
    if image is None:
        print(f"Error: Could not read image from")
        return

    # Initialize extractor with default config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'custom_markers.yaml')
    extractor = WorkspaceExtractor(aruco_factory(config_path))
    
    print(f"Detecting markers in ...")
    markers = extractor.aruco_detector.detect_markers(image)
    
    print(f"Found {len(markers)} markers.")

    for marker in markers:
        # Draw corners
        corners = np.array(marker['corners'], dtype=np.int32)
        cv2.polylines(image, [corners], True, (0, 255, 0), 2)
        
        # Draw center
        center = tuple(map(int, marker['center']))
        cv2.circle(image, center, 5, (0, 0, 255), -1)
        
        # Draw ID and Info
        text = f"ID: {marker['id']} ({marker['dictionary']})"
        cv2.putText(image, text, (center[0] + 10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"Marker ID: {marker['id']}, Dictionary: {marker['dictionary']}, Center: {marker['center']}")

    # Resize for better viewing if image is too large
    height, width = image.shape[:2]
    max_height = 800
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, max_height))

    cv2.imshow("Detected Markers", image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and display custom markers in an image.")
    args = parser.parse_args()

    display_markers()
