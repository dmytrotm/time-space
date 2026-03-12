import cv2
import argparse
import sys
import os
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processors.workspace_extractor import WorkspaceExtractor
from processors.aruco_detector import aruco_factory


def display_markers():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)

    for _ in range(4):
        cap.read()

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'custom_markers.yaml')
    extractor = WorkspaceExtractor(aruco_factory( track_time=False))
    
    print("Початок трансляції... Натисніть 'q' у вікні для виходу.")

    prev_time = time.time()
    last_console_print = time.time()
    print_interval = 2.0 

    while True:
        ret, image = cap.read()
        
        if not ret or image is None:
            print("Помилка: Не вдалося отримати кадр з камери.")
            break

        current_time = time.time()
        
        markers = extractor.aruco_detector.detect_markers(image)
        
        delta_time = current_time - prev_time
        fps = 1.0 / delta_time if delta_time > 0 else 0.0
        prev_time = current_time

        if current_time - last_console_print >= print_interval:
            print(f"FPS: {fps:.1f} | Знайдено маркерів: {len(markers)}")
            last_console_print = current_time

        for marker in markers:
            corners = np.array(marker['corners'], dtype=np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 6)
            
            center = tuple(map(int, marker['center']))
            cv2.circle(image, center, 15, (0, 0, 255), -1)
            
            dict_name = marker.get('dictionary', 'unknown')
            text = f"ID: {marker['id']} ({dict_name})"
            cv2.putText(image, text, (center[0] + 20, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        cv2.putText(image, f"FPS: {fps:.1f}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 6)

        height, width = image.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            new_width = int(width * scale)
            display_image = cv2.resize(image, (new_width, max_height))
        else:
            display_image = image

        cv2.imshow("Detected Markers", display_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous detection and display of custom markers in a video stream.")
    args = parser.parse_args()

    display_markers()