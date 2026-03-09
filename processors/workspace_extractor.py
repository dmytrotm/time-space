import cv2
import cv2.aruco as aruco
import numpy as np
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from processors.aruco_detector import IArucoDetector


class WorkspaceExtractor:
    def __init__(self, aruco_detector: IArucoDetector):
        self.aruco_detector = aruco_detector
        self.logger = logging.getLogger(__name__)
        
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def extract_workspace(self, image):
        if image is None:
            print("Input image is None")
            return None

        found_markers = self.aruco_detector.detect_markers(image)

        if not found_markers or len(found_markers) < 4:
            return None

        marker_centers = np.array([marker["center"] for marker in found_markers])

        try:
            hull = cv2.convexHull(marker_centers.astype(np.float32))

            if len(hull) >= 4:
                boundary_points = hull.reshape(-1, 2)[:4]
            else:
                x_coords = marker_centers[:, 0]
                y_coords = marker_centers[:, 1]

                top_left_idx = np.argmin(x_coords + y_coords)
                top_right_idx = np.argmin(-x_coords + y_coords)
                bottom_right_idx = np.argmin(-x_coords - y_coords)
                bottom_left_idx = np.argmin(x_coords - y_coords)

                corner_indices = list(
                    set(
                        [top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]
                    )
                )

                if len(corner_indices) < 4:
                    distances_from_center = np.sqrt(
                        (marker_centers[:, 0] - np.mean(x_coords)) ** 2
                        + (marker_centers[:, 1] - np.mean(y_coords)) ** 2
                    )
                    corner_indices = np.argsort(distances_from_center)[-4:]

                boundary_points = marker_centers[corner_indices[:4]]

            corrected_image = self.four_point_transform(image, boundary_points)

            return corrected_image

        except Exception as e:
            print(f"Error during perspective correction: {e}")
            return None