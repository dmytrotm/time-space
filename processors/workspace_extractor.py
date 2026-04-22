import cv2
import numpy as np
import logging
from processors.aruco_detector import IArucoDetector

class WorkspaceExtractor:
    def __init__(self, aruco_detector: IArucoDetector, defined_zones: dict = None):
        self.aruco_detector = aruco_detector
        self.logger = logging.getLogger(__name__)
        
        self.defined_zones = defined_zones if defined_zones else {}#1,2,3,6

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 

        return rect

    def detect_zone(self, image):
        """
        Шукає маркери і перевіряє, чи співпадають вони з якоюсь із заданих зон.
        Повертає (zone_id, rect), де rect - це впорядковані 4 точки.
        Якщо зона не знайдена, повертає (-1, None).
        """
        if image is None:
            return -1, None

        found_markers = self.aruco_detector.detect_markers(image)
        if not found_markers:
            return -1, None

        detected_dict = {marker["id"]: marker for marker in found_markers}
        detected_ids = set(detected_dict.keys())

        for zone_id, zone_marker_ids in self.defined_zones.items():
            zone_marker_ids_set = set(zone_marker_ids)
            
            if zone_marker_ids_set.issubset(detected_ids):
                zone_markers = [detected_dict[m_id] for m_id in zone_marker_ids]
                marker_centers = np.array([m["center"] for m in zone_markers], dtype=np.float32)

                rect = self.order_points(marker_centers)
                
                return zone_id, rect

        return -1, None

    def four_point_transform(self, image, rect):
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
        """
        Комплексний метод: знаходить зону і одразу вирізає її.
        """
        zone_id, rect = self.detect_zone(image)
        
        if zone_id == -1 or rect is None:
            return -1 , None

        corrected_image = self.four_point_transform(image, rect)
        return zone_id,corrected_image