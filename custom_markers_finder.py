import cv2

import numpy as np
import os
from collections import defaultdict

class MarkerDetector:

    def __init__(self) -> None:
        self.dictionaries = []

    def load_dictionaries(self, file_path, dict_names):
        fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError(f"Не вдалося відкрити файл {file_path}")

        dictionaries = []
        for name in dict_names:
            dict_node = fs.getNode(name)
            if dict_node.empty():
                continue

            dictionary = cv2.aruco.Dictionary()
            if dictionary.readDictionary(dict_node):
                dictionaries.append(dictionary)

        fs.release()
        self.dictionaries = dictionaries

    def set_dictionaries(self, dictionaries):
        if dictionaries is []:
            self.dictionaries = dictionaries
        else :
            self.dictionaries = [dictionaries]

    @staticmethod
    def compute_center(corner):
        pts = corner[0]
        return np.mean(pts, axis=0)

    @staticmethod
    def marker_size(corner):
        pts = corner[0]
        width = np.linalg.norm(pts[0] - pts[1])
        height = np.linalg.norm(pts[1] - pts[2])
        return (width + height) / 2

    def filter_duplicate_markers(self, corners, ids, border_info):
        from collections import defaultdict

        grouped = defaultdict(list)
        for i, full_id in enumerate(ids):
            true_id = full_id[0] % 10000
            grouped[true_id].append({
                'id': full_id,
                'true_id': true_id,
                'corner': corners[i],
                'borderBits': border_info[i],
                'index': i
            })

        keep_indices = []

        for group in grouped.values():
            if len(group) == 1:
                keep_indices.append(group[0]['index'])
            else:
                centers = [self.compute_center(g['corner']) for g in group]
                sizes = [self.marker_size(g['corner']) for g in group]
                # Визначаємо динамічну мінімальну відстань — половина середнього розміру
                avg_size = np.mean(sizes)
                dynamic_min_distance = avg_size * 0.5

                all_far = True
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < dynamic_min_distance:
                            all_far = False
                            break
                    if not all_far:
                        break

                if all_far:
                    keep_indices.extend(g['index'] for g in group)
                else:
                    # Залишаємо лише з меншим borderBits
                    chosen = max(group, key=lambda g: g['borderBits'])
                    keep_indices.append(chosen['index'])

        filtered_corners = [corners[i] for i in keep_indices]
        filtered_ids = np.array([ids[i] for i in keep_indices], dtype=np.int32)
        return filtered_corners, filtered_ids


    def detect_all_markers(self, image, allow_partial=True):
        all_corners = []
        all_ids = []
        all_border_info = []
        all_rejected = []

        for border in [1,2]:
            parameters = cv2.aruco.DetectorParameters()
            parameters.minMarkerPerimeterRate = 0.07
            parameters.minDistanceToBorder = 0
            parameters.markerBorderBits = border
            
            if allow_partial:
                # Parameters for better partial marker detection
                parameters.polygonalApproxAccuracyRate = 0.15  # More tolerant to shape changes
                parameters.minCornerDistanceRate = 0.01        # Closer corners allowed
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                parameters.cornerRefinementWinSize = 5
                parameters.errorCorrectionRate = 0.7           # More error correction
                parameters.minMarkerDistanceRate = 0.01 


            for i, dictionary in enumerate(self.dictionaries):
                if dictionary.markerSize !=6 and border == 2:
                    continue
                detector = cv2.aruco.ArucoDetector(dictionary, parameters)
                corners, ids, rejected = detector.detectMarkers(image)

                if ids is not None:
                    ids += i * 1000 + (border-1)*10000
                    all_corners.extend(corners)
                    all_ids.extend(ids)
                    all_border_info.extend([border] * len(ids))
                if rejected is not None:
                        all_rejected.extend(rejected)
        if all_ids:
            filtered_corners, filtered_ids = self.filter_duplicate_markers(all_corners, np.array(all_ids), all_border_info)
            return filtered_corners, all_rejected, filtered_ids, None
        else:
            return [], all_rejected, None, None

if __name__ == "__main__":
    m = MarkerDetector()
    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    m.load_dictionaries("custom_dictionaries.yml",dict_names)


    image = cv2.imread("jpg_discovery/IMG_1689.jpg")

    corners,rejected, ids,_ = m.detect_all_markers(image)
    # Візуалізація результатів
    if ids is not None:
        output = image.copy()
        cv2.aruco.drawDetectedMarkers(output, corners, ids)
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(output, rejected, borderColor=(0, 0, 255))
        # Вивід інформації
        print(f"Знайдено {len(ids)} маркерів:")
        for i, marker_id in enumerate(ids):
            print(f"ID: {marker_id[0]}, Словник: {marker_id[0] // 1000}")


        cv2.imwrite("detected.jpg", output)

    else:
        print("Маркери не знайдені")

