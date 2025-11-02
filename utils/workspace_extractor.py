import cv2
import cv2.aruco as aruco
import numpy as np
import os
import logging


class WorkspaceExtractor:
    def __init__(self, custom_yaml_path="custom_markers.yaml"):
        self.custom_yaml_path = custom_yaml_path
        self.all_dicts = {}
        self.logger = logging.getLogger(__name__)
        self.load_dictionaries()

    def load_dictionaries(self):
        aruco_dict_list = {
            "DICT_6X6_250": aruco.DICT_6X6_250,
        }

        for name, dict_key in aruco_dict_list.items():
            try:
                self.all_dicts[name] = aruco.getPredefinedDictionary(dict_key)
            except Exception as e:
                self.logger.error(f"Error loading predefined dictionary {name}: {e}")

        custom_dicts = {}
        if os.path.exists(self.custom_yaml_path):
            try:
                fs = cv2.FileStorage(self.custom_yaml_path, cv2.FILE_STORAGE_READ)
                if fs.isOpened():
                    dict_names = [
                        "cust_dictionary4",
                        "cust_dictionary5",
                        "cust_dictionary6",
                        "cust_dictionary8",
                    ]
                    for name in dict_names:
                        dict_node = fs.getNode(name)
                        if not dict_node.empty():
                            try:
                                dictionary = cv2.aruco.Dictionary()
                                if dictionary.readDictionary(dict_node):
                                    custom_dicts[name] = dictionary
                            except Exception as e:
                                self.logger.error(
                                    f"Error loading custom dictionary {name}: {e}"
                                )
                                continue
                    fs.release()
                else:
                    self.logger.warning(
                        f"Could not open YAML file: {self.custom_yaml_path}"
                    )
            except Exception as e:
                self.logger.error(f"Error reading custom dictionary file: {e}")
        else:
            self.logger.warning(
                f"Custom dictionary file not found: {self.custom_yaml_path}"
            )

        self.all_dicts.update(custom_dicts)
        self.logger.info(f"Total dictionaries loaded: {len(self.all_dicts)}")

    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_found_markers = []

        for name, aruco_dict in self.all_dicts.items():
            try:
                parameters = aruco.DetectorParameters()
                parameters.minMarkerPerimeterRate = 0.07
                parameters.minDistanceToBorder = 0

                for border in [1, 2]:
                    if aruco_dict.markerSize != 6 and border == 2:
                        continue

                    parameters.markerBorderBits = border
                    detector = aruco.ArucoDetector(aruco_dict, parameters)
                    corners, ids, rejected = detector.detectMarkers(gray)

                    if ids is not None and len(ids) > 0:
                        for i, marker_id in enumerate(ids):
                            corner_points = corners[i][0]
                            center = np.mean(corner_points, axis=0)
                            area = cv2.contourArea(corners[i])

                            all_found_markers.append(
                                {
                                    "dictionary": name,
                                    "id": int(marker_id[0]),
                                    "border": border,
                                    "center": center,
                                    "corners": corners[i].tolist(),
                                    "area": area,
                                }
                            )

            except Exception as e:
                self.logger.error(f"Error detecting markers in dictionary {name}: {e}")
                continue

        unique_markers = self.remove_duplicate_markers(all_found_markers)
        return unique_markers

    def remove_duplicate_markers(self, markers):
        if not markers:
            return []

        marker_groups = {}
        for marker in markers:
            marker_id = marker["id"]
            if marker_id not in marker_groups:
                marker_groups[marker_id] = []
            marker_groups[marker_id].append(marker)

        unique_markers = []
        for marker_id, group in marker_groups.items():
            if len(group) == 1:
                unique_markers.append(group[0])
            else:
                best_marker = max(group, key=lambda m: m["area"])
                unique_markers.append(best_marker)

        return unique_markers

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
            self.logger.error("Input image is None")
            return None

        found_markers = self.detect_markers(image)

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
            self.logger.error(f"Error during perspective correction: {e}")
            return None
