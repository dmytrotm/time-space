import cv2
import cv2.aruco as aruco
import numpy as np
import os
import logging

from concurrent.futures import ThreadPoolExecutor

class IArucoDetector:
    def detect_markers(self, image):
        pass

def aruco_factory(custom_yaml_path=None, track_time=True):
    if custom_yaml_path:
        detector =  CustomArucoDetector(custom_yaml_path)
    else:
        detector = ArucoDetector()
    if track_time:
        detector = ArucoDetectorTime(detector)
    return detector


class CustomArucoDetector(IArucoDetector):
    def __init__(self, custom_yaml_path="custom_markers.yaml"):
        self.custom_yaml_path = custom_yaml_path
        self.logger = logging.getLogger(__name__)
        self.detectors = []
        self.load_dictionaries()

    def load_dictionaries(self):
        aruco_dict_list = {}

        all_dicts = {}

        for name, dict_key in aruco_dict_list.items():
            try:
                all_dicts[name] = aruco.getPredefinedDictionary(dict_key)
            except Exception as e:
                print(f"Error loading predefined dictionary {name}: {e}")

        custom_dicts = {}
        if os.path.exists(self.custom_yaml_path):
            try:
                fs = cv2.FileStorage(self.custom_yaml_path, cv2.FILE_STORAGE_READ)
                if fs.isOpened():
                    root = fs.root()
                    for name in root.keys():
                        dict_node = fs.getNode(name)
                        if not dict_node.empty():
                            try:
                                dictionary = cv2.aruco.Dictionary()
                                if dictionary.readDictionary(dict_node):
                                    custom_dicts[name] = dictionary
                            except Exception as e:
                                print(f"Error loading custom dictionary {name}: {e}")
                                continue
                    fs.release()
                else:
                    print(f"Could not open YAML file: {self.custom_yaml_path}")
            except Exception as e:
                print(f"Error reading custom dictionary file: {e}")
        else:
            print(f"Custom dictionary file not found: {self.custom_yaml_path}")

        all_dicts.update(custom_dicts)

        for name, aruco_dict in all_dicts.items():
            for border in [1, 2]:
                if aruco_dict.markerSize != 6 and border == 2:
                    continue
                parameters = aruco.DetectorParameters()
                parameters.minMarkerPerimeterRate = 0.07
                parameters.minDistanceToBorder = 0
                parameters.markerBorderBits = border
                detector = aruco.ArucoDetector(aruco_dict, parameters)
                self.detectors.append((name, border, detector))
                print(f"Added detector: {name}, Border: {border}, Size: {aruco_dict.markerSize}")

        self.logger.info(f"Total detectors created: {len(self.detectors)}")


    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        all_found_markers = []

        def detect_task(args):
            name, border, detector = args
            corners, ids, rejected = detector.detectMarkers(gray)
            found_markers = []
            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids):
                    corner_points = corners[i][0]
                    center = np.mean(corner_points, axis=0)
                    area = cv2.contourArea(corners[i])
                    found_markers.append(
                        {
                            "dictionary": name,
                            "id": int(marker_id[0]),
                            "border": border,
                            "center": center,
                            "corners": corners[i].tolist(),
                            "area": area,
                        }
                    )
            return found_markers

        with ThreadPoolExecutor() as executor:
            results = executor.map(detect_task, self.detectors)

        for result in results:
            all_found_markers.extend(result)

        unique_markers = self.remove_duplicate_markers(all_found_markers)
        return unique_markers
        
    def remove_duplicate_markers(self, markers):
        if not markers:
            return []

        marker_groups = {}
        for marker in markers:
            marker_key = (marker["dictionary"], marker["id"])
            if marker_key not in marker_groups:
                marker_groups[marker_key] = []
            marker_groups[marker_key].append(marker)

        unique_markers = []
        for marker_id, group in marker_groups.items():
            if len(group) == 1:
                unique_markers.append(group[0])
            else:
                best_marker = max(group, key=lambda m: m["area"])
                unique_markers.append(best_marker)

        return unique_markers



class ArucoDetector(IArucoDetector):
    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    
    def detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        marker_list = []
        
        if ids is not None:
            for i in range(len(ids)):
                marker_corners = corners[i][0]
                
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                marker_data = {
                    "id": int(ids[i][0]),
                    "center": (center_x, center_y),
                    "corners": marker_corners.tolist()
                }
                marker_list.append(marker_data)
                
        return marker_list

import time

class ArucoDetectorTime(IArucoDetector):
    def __init__(self, detector: IArucoDetector):
        self._detector = detector

    def detect_markers(self, image):
        start_time = time.perf_counter()
        
        result = self._detector.detect_markers(image)
        
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        print(f"[DEBUG] Detection time: {execution_time:.4f} seconds")
        
        return result