import cv2
import os
from processors.workspace_extractor import WorkspaceExtractor

class ImageServer:
    def __init__(self, *image_paths):
        """
        Initializes with a list of image paths.
        Args:
            *image_paths: A variable number of paths to the images.
        """
        self.image_paths = list(image_paths)
        self.ordered_paths = None  # Will be set during setup
        self.zone_mapping = {}  # Maps path -> zone_id
        
        # Use absolute path or relative to project root
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'custom_markers.yaml')
        self.extractor = WorkspaceExtractor(custom_yaml_path=config_path)
    
    def setup(self):
        """
        Detects zones for all images and establishes the order.
        Should be called once during initialization.
        Returns:
            bool: True if both zones (1 and 2) are detected, False otherwise.
        """
        zone1_path = None
        zone2_path = None
        
        # Detect zones for each image
        for path in self.image_paths:
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                continue
            
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read image at {path}")
                continue
            
            zone_id = self._detect_zone(img)
            self.zone_mapping[path] = zone_id
            
            if zone_id == 1:
                zone1_path = path
            elif zone_id == 2:
                zone2_path = path
        
        # Check if both zones are present
        if zone1_path and zone2_path:
            # Set the order: Zone 1 first, then Zone 2
            self.ordered_paths = [zone1_path, zone2_path]
            print(f"Setup complete: Zone 1 -> {zone1_path}, Zone 2 -> {zone2_path}")
            return True
        else:
            print(f"Setup failed: Missing zones. Zone 1: {zone1_path}, Zone 2: {zone2_path}")
            # Fallback to original order if zones not properly detected
            self.ordered_paths = self.image_paths
            return False
    
    def take_photos(self):
        """
        Loads images from the stored paths.
        Returns images in the order established by setup() if available,
        otherwise in the original order provided during initialization.
        
        Returns:
            A list of images (as numpy arrays).
        """
        # Use ordered paths if setup was called, otherwise use original order
        paths_to_use = self.ordered_paths if self.ordered_paths else self.image_paths
        
        images = []
        for path in paths_to_use:
            if not os.path.exists(path):
                continue
            
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read image at {path}")
        
        return images
    
    def _detect_zone(self, image):
        """
        Detects which zone an image belongs to based on ArUco markers.
        
        Args:
            image: numpy array of the image
            
        Returns:
            int or None: 1 for zone 1, 2 for zone 2, None if undetermined
        """
        markers = self.extractor.detect_markers(image)
        zone1_count = 0
        zone2_count = 0
        
        for marker in markers:
            dictionary = marker.get("dictionary", "")
            if dictionary == "zone1_markers":
                zone1_count += 1
            elif dictionary == "zone2_markers":
                zone2_count += 1
        
        total_markers = len(markers)
        
        # Need at least 4 markers total and exactly 2 exclusive markers for one zone
        if total_markers >= 4:
            if zone1_count == 2 and zone2_count == 0:
                return 1
            elif zone2_count == 2 and zone1_count == 0:
                return 2
        
        return None
    
    def get_zone_info(self):
        """
        Returns the current zone mapping.
        
        Returns:
            dict: Dictionary mapping image paths to their detected zone_id
        """
        return self.zone_mapping.copy()