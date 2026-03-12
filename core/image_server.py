import cv2
import os
import time
from processors import WorkspaceExtractor, aruco_factory


class ImageServer:
    def __init__(self, *image_paths, use_cameras=False, camera_ids=None):
        """
        Initializes with a list of image paths or camera IDs.
        Args:
            *image_paths: A variable number of paths to the images (used if use_cameras=False).
            use_cameras: If True, use live cameras instead of image files.
            camera_ids: List of camera IDs to use (e.g., [0, 2]). If None, uses [0, 1].
        """
        self.use_cameras = use_cameras
        self.image_paths = list(image_paths)
        self.ordered_paths = None
        self.zone_mapping = {}
        self.cameras = []
        self.camera_ids = camera_ids if camera_ids is not None else [0, 1]
        
        # Use absolute path or relative to project root
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'custom_markers.yaml')
        detector = aruco_factory()
        self.extractor = WorkspaceExtractor(detector)
        
        # Initialize cameras if needed
        if self.use_cameras:
            self._init_cameras()
    
    def _init_cameras(self):
        """Initialize cameras with proper settings."""
        self.cameras = []
        for cam_id in self.camera_ids:
            cam = None
            # Try V4L2 first, then fall back to CAP_ANY
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    cam = cv2.VideoCapture(cam_id, backend)
                    if cam.isOpened():
                        # Try to set MJPG if possible
                        try:
                            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                        except:
                            pass
                        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
                        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
                        
                        # Test if we can actually read a frame
                        ret, _ = cam.read()
                        if ret:
                            self.cameras.append(cam)
                            backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
                            print(f"Camera {cam_id} initialized successfully (using {backend_name})")
                            break
                        else:
                            cam.release()
                except Exception as e:
                    if cam is not None:
                        cam.release()
                    continue
            
            if cam is None or not cam.isOpened():
                print(f"Warning: Could not open camera {cam_id}")
    
    def _capture_with_temp_resolution(self, cam, width=4000, height=3000, warmup=3):
        """
        Capture a high-resolution frame temporarily, then restore original resolution.
        Args:
            cam: OpenCV VideoCapture object
            width: Temporary width for capture
            height: Temporary height for capture
            warmup: Number of warmup frames to skip
        Returns:
            numpy array: Captured frame
        """
        # old_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        # old_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # # Try to set higher resolution
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # # Check what resolution was actually set
        # actual_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        # actual_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # if actual_w != width or actual_h != height:
        #     print(f"  Note: Requested {width}x{height}, camera using {actual_w}x{actual_h}")
        
        # # Warmup frames
        for _ in range(warmup):
            cam.read()
        
        ret, frame = cam.read()
        
        # Restore original resolution
        # cam.set(cv2.CAP_PROP_FRAME_WIDTH, old_w)
        # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, old_h)
        
        # Clear buffer with one read
        # cam.read()
        
        if not ret or frame is None:
            raise RuntimeError("Camera did not capture the frame")
        
        return frame
    
    def setup(self):
        """
        Detects zones for all images/cameras and establishes the order.
        Should be called once during initialization.
        Returns:
            bool: True if both zones (1 and 2) are detected, False otherwise.
        """
        zone1_source = None
        zone2_source = None
        
        if self.use_cameras:
            # Detect zones for each camera
            for idx, cam in enumerate(self.cameras):
                try:
                    print(f"Capturing from camera {self.camera_ids[idx]}...")
                    img = self._capture_with_temp_resolution(cam, warmup=4)
                    zone_id = self._detect_zone(img)
                    cam_key = f"camera_{self.camera_ids[idx]}"
                    self.zone_mapping[cam_key] = zone_id
                    print(f"  Camera {self.camera_ids[idx]}: Zone {zone_id}")
                    
                    if zone_id == 1:
                        zone1_source = idx
                    elif zone_id == 2:
                        zone2_source = idx
                except Exception as e:
                    print(f"Warning: Could not capture from camera {self.camera_ids[idx]}: {e}")
            
            # Check if both zones are present
            if zone1_source is not None and zone2_source is not None:
                # Store ordered camera indices
                self.ordered_paths = [zone1_source, zone2_source]
                print(f"Setup complete: Zone 1 -> Camera {self.camera_ids[zone1_source]}, Zone 2 -> Camera {self.camera_ids[zone2_source]}")
                return True
            else:
                print(f"Setup failed: Missing zones. Zone 1: {zone1_source}, Zone 2: {zone2_source}")
                self.ordered_paths = list(range(len(self.cameras)))
                return False
        else:
            # Original file-based logic
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
                    zone1_source = path
                elif zone_id == 2:
                    zone2_source = path
            
            # Check if both zones are present
            if zone1_source and zone2_source:
                self.ordered_paths = [zone1_source, zone2_source]
                print(f"Setup complete: Zone 1 -> {zone1_source}, Zone 2 -> {zone2_source}")
                return True
            else:
                print(f"Setup failed: Missing zones. Zone 1: {zone1_source}, Zone 2: {zone2_source}")
                self.ordered_paths = self.image_paths
                return False
    
    def take_photos(self):
        """
        Captures images from cameras or loads from files.
        Returns images in the order established by setup() if available,
        otherwise in the original order.
        
        Returns:
            A list of images (as numpy arrays).
        """
        images = []
        
        if self.use_cameras:
            # Use ordered camera indices if setup was called
            indices_to_use = self.ordered_paths if self.ordered_paths else list(range(len(self.cameras)))
            
            for idx in indices_to_use:
                if idx < len(self.cameras):
                    try:
                        img = self._capture_with_temp_resolution(self.cameras[idx])
                        images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not capture from camera {self.camera_ids[idx]}: {e}")
        else:
            # Original file-based logic
            paths_to_use = self.ordered_paths if self.ordered_paths else self.image_paths
            
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
            dict: Dictionary mapping image paths/cameras to their detected zone_id
        """
        return self.zone_mapping.copy()
    
    def release(self):
        """Release all camera resources."""
        if self.use_cameras:
            for cam in self.cameras:
                cam.release()
            self.cameras = []
            print("All cameras released")


# Usage examples:
if __name__ == "__main__":
    # Example 1: Using files (original behavior)
    server_files = ImageServer("path/to/image1.jpg", "path/to/image2.jpg")
    server_files.setup()
    images = server_files.take_photos()
    
    # Example 2: Using cameras with default IDs [0, 1]
    server_cams = ImageServer(use_cameras=True)
    server_cams.setup()
    images = server_cams.take_photos()
    server_cams.release()
    
    # Example 3: Using specific camera IDs
    server_custom = ImageServer(use_cameras=True, camera_ids=[0, 2])
    server_custom.setup()
    images = server_custom.take_photos()
    server_custom.release()