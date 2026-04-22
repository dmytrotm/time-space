import cv2
import os
import time
from processors import WorkspaceExtractor, aruco_factory

class ImageServer:
    def __init__(self, *image_paths, use_cameras=False, camera_ids=None):
        """
        Initializes with a list of image paths or camera IDs/Paths.
        """
        self.use_cameras = use_cameras
        self.image_paths = list(image_paths)
        self.ordered_paths = None
        self.zone_mapping = {}
        
        # Default to scanning first 10 camera IDs
        default_cams = list(range(10))
        self.camera_ids = camera_ids if camera_ids is not None else default_cams
        self.cameras = []
        
        # Ініціалізація ArUco
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'custom_markers.yaml')
        detector = aruco_factory()
        self.extractor = WorkspaceExtractor(detector)
        
        if self.use_cameras:
            self._init_cameras()
    
    def _connect_single_camera(self, cam_id):
        """Спроба підключити одну камеру із захистом від зависань"""
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            try:
                cam = cv2.VideoCapture(cam_id, backend)
                if cam.isOpened():
                    try:
                        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except:
                        pass
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
                    cam.set(cv2.CAP_PROP_FPS, 5) 
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = cam.read()
                    if ret:
                        backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
                        short_name = str(cam_id).split('/')[-1] if isinstance(cam_id, str) else cam_id
                        print(f"Camera [{short_name}] connected successfully ({backend_name})")
                        return cam
                    else:
                        cam.release()
            except Exception:
                if 'cam' in locals() and cam is not None:
                    cam.release()
                continue
        return None

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
    
    def _capture_with_temp_resolution(self, cam):
        for _ in range(6):
            cam.grab()  
        
        ret, frame = cam.retrieve() 
        
        if not ret or frame is None:
            raise RuntimeError("Camera lost connection")
        
        return frame
    
    def take_photos(self):
        """Робить знімки з гарячим перепідключенням, якщо камера відвалилась"""
        images = []
        
        if self.use_cameras:
            indices_to_use = self.ordered_paths if self.ordered_paths else list(range(len(self.camera_ids)))
            
            for idx in indices_to_use:
                if idx >= len(self.camera_ids):
                    continue
                
                cam_id = self.camera_ids[idx]
                cam = self.cameras[idx]
                short_name = str(cam_id).split('/')[-1] if isinstance(cam_id, str) else cam_id
                
                if cam is None:
                    print(f"Info: Camera [{short_name}] is offline. Reconnecting...")
                    cam = self._connect_single_camera(cam_id)
                    self.cameras[idx] = cam
                
                if cam is not None:
                    try:
                        img = self._capture_with_temp_resolution(cam)
                        images.append(img)
                    except Exception as e:
                        print(f"Warning: Camera [{short_name}] dropped during capture. Reconnecting...")
                        cam.release() #
                        
                        cam = self._connect_single_camera(cam_id)
                        self.cameras[idx] = cam
                        
                        if cam is not None:
                            try:
                                print(f"Success: Camera [{short_name}] reconnected. Taking photo...")
                                img = self._capture_with_temp_resolution(cam, warmup=5)
                                images.append(img)
                            except Exception as e2:
                                print(f"Error: Camera [{short_name}] failed again: {e2}")
                        else:
                            print(f"Error: Could not reconnect camera [{short_name}]")
        else:
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
        if total_markers >= 4:
            if zone1_count == 2 and zone2_count == 0:
                return 1
            elif zone2_count == 2 and zone1_count == 0:
                return 2
        return None
    
    def get_zone_info(self):
        return self.zone_mapping.copy()
    
    def release(self):
        if self.use_cameras:
            for cam in self.cameras:
                if cam is not None:
                    cam.release()
            self.cameras = []
            print("All cameras released")

if __name__ == "__main__":
    server_cams = ImageServer(use_cameras=True)
    images = server_cams.take_photos()
    print(f"Успішно зроблено {len(images)} фотографій.")
    server_cams.release()