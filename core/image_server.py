import cv2
import os
import time
import threading
from processors import WorkspaceExtractor, aruco_factory
from configs.config import ZONES_DICT_WS1, ZONES_DICT_WS2

_ISP_PREFIXES = ("pispbe", "rpi-hevc", "bcm2835")

class ImageServer:
    def __init__(self, *image_paths, use_cameras=False, camera_ids=None, init_on_start=True):
        """
        Initializes the ImageServer for multiple workspaces and cameras.
        """
        self.use_cameras = use_cameras
        self.image_paths = list(image_paths)
        self.ordered_paths = None
        
        # workspaces[workspace_id][zone_id] = camera_id (/dev/videoX)
        self.workspaces = {
            1: {1: None, 2: None},
            2: {1: None, 2: None}
        }
        
        self.camera_lock = threading.Lock()
        
        # Ініціалізація ArUco зі стандартними маркерами DICT_4X4_50
        detector = aruco_factory(resize_for_speed=False)
        self.extractor = WorkspaceExtractor(detector)
        
        if self.use_cameras and init_on_start:
            self._init_cameras()

    @staticmethod
    def find_capture_indices():
        """Швидке сканування V4L2 capture-портів через sysfs"""
        sysfs_base = "/sys/class/video4linux"
        indices = []
        if not os.path.isdir(sysfs_base):
            return indices
            
        for entry in sorted(os.listdir(sysfs_base)):
            if not entry.startswith("video"):
                continue
            sysfs_path = os.path.join(sysfs_base, entry)
            
            index_file = os.path.join(sysfs_path, "index")
            try:
                with open(index_file) as f:
                    dev_index = int(f.read().strip())
            except (OSError, ValueError):
                continue
                
            if dev_index != 0:
                continue  
                
            name_file = os.path.join(sysfs_path, "name")
            try:
                with open(name_file) as f:
                    dev_name = f.read().strip()
            except OSError:
                continue
                
            if any(dev_name.lower().startswith(p) for p in _ISP_PREFIXES):
                continue
                
            try:
                video_num = int(entry.replace("video", ""))
                indices.append((video_num, dev_name))
            except ValueError:
                continue
                
        return indices

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
                    from configs.config import CAMERA_RESOLUTION
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                    cam.set(cv2.CAP_PROP_FPS, 5) 
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, _ = cam.read()
                    if ret:
                        return cam
                    else:
                        cam.release()
            except Exception:
                if 'cam' in locals() and cam is not None:
                    cam.release()
                continue
        return None

    def _capture_with_temp_resolution(self, cam, warmup=6):
        """Розігрів та захоплення кадру"""
        for _ in range(warmup):
            cam.grab()  
        ret, frame = cam.retrieve() 
        if not ret or frame is None:
            raise RuntimeError("Camera lost connection during retrieve")
        return frame

    def _identify_camera(self, cam_id):
        """Робить тестове фото та визначає робоче місце і зону за маркерами"""
        print(f"[{cam_id}] Ідентифікація камери...")
        cam = self._connect_single_camera(cam_id)
        if not cam:
            return None, None
            
        try:
            image = self._capture_with_temp_resolution(cam, warmup=10)
        except Exception as e:
            print(f"[{cam_id}] Помилка при зйомці для ідентифікації: {e}")
            cam.release()
            return None, None
            
        cam.release() # Відпускаємо шину одразу
        
        markers = self.extractor.aruco_detector.detect_markers(image)
        if not markers:
            print(f"[{cam_id}] Маркерів не знайдено.")
            return None, None
            
        zone_id = None
        workspace_id = 1 # Дефолт
        
        detected_ids = set()
        for m in markers:
            # Depending on detector, id might be an int or a list/array
            val = m.get("id")
            if isinstance(val, (list, tuple)) or type(val).__name__ == 'ndarray':
                detected_ids.add(int(val[0]))
            elif val is not None:
                detected_ids.add(int(val))
                
        # Логіка визначення зони: суворе співпадіння ВСІХ маркерів зони
        for z_id, target_ids in ZONES_DICT_WS1.items():
            if all(tid in detected_ids for tid in target_ids):
                print(f"[{cam_id}] Визначено: Workspace 1, Zone {z_id} (Знайдені ID: {list(detected_ids)})")
                return 1, z_id
                
        for z_id, target_ids in ZONES_DICT_WS2.items():
            if all(tid in detected_ids for tid in target_ids):
                print(f"[{cam_id}] Визначено: Workspace 2, Zone {z_id} (Знайдені ID: {list(detected_ids)})")
                return 2, z_id
                
        print(f"[{cam_id}] Маркери знайдені, але жодна зона не містить всіх необхідних маркерів: {detected_ids}")
        return None, None

    def _init_cameras(self):
        """Ініціалізація та прив'язка камер до робочих місць"""
        print("\n--- Сканування та калібрування камер ---")
        capture_ports = self.find_capture_indices()
        
        if not capture_ports:
            print("ПОМИЛКА: Не знайдено жодної камери (sysfs)!")
            return
            
        print(f"Знайдені порти (sysfs): {[f'/dev/video{idx}' for idx, _ in capture_ports]}")
        
        # Скануємо кожну знайдену камеру по черзі (захист USB-хабу)
        for idx, name in capture_ports:
            ws_id, z_id = self._identify_camera(idx)
            if ws_id and z_id:
                self.workspaces[ws_id][z_id] = idx
                print(f"-> Камера /dev/video{idx} успішно прив'язана до Workspace {ws_id}, Zone {z_id}")
            else:
                print(f"-> Камера /dev/video{idx} не розпізнана (не видно маркерів зони/місця)")
                
        print("--- Калібрування завершено ---\n")
        print(f"Конфігурація камер: {self.workspaces}\n")

    def reinit_missing_cameras(self):
        """Спроба пересканувати лише ті порти, які ще не прив'язані до жодного робочого місця."""
        print("\n--- Спроба переініціалізації невідомих камер ---")
        assigned_indices = []
        for ws in self.workspaces.values():
            for cam_id in ws.values():
                if cam_id is not None:
                    assigned_indices.append(cam_id)
                    
        capture_ports = self.find_capture_indices()
        found_new = False
        
        for idx, name in capture_ports:
            if idx in assigned_indices:
                continue
                
            ws_id, z_id = self._identify_camera(idx)
            if ws_id and z_id:
                if ws_id not in self.workspaces:
                    self.workspaces[ws_id] = {}
                self.workspaces[ws_id][z_id] = idx
                print(f"-> Камера /dev/video{idx} успішно прив'язана до Workspace {ws_id}, Zone {z_id}")
                found_new = True
            else:
                print(f"-> Камера /dev/video{idx} все ще не розпізнана.")
                
        print("--- Переініціалізація завершена ---\n")
        if found_new:
            print(f"Нова конфігурація камер: {self.workspaces}\n")
        return found_new
    
    def take_photos(self, workspace_id=1):
        """
        Робить знімки для конкретного робочого місця.
        Гарантує, що камери відкриваються по черзі під Lock для збереження USB Bandwidth.
        """
        images = []
        
        if not self.use_cameras:
            # Fallback для зображень з диска
            paths_to_use = self.ordered_paths if self.ordered_paths else self.image_paths
            for path in paths_to_use:
                if not os.path.exists(path): continue
                img = cv2.imread(path)
                if img is not None: images.append(img)
            return images
            
        # Робота з реальними камерами під м'ютексом
        with self.camera_lock:
            ws_cameras = self.workspaces.get(workspace_id, {})
            
            # Знімаємо зону 1, потім зону 2
            for z_id in [1, 2]:
                cam_id = ws_cameras.get(z_id)
                if cam_id is None:
                    print(f"[WS{workspace_id}] Попередження: Камера для Зони {z_id} не ініціалізована!")
                    continue
                    
                cam = self._connect_single_camera(cam_id)
                if cam is not None:
                    try:
                        print(f"[WS{workspace_id}-Z{z_id}] Зйомка (video{cam_id})...")
                        img = self._capture_with_temp_resolution(cam, warmup=7)
                        images.append(img)
                    except Exception as e:
                        print(f"[WS{workspace_id}-Z{z_id}] Помилка зйомки: {e}")
                    finally:
                        cam.release()
                else:
                    print(f"[WS{workspace_id}-Z{z_id}] Не вдалося відкрити камеру /dev/video{cam_id}")

        return images
    
    def capture_frame(self, cam_id):
        """Capture a single raw frame + extracted workspace frame for streaming.
        Returns (raw_bgr, ws_bgr_or_None). Both are None if camera unavailable.
        """
        with self.camera_lock:
            cam = self._connect_single_camera(cam_id)
            if cam is None:
                return None, None
            try:
                frame = self._capture_with_temp_resolution(cam, warmup=2)
            except Exception:
                frame = None
            finally:
                cam.release()

        if frame is None:
            return None, None

        try:
            _, ws_frame = self.extractor.extract_workspace(frame)
        except Exception:
            ws_frame = None

        return frame, ws_frame

    def get_assigned_cameras(self):
        """Return sorted list of all camera IDs currently assigned to any workspace."""
        cams = set()
        for ws in self.workspaces.values():
            for cam_id in ws.values():
                if cam_id is not None:
                    cams.add(cam_id)
        return sorted(cams)

    def release(self):
        """Камери тепер не тримаються відкритими, тому release майже порожній."""
        print("ImageServer released (камери вже закриті після зйомки).")

if __name__ == "__main__":
    server = ImageServer(use_cameras=True)
    images = server.take_photos(workspace_id=1)
    print(f"Workspace 1: Зроблено {len(images)} фото.")