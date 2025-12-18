import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
from processors.preprocess import Preprocessor
from detectors.base_detector import BaseDetector


import numpy as np

class YoloLikeBoxes:
    """Клас, що імітує result.boxes з Ultralytics"""
    def __init__(self, boxes_array, orig_shape):
        self.data = boxes_array 
        # orig_shape очікується як (Height, Width)
        self.orig_shape = orig_shape
        
        # Виправляємо розмірність для правильної роботи numpy
        if self.data.ndim == 1 and len(self.data) > 0:
             self.data = self.data[np.newaxis, :]

        if len(self.data) > 0:
            self.xyxy = self.data[:, :4] # Координати (x1, y1, x2, y2)
            self.conf = self.data[:, 4]  # Впевненість
            self.cls = self.data[:, 5]   # ID класу
            
            # Абсолютні координати (пікселі)
            self.xywh = self._xyxy2xywh(self.xyxy)
            
            # --- ВИПРАВЛЕННЯ ТУТ ---
            # Нормалізовані координати (0.0 - 1.0)
            self.xywhn = self._xywh2xywhn(self.xywh, self.orig_shape)
        else:
            self.xyxy = np.empty((0, 4))
            self.conf = np.empty((0,))
            self.cls = np.empty((0,))
            self.xywh = np.empty((0, 4))
            self.xywhn = np.empty((0, 4))

    def _xyxy2xywh(self, x):
        # Перетворення x1y1x2y2 -> xywh (center_x, center_y, width, height)
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]        # width
        y[:, 3] = x[:, 3] - x[:, 1]        # height
        return y

    def _xywh2xywhn(self, x, shape):
        # Перетворення пікселів у нормалізовані значення (0-1)
        y = np.copy(x)
        h, w = shape # shape зазвичай (Height, Width)
        
        # Захист від ділення на нуль
        if w > 0 and h > 0:
            y[:, 0] /= w  # x center / width
            y[:, 2] /= w  # width / width
            y[:, 1] /= h  # y center / height
            y[:, 3] /= h  # height / height
        return y
    
    def cpu(self): return self
    def numpy(self): return self.data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sub_data = self.data[idx]
        if isinstance(idx, int):
            sub_data = sub_data[np.newaxis, :]
        return YoloLikeBoxes(sub_data, self.orig_shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"ultralytics.engine.results.Boxes object with shape {self.data.shape}"

class YoloLikeResult:
    """Клас, що імітує основний об'єкт Results"""
    def __init__(self, original_img, boxes_array, names_dict):
        self.orig_img = original_img
        self.orig_shape = original_img.shape[:2]
        self.names = names_dict # Словник {0: 'Tape', 1: 'Label'}
        self.boxes = YoloLikeBoxes(boxes_array, self.orig_shape)
        
    def __len__(self):
        return len(self.boxes)
    def __repr__(self):
        """Робимо вивід ідентичним до Ultralytics"""
        return (
            f"ultralytics.engine.results.Results object with attributes:\n\n"
            f"boxes: {self.boxes}\n"
            f"names: {self.names}\n"
            f"orig_img: array(shape={self.orig_img.shape}, dtype={self.orig_img.dtype})\n"
            f"orig_shape: {self.orig_shape}\n"
            
        )
# --- ОСНОВНИЙ ДЕТЕКТОР HAILO ---

class TapeDetectorHailo(BaseDetector): 
    def __init__(self, model_path="models/tape_detector.hef", conf_threshold=0.25):
        self.hef_path = model_path
        self.labels_map = {1: "Tape", 0: "Label"} 
        self.conf_threshold = conf_threshold
        self.preprocess = Preprocessor()
        print(f"[Hailo] Init Detector: {self.hef_path}")
        
        # Ініціалізація Hailo
        self.target = VDevice()
        self.hef = HEF(self.hef_path)

        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = self.network_groups[0]
        
        self.input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        shape = self.input_vstream_info.shape
        
        if len(shape) == 3:
            self.model_h, self.model_w = shape[0], shape[1]
        elif len(shape) == 4:
            self.model_h, self.model_w = shape[1], shape[2]
        else:
            self.model_h, self.model_w = 640, 640

        # Активація контексту
        self.network_ctx = self.network_group.activate()
        self.network_ctx.__enter__()

        self.pipeline = InferVStreams(self.network_group, self.input_params, self.output_params)
        self.pipeline_ctx = self.pipeline.__enter__()
        self._resources_released = False
    def detect(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        """
        Повертає список об'єктів, сумісних з Ultralytics Results.
        """
        batch_data = []
        original_images = [] 

        for img in images:
            original_images.append(img)
            img = self.preprocess.preprocess(img)
            resized = cv2.resize(img, (self.model_w, self.model_h))
            batch_data.append(resized)
        
        batch_numpy = np.array(batch_data, dtype=np.uint8)

        input_name = self.input_vstream_info.name
        res = self.pipeline.infer({input_name: batch_numpy})
        print(res)
        output_name = list(res.keys())[0]
        raw_output = res[output_name]
        
        # Формування результатів
        final_results = []
        for i, img_raw_result in enumerate(raw_output):
            orig_img = original_images[i]
            orig_h, orig_w = orig_img.shape[:2]
            
            # Парсинг Hailo -> Numpy array [N, 6] (x1, y1, x2, y2, conf, cls)
            parsed_boxes = self._parse_to_yolo_format(img_raw_result, orig_w, orig_h)
            
            # Створення фейкового об'єкта Result
            result_obj = YoloLikeResult(orig_img, parsed_boxes, self.labels_map)
            final_results.append(result_obj)
            
        return final_results

    def _parse_to_yolo_format(self, class_list, orig_w, orig_h):
        """
        Перетворює вихід Hailo у матрицю numpy, яку очікує YOLO-логіка.
        Формат виходу: [[x1, y1, x2, y2, conf, cls_id], ...]
        """
        detections = []

        if class_list is None or len(class_list) == 0:
            return np.empty((0, 6), dtype=np.float32)

        for class_id, class_boxes in enumerate(class_list):
            print(class_id)
            if class_boxes is None or len(class_boxes) == 0:
                continue
            
            for box in class_boxes:
                # Hailo box format: [ymin, xmin, ymax, xmax, score]
                if len(box) < 5: continue
                ymin, xmin, ymax, xmax, score = box
                
                if score < self.conf_threshold:
                    continue
                
                # Конвертація в абсолютні пікселі [x1, y1, x2, y2]
                x1 = xmin * orig_w
                y1 = ymin * orig_h
                x2 = xmax * orig_w
                y2 = ymax * orig_h

                # Додаємо в список: [x1, y1, x2, y2, conf, cls]
                detections.append([x1, y1, x2, y2, score, float(class_id)])

        if not detections:
             return np.empty((0, 6), dtype=np.float32)

        return np.array(detections, dtype=np.float32)
    
    def release(self):
        """Коректне очищення ресурсів"""
        if self._resources_released:
            return
            
        print("[Hailo] Releasing resources...")
        try:
            if hasattr(self, 'pipeline_ctx'):
                self.pipeline_ctx.__exit__(None, None, None)
            if hasattr(self, 'network_ctx'):
                self.network_ctx.__exit__(None, None, None)
        except Exception as e:
            print(f"[Hailo Warning] Error during release: {e}")
        finally:
            self._resources_released = True

    def __del__(self):
        # Автоматичний виклик при видаленні об'єкта
        self.release()
