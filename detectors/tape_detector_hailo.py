import numpy as np
import cv2
from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                            ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
from detectors.base_detector import BaseDetector
from processors.preprocess import Preprocessor
class YoloLikeBoxes:
    """
    Клас, що повністю імітує result.boxes з Ultralytics.
    Підтримує ітерацію, індексацію та атрибути xyxy, xywh, xywhn, conf, cls.
    """
    def __init__(self, boxes_array, orig_shape):
        # boxes_array: (N, 6) -> [x1, y1, x2, y2, conf, cls]
        self.data = boxes_array 
        self.orig_shape = orig_shape # (Height, Width)
        
        if self.data.ndim == 1 and len(self.data) > 0:
             self.data = self.data[np.newaxis, :]
        elif self.data.ndim == 1 and len(self.data) == 0:
             self.data = np.empty((0, 6), dtype=np.float32)

        if len(self.data) > 0:
            self.xyxy = self.data[:, :4]
            self.conf = self.data[:, 4]
            self.cls = self.data[:, 5]
            self.xywh = self._xyxy2xywh(self.xyxy)
            self.xywhn = self._xywh2xywhn(self.xywh, self.orig_shape)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)
            self.conf = np.empty((0,), dtype=np.float32)
            self.cls = np.empty((0,), dtype=np.float32)
            self.xywh = np.empty((0, 4), dtype=np.float32)
            self.xywhn = np.empty((0, 4), dtype=np.float32)
        

    def _xyxy2xywh(self, x):
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]        # width
        y[:, 3] = x[:, 3] - x[:, 1]        # height
        return y

    def _xywh2xywhn(self, x, shape):
        y = np.copy(x)
        h, w = shape
        if w > 0 and h > 0:
            y[:, 0] /= w
            y[:, 2] /= w
            y[:, 1] /= h
            y[:, 3] /= h
        return y
    
    # --- Методи сумісності ---
    def cpu(self): return self
    def numpy(self): return self.data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Повертає новий об'єкт YoloLikeBoxes для конкретного індексу.
        Це дозволяє писати box.xywhn[0] всередині циклу.
        """
        sub_data = self.data[idx]
        return YoloLikeBoxes(sub_data, self.orig_shape)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"ultralytics.engine.results.Boxes object with shape {self.data.shape}"

    


class YoloLikeResult:
    """Клас, що імітує основний об'єкт Results"""
    def __init__(self, original_img, boxes_array, names_dict, path=""):
        self.orig_img = original_img
        self.orig_shape = original_img.shape[:2]
        self.names = names_dict
        self.path = path
        
        self.boxes = YoloLikeBoxes(boxes_array, self.orig_shape)
        
        # Заглушки
        self.keypoints = None
        self.masks = None
        self.obb = None
        self.probs = None
        self.speed = {'inference': 0.0}

    def __len__(self):
        return len(self.boxes)

    def __repr__(self):
        return (
            f"ultralytics.engine.results.Results object with attributes:\n"
            f"boxes: {self.boxes}\n"
            f"names: {self.names}\n"
            f"orig_shape: {self.orig_shape}\n"
        )


class TapeDetectorHailo(BaseDetector): 
    def __init__(self, shared_device, model_path, labels_map, conf_threshold=0.25):
        self.hef_path = model_path
        self.labels_map = labels_map 
        self.conf_threshold = conf_threshold
        self.preprocess = Preprocessor()
        print(f"[Hailo] Init Detector: {self.hef_path}")
        
        self.target = shared_device
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
            self.model_h, self.model_w = 576, 576

        self._resources_released = False

    def detect(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        batch_data = []
        original_images = [] 

        for img in images:
            original_images.append(img)
            
            resized = cv2.resize(img, (self.model_w, self.model_h), interpolation=cv2.INTER_LINEAR)
            
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            batch_data.append(img_rgb)
        
        batch_numpy = np.array(batch_data, dtype=np.uint8)

        if self._resources_released:
             raise RuntimeError("Hailo detector is already released!")

        input_name = self.input_vstream_info.name
        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_params, self.output_params) as pipeline:
                res = pipeline.infer({input_name: batch_numpy})
        final_results = []
        for i, img_raw_result in enumerate(batch_numpy): 
            orig_img = original_images[i]
            orig_h, orig_w = orig_img.shape[:2]
            
            raw_tensors = {k: v[i] for k, v in res.items()}
            
            parsed_boxes = self._parse_to_yolo_format(raw_tensors, orig_w, orig_h)
            
            result_obj = YoloLikeResult(orig_img, parsed_boxes, self.labels_map)
            final_results.append(result_obj)
            
        return final_results

    def _parse_to_yolo_format(self, raw_tensors, orig_w, orig_h):
        # raw_tensors - це словник {назва_шару: тензор_розміру_(H, W, C)}
        tensors = list(raw_tensors.values())
        
        # Групуємо тензори за їх розміром (H). Для 576x576 це будуть розміри 72, 36 та 18
        grouped = {}
        for t in tensors:
            h = t.shape[0]
            if h not in grouped:
                grouped[h] = []
            grouped[h].append(t)
            
        strides = [8, 16, 32]
        reg_max = 16 # Стандарт для YOLOv8/11
        proj = np.arange(reg_max, dtype=np.float32)
        
        all_boxes, all_scores, all_class_ids = [], [], []
        
        for stride in strides:
            h = self.model_h // stride
            w = self.model_w // stride
            
            # Знаходимо пару тензорів для поточного масштабу
            if h not in grouped or len(grouped[h]) != 2:
                continue
                
            t1, t2 = grouped[h]
            # Тензор з боксами завжди має 64 канали (4 * reg_max)
            if t1.shape[-1] == 64: 
                box_tensor, cls_tensor = t1, t2
            else:
                box_tensor, cls_tensor = t2, t1
                
            # Перетворюємо (H, W, C) у плоский вигляд (H*W, C)
            box_tensor = box_tensor.reshape(-1, 4, reg_max)
            cls_tensor = cls_tensor.reshape(-1, cls_tensor.shape[-1])
            
            # --- 1. Класи (Sigmoid) ---
            cls_scores = 1.0 / (1.0 + np.exp(-cls_tensor)) # Застосовуємо Sigmoid
            max_scores = np.max(cls_scores, axis=-1)
            max_classes = np.argmax(cls_scores, axis=-1)
            
            # Фільтруємо за порогом (confidence)
            mask = max_scores > self.conf_threshold
            if not np.any(mask):
                continue
                
            filtered_scores = max_scores[mask]
            filtered_classes = max_classes[mask]
            filtered_boxes = box_tensor[mask]
            
            # --- 2. Бокси (DFL: Softmax + Dot Product) ---
            box_max = np.max(filtered_boxes, axis=-1, keepdims=True)
            box_exp = np.exp(filtered_boxes - box_max)
            box_softmax = box_exp / np.sum(box_exp, axis=-1, keepdims=True)
            boxes_ltrb = np.sum(box_softmax * proj, axis=-1) # Отримуємо дистанції до країв
            
            # --- 3. Декодування координат ---
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), axis=2).reshape(-1, 2).astype(np.float32) + 0.5
            grid = grid[mask]
            
            x1y1 = grid - boxes_ltrb[:, :2]
            x2y2 = grid + boxes_ltrb[:, 2:]
            boxes_xyxy = np.concatenate((x1y1, x2y2), axis=-1) * stride
            
            all_boxes.append(boxes_xyxy)
            all_scores.append(filtered_scores)
            all_class_ids.append(filtered_classes)
            
        if not all_boxes:
            return np.empty((0, 6), dtype=np.float32)
            
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_class_ids = np.concatenate(all_class_ids, axis=0)
        
        # Масштабуємо координати під оригінальний розмір картинки
        scale_x = orig_w / self.model_w
        scale_y = orig_h / self.model_h
        all_boxes[:, 0] *= scale_x
        all_boxes[:, 1] *= scale_y
        all_boxes[:, 2] *= scale_x
        all_boxes[:, 3] *= scale_y
        
        # --- 4. NMS (Фільтрація боксів, що перекриваються) ---
        boxes_xywh = np.copy(all_boxes)
        boxes_xywh[:, 2] = all_boxes[:, 2] - all_boxes[:, 0] # width
        boxes_xywh[:, 3] = all_boxes[:, 3] - all_boxes[:, 1] # height
        
        # Робимо NMS незалежним для різних класів (Class-aware NMS)
        max_coord = 10000.0
        shifted_boxes = boxes_xywh.copy()
        shifted_boxes[:, 0] += all_class_ids * max_coord
        shifted_boxes[:, 1] += all_class_ids * max_coord

        iou_threshold = 0.45 # Стандартний IOU поріг для YOLO
        indices = cv2.dnn.NMSBoxes(
            shifted_boxes.tolist(), 
            all_scores.tolist(), 
            self.conf_threshold, 
            iou_threshold
        )
        
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)
            
        indices = indices.flatten()
        final_boxes = all_boxes[indices]
        final_scores = all_scores[indices].reshape(-1, 1)
        final_classes = all_class_ids[indices].reshape(-1, 1).astype(np.float32)
        
        return np.concatenate((final_boxes, final_scores, final_classes), axis=-1)





class MultiClassHailoDetector(BaseDetector):
    def __init__(self, model1_path="models/tape_detector.hef", model2_path="models/connectors.hef", conf_threshold=0.25):
        self.target = VDevice()
        self.target.__enter__()
        print("[System] VDevice initialized successfully.")

        self.merged_names = {0: "Connector", 1: "Label", 2: "Tape"}

        self.det1 = TapeDetectorHailo(self.target, model1_path, self.merged_names, conf_threshold)
        self.det2 = TapeDetectorHailo(self.target, model2_path, self.merged_names, conf_threshold)

    def detect(self, image):
        """Коротка обгортка для одного зображення"""
        return self.predict_batch([image])[0]

    def predict_batch(self, images, metadata=None):
        """Розумний роутинг: відправляємо картинку тільки у відповідну модель"""
        
        if metadata is None:
            print("[Warning] No metadata provided, running both models on all images!")
            pass

        results = [None] * len(images)
        
        tape_indices, tape_images = [], []
        conn_indices, conn_images = [], []
        
        for i, (img, meta) in enumerate(zip(images, metadata)):
            if meta["type"] in ["TAPE", "LABEL"]:
                tape_indices.append(i)
                tape_images.append(img)
            elif meta["type"] == "CONNECTORS":
                conn_indices.append(i)
                conn_images.append(img)

        if tape_images:
            res1 = self.det1.predict_batch(tape_images)
            for idx, r1 in zip(tape_indices, res1):
                boxes = np.copy(r1.boxes.data)
                if len(boxes) > 0:
                    boxes[:, 5] += 1  
                results[idx] = YoloLikeResult(r1.orig_img, boxes, self.merged_names)

        if conn_images:
            res2 = self.det2.predict_batch(conn_images)
            for idx, r2 in zip(conn_indices, res2):
                boxes = np.copy(r2.boxes.data)
                results[idx] = YoloLikeResult(r2.orig_img, boxes, self.merged_names)

        return results

    def release(self):
        """Правильне закриття всіх ресурсів Hailo"""
        self.det1.release()
        self.det2.release()
        self.target.__exit__(None, None, None)
        print("[System] MultiClassHailoDetector resources released.")

    def __del__(self):
        self.release()