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

        self.network_ctx = self.network_group.activate()
        self.network_ctx.__enter__()

        self.pipeline = InferVStreams(self.network_group, self.input_params, self.output_params)
        self.pipeline_ctx = self.pipeline.__enter__()
        self._resources_released = False

    def detect(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        batch_data = []
        original_images = [] 

        for img in images:
            original_images.append(img)
            img = self.preprocess.preprocess(img)
            resized = cv2.resize(img, (self.model_w, self.model_h))
            batch_data.append(resized)
        
        batch_numpy = np.array(batch_data, dtype=np.uint8)

        if self._resources_released:
             raise RuntimeError("Hailo detector is already released!")

        input_name = self.input_vstream_info.name
        res = self.pipeline.infer({input_name: batch_numpy})
        
        output_name = list(res.keys())[0]
        raw_output = res[output_name]
        
        final_results = []
        for i, img_raw_result in enumerate(raw_output):
            orig_img = original_images[i]
            orig_h, orig_w = orig_img.shape[:2]
            
            parsed_boxes = self._parse_to_yolo_format(img_raw_result, orig_w, orig_h)
            
            result_obj = YoloLikeResult(orig_img, parsed_boxes, self.labels_map)
            final_results.append(result_obj)
        return final_results

    def _parse_to_yolo_format(self, class_list, orig_w, orig_h):
        detections = []

        if class_list is None or len(class_list) == 0:
            return np.empty((0, 6), dtype=np.float32)

        for class_id, class_boxes in enumerate(class_list):
            if class_boxes is None or len(class_boxes) == 0:
                continue
            
            for box in class_boxes:
                if len(box) < 5: continue
                ymin, xmin, ymax, xmax, score = box
                
                if score < self.conf_threshold:
                    print(score,class_id)
                    continue
                
                x1 = xmin * orig_w
                y1 = ymin * orig_h
                x2 = xmax * orig_w
                y2 = ymax * orig_h
                
                detections.append([x1, y1, x2, y2, score, int(class_id)])

        if not detections:
             return np.empty((0, 6), dtype=np.float32)

        return np.array(detections, dtype=np.float32)
    
    def release(self):
        if self._resources_released:
            return
        print("[Hailo] Releasing resources...")
        try:
            if hasattr(self, 'pipeline_ctx'): self.pipeline_ctx.__exit__(None, None, None)
            if hasattr(self, 'network_ctx'): self.network_ctx.__exit__(None, None, None)
        except Exception as e:
            print(f"[Hailo Warning] Error during release: {e}")
        finally:
            self._resources_released = True

    def __del__(self):
        self.release()





class MultiClassHailoDetector(BaseDetector):
    def __init__(self, model1_path, model2_path, conf_threshold=0.25):
        self.target = VDevice()
        self.target.__enter__()
        print("[System] VDevice initialized successfully.")

        self.merged_names = {0: "Connector", 1: "Label", 2: "Tape"}

        self.det1 = TapeDetectorHailo(self.target, model1_path, self.merged_names, conf_threshold)
        self.det2 = TapeDetectorHailo(self.target, model2_path, self.merged_names, conf_threshold)

    def detect(self, image):
        """Коротка обгортка для одного зображення"""
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        """Основна логіка інференсу та злиття результатів"""
        results1 = self.det1.predict_batch(images)
        results2 = self.det2.predict_batch(images)
        
        merged_results = []
        for r1, r2 in zip(results1, results2):
            boxes1 = np.copy(r1.boxes.data)
            boxes2 = np.copy(r2.boxes.data)
            
            if len(boxes1) > 0:
                boxes1[:, 5] += 1
                
            merged_boxes = np.vstack((boxes1, boxes2))
            
            merged_results.append(YoloLikeResult(r1.orig_img, merged_boxes, self.merged_names))
            
        return merged_results

    def release(self):
        """Правильне закриття всіх ресурсів Hailo"""
        self.det1.release()
        self.det2.release()
        self.target.__exit__(None, None, None)
        print("[System] MultiClassHailoDetector resources released.")

    def __del__(self):
        self.release()