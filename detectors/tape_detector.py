from ultralytics import YOLO
from processors.preprocess import Preprocessor
from detectors.base_detector import BaseDetector


class TapeDetector(BaseDetector):
    def __init__(self, model_path="models/tape_detector.pt", conf_threshold=0.25):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.preprocessor = Preprocessor()
        self.conf_threshold = conf_threshold

    def detect(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        """
        Run inference on a batch of images.
        """
        preprocessed_images = [self.preprocessor.preprocess(img) for img in images]
        
       
        results = self.model(preprocessed_images, conf=self.conf_threshold, verbose=False)
        return results


import torch
from ultralytics.engine.results import Results

class MultiClassYoloDetector(BaseDetector):
    def __init__(self, model1_path="models/tape_detector.pt", model2_path="models/connectors.pt", conf_threshold=0.25):
        print(f"[PyTorch] Init Multi-Class Detector")
        self.det1 = TapeDetector(model1_path, conf_threshold) 
        self.det2 = TapeDetector(model2_path, conf_threshold) 
        
        self.merged_names = {0: "Connector", 1: "Label", 2: "Tape"}

    def detect(self, image):
        return self.predict_batch([image])[0]

    def predict_batch(self, images, metadata=None):
        """Розумний роутинг для PyTorch YOLO моделей"""
        
        if metadata is None:
            results1 = self.det1.predict_batch(images)
            results2 = self.det2.predict_batch(images)
            
            merged_results = []
            for r1, r2 in zip(results1, results2):
                boxes1 = r1.boxes.data.clone() if len(r1.boxes) > 0 else torch.empty((0, 6), device=r1.boxes.data.device)
                boxes2 = r2.boxes.data.clone() if len(r2.boxes) > 0 else torch.empty((0, 6), device=r2.boxes.data.device)
            
                if len(boxes1) > 0:
                    boxes1[:, 5] += 1  
                merged_boxes = torch.cat((boxes1, boxes2), dim=0)
                
                merged_result = Results(
                    orig_img=r1.orig_img, 
                    path=r1.path, 
                    names=self.merged_names, 
                    boxes=merged_boxes
                )
                merged_results.append(merged_result)
            return merged_results

        merged_results = [None] * len(images)
        
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
                boxes = r1.boxes.data.clone() if len(r1.boxes) > 0 else torch.empty((0, 6), device=r1.boxes.data.device)
                
                if len(boxes) > 0:
                    boxes[:, 5] += 1  
                
                merged_results[idx] = Results(
                    orig_img=r1.orig_img,
                    path=r1.path,
                    names=self.merged_names,
                    boxes=boxes
                )
                
        if conn_images:
            res2 = self.det2.predict_batch(conn_images)
            for idx, r2 in zip(conn_indices, res2):
                boxes = r2.boxes.data.clone() if len(r2.boxes) > 0 else torch.empty((0, 6), device=r2.boxes.data.device)
                
                merged_results[idx] = Results(
                    orig_img=r2.orig_img,
                    path=r2.path,
                    names=self.merged_names,
                    boxes=boxes
                )
                
        return merged_results
    
    def release(self):
        print("[PyTorch] MultiClassYoloDetector released.")

    def __del__(self):
        self.release()