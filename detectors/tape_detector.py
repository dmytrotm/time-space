from ultralytics import YOLO
from utils.preprocess import Preprocessor

class TapeDetector:
    def __init__(self, model_path="models/tape_detector.pt", conf_threshold=0.25):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.preprocessor = Preprocessor()
        self.conf_threshold = conf_threshold
    
    def detect(self, image):
        preprocessed = self.preprocessor.preprocess(image)
        results = self.model(preprocessed, conf=self.conf_threshold, verbose=False)
        return results
