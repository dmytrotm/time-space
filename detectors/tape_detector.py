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
        # Keep legacy method for compatibility if needed, or redirect to batch
        return self.predict_batch([image])[0]

    def predict_batch(self, images):
        """
        Run inference on a batch of images.
        """
        # Preprocess all images
        preprocessed_images = [self.preprocessor.preprocess(img) for img in images]
        
        # Run batch inference
        # Ultralytics YOLO supports list of images
        results = self.model(preprocessed_images, conf=self.conf_threshold, verbose=False)
        return results
