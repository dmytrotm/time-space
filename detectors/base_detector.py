from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def predict_batch(self, images):
        """
        Run inference on a batch of images.
        
        Args:
            images (list): List of numpy arrays (images).
            
        Returns:
            list: List of results corresponding to the input images.
        """
        pass
