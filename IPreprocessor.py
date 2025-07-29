import cv2
import numpy as np

class IPreprocessor:

    def preprocess(self, image:cv2.Mat | np.ndarray)->cv2.Mat | np.ndarray:
        """Може бути як скейлінг, препроцесор або декоратор що об'єднує ці процеси"""
        pass
    def set_params(self, params:dict)->None:
        """Передати параметри для препроцесору"""
        pass