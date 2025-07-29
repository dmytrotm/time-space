import cv2
import numpy as np

from IPreprocessor import IPreprocessor


class EnhancedPreprocessor(IPreprocessor):
    def preprocess(self, image:cv2.Mat | np.ndarray)->cv2.Mat | np.ndarray:

        border_size = 1
        image = cv2.copyMakeBorder(
            image,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Light denoising without losing edge information
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Gentle contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
        gray = clahe.apply(gray)

        # Sharpen slightly to enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel * 0.1 + np.eye(3) * 0.9)

        return gray

    def set_params(self, params:dict) ->None:
        self.params = params