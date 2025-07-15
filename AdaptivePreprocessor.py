import cv2
import numpy as np

from IPreprocessor import IPreprocessor


class AdaptivePreprocessor(IPreprocessor):
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
 
         # Analyze image characteristics
         mean_intensity = np.mean(gray)
         std_intensity = np.std(gray)
 
         # Apply CLAHE only if image has low contrast
         if std_intensity < 50:  # Low contrast threshold
             clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
             gray = clahe.apply(gray)
 
         # Apply bilateral filter only for noisy images (high std deviation)
         if std_intensity > 80:  # High noise threshold
             gray = cv2.bilateralFilter(gray, 5, 50, 50)  # Reduced parameters
 
         return gray

    def set_params(self, params:dict) ->None:
        self.params = params