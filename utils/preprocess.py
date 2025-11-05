import cv2
import numpy as np


class Preprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess(self, image):
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_clahe = self.clahe.apply(l)

        lab_clahe = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return result
