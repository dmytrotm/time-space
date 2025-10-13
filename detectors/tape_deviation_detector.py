import json
from detectors.tape_detector import TapeDetector

class TapeDeviationDetector:
    def __init__(self, positions_json):
        self.positions_json = positions_json
        self.tape_detector = TapeDetector(conf_threshold=0.8)
    
    def detect_tape_and_find_deviation(self, image, roi_name):
        results = self.tape_detector.detect(image)
        deviation_results = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.cls.tolist() == [1.0]:
                    x, y, w, h = box.xywh[0].tolist()
                    index = int(roi_name.split('_')[-1])
                    deviation = self.is_tape_correct(index, x, w)
                    deviation_results.append(deviation)
        return deviation_results

    def is_tape_correct(self, index, new_x, new_w) -> int:
        '''
            -1 - too far
            1 - too long or too short
            0 - correct
        '''

        x = self.positions_json[(index-1)*2]["Mean"]  
        dx = 0.3
        w = self.positions_json[(index - 1) * 2 + 1]["Mean"]
        dw = self.positions_json[(index - 1) * 2 + 1]["IQR"] * 2


        if abs(x - new_x) > dx:
            return -1
        elif abs(w - new_w) > dw:
            return 1
        return 0