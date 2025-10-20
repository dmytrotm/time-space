import unittest
import pandas
from detectors.tape_deviation_detector import TapeDeviationDetector
from ultralytics import YOLO
import os
dataframe_path = "UnitTests/labels_parsed.csv"
dataset_path = "yolo11dataset_grouped"
model_path = "models/tape_detector.pt"
positions_json = "configs/positions.json"

class TestTapeDeviationDetector(unittest.TestCase):
    def test_detector(self):
        df = pandas.read_csv(dataframe_path)
        df["GROI"] = (df["Зона"]-2)*-5+df['ROI'] - 2*(df['class']-1)
        model = YOLO(model_path)
        count = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                
                if file.endswith((".jpg", ".png", ".jpeg")):
                    txt_filename = os.path.splitext(file)[0] + ".txt"
            
                    if txt_filename in df["Назва файлу"].values:
                        current = df[df["Назва файлу"]==txt_filename]
                        if(current.empty):
                            continue
                        if(current["GROI"].values[0] == 8):
                            continue
                        groi = current["GROI"].values[0]
                        test_case = current["Test_Case"].values[0]
                        with self.subTest(file = file,groi = groi, test_case = test_case):
                            img_path = os.path.join(root, file)
                            results = model(img_path)
                            boxes = results[0].boxes
                            if len(boxes)==0:
                                self.skipTest()
                            idx = boxes.conf.argmax()
                                
                            x, y, w, h = boxes.xywhn[idx].cpu().numpy()

                            tdd = TapeDeviationDetector(positions_json)
                            result = tdd.is_tape_correct(groi-1,x,w)
                            count+=1
                            if test_case == 6 and groi == 6:
                                self.assertEqual(result,-1)
                            elif test_case == 5 and groi == 7:
                                self.assertEqual(result,1)
                            else:
                                self.assertEqual(result,0)
        print(count)



if __name__ == "__main__":
    unittest.main()
    
