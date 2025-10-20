import unittest
import pandas
from TapeDeviationDetector import TapeDeviationDetector, tape_deviation_detector_factory
from ultralytics import YOLO
import os
dataframe_path = "labels_parsed.csv"
dataset_path = "yolo11dataset_grouped"
model_path = "Models/best.pt"

class TestTapeDeviationDetector(unittest.TestCase):
    def test_detector(self):
        df = pandas.read_csv(dataframe_path)
        df["GROI"] = (df["Зона"]-2)*-5+df['ROI'] - 2*(df['class']-1)
        model = YOLO(model_path)
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

                            tdd = tape_deviation_detector_factory(groi-1)
                            result = tdd.is_tape_correct(x,w)
                            if test_case == 6 and groi == 6:
                                self.assertEqual(result,-1)
                            elif test_case == 5 and groi == 7:
                                self.assertEqual(result,1)
                            else:
                                self.assertEqual(result,0)



if __name__ == "__main__":
    unittest.main()
    # model = YOLO(model_path)
    # result = model("yolo11dataset_grouped/valid/images/Test_Case_4_Z2_E-1_3_ROI_001_png.rf.d1aa9fd804a511c916c858a252baa259.jpg")
    # print(result[0].boxes)
    # print(len(result[0].boxes))
    # x, y, w, h = result[0].boxes.xywhn[0].cpu().numpy()
    # print(x, y, w, h)
