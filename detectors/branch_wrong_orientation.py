import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import cv2

class WrongOrientation:
    def __init__(self,model_path="models/resnet_model.pth"):
        self.model =models.resnet18(weights=None)

        self.model.fc = nn.Sequential(
                        nn.Linear(self.model.fc.in_features, 1),
                        nn.Sigmoid())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    # img in RGB format
    def detect(self,img:cv2.UMat):
        
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor).item()
        return output > 0.5            

if __name__ == "__main__":
    detector = WrongOrientation()
    img1 = cv2.imread("branchdataset/Test_Case_3_4_Z1_14_4_ORIENTATION_001.png")
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img0 = cv2.imread("branchdataset/Test_Case_2_2_Z2_4_2_ORIENTATION_003.png")
    img0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)

    print(f"1 - {detector.detect(img1)}\n0 - {detector.detect(img0)}")
