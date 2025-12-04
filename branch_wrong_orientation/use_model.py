import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2

class ResNet18:
    def __init__(self, path_to_weights="models/resnet_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1),
            nn.Sigmoid()
        )
        self.model = self.model.to(self.device)

        state = torch.load(path_to_weights, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.eval()


    def preprocess_single(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb)
        return tensor

    def preprocess(self, frames):
        if isinstance(frames, (list, tuple)):
            tensors = [self.preprocess_single(f) for f in frames]
            batch = torch.stack(tensors)
        else:
            batch = self.preprocess_single(frames).unsqueeze(0)  
        return batch.to(self.device) 

    @torch.no_grad()
    def predict(self, frames):
        batch = self.preprocess(frames)
        probs = self.model(batch).squeeze(-1)  

        preds = (probs > 0.5).long()  
        preds = preds.cpu().tolist()
        confs = probs.cpu().tolist()

        if isinstance(frames, (list, tuple)):
            return list(zip(preds, confs))
        else:
            return preds[0], confs[0]

if __name__ == "__main__":
    img1 = cv2.imread("branchdataset/Test_Case_2_Z2_11_5_ORIENTATION_007.png")
    model = ResNet18()
    cls, conf = model.predict(img1)
    print("Single image:", cls, conf)