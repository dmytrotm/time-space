import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import cv2


class TwistedWires:
    
    def __init__(self, model_path="models/twisted_wires.pth"):
        self.model = models.resnet18(weights=None)
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, 2)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform pipeline (без CLAHE згідно з найкращими параметрами)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def detect(self, img):
        """
        Детекція перекручених проводів
        
        Args:
            img: numpy array у форматі RGB (використовуйте cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                 або cv2.UMat
        
        Returns:
            bool: True якщо проводи перекручені (twisted), False якщо ні (not_twisted)
        """
        # Підготовка зображення
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return predicted_class == 1


if __name__ == "__main__":
    # Приклад використання
    detector = TwistedWires(model_path="models/twisted_wires.pth")
    
    # Приклад 1: Перекручені проводи
    img_twisted = cv2.imread("dataset_split/train/twisted/Frame-1762165564610_(3000, 4000, 3)_TWISTED-WIRES_001.png")
    if img_twisted is not None:
        img_twisted = cv2.cvtColor(img_twisted, cv2.COLOR_BGR2RGB)
        result = detector.detect(img_twisted)
        print(f"Image 1 - Twisted wires: {result}")