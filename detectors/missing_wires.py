from PIL import Image
import torch
import os
import torchvision.models as models
import torchvision.transforms
import torch.nn as nn
import cv2
import tempfile
import logging

class WirePredictor:
    def __init__(self, model_path_pth, cropping_coordinates):
        """
        Initializes the WirePredictor with the trained model from a .pth file and vocabulary.

        Args:
            model_path_pth (str): Path to the saved PyTorch model (.pth file).
            vocab (list): List of class names (vocabulary) used during training.
        """
        self.vocab = ['black', 'blue_brown', 'without_wires']
        self.num_classes = len(self.vocab)

        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.model.load_state_dict(torch.load(model_path_pth, map_location=torch.device('cpu')))
        self.model.eval()

        self.cropping_coordinates = cropping_coordinates or {
            'blue_brown': (0.0, 0.582, 0.300, 0.699),      # Left side: blue and brown wires
            'black': (0.244, 0.233, 0.309, 0.466)         # Center: black wire
        }

    def predict_region(self, image_path, region_name):
        """
        Loads an image, crops a specific region, and makes a prediction on that region.

        Args:
            image_path (str): Path to the input image.
            region_name (str): Name of the region to crop (e.g., 'blue_brown', 'black').

        Returns:
            tuple: A tuple containing the predicted class name, class index, and confidence score.
                   Returns (None, None, None) if the region name is invalid or image processing fails.
        """
        if region_name not in self.cropping_coordinates:
            print(f"Error: Invalid region name '{region_name}'. Available regions: {list(self.cropping_coordinates.keys())}")
            return None, None, None

        try:
            img = Image.open(image_path).convert('RGB')
            img_width, img_height = img.size
            x, y, w, h = self.cropping_coordinates[region_name]

            left = int(x * img_width)
            top = int(y * img_height)
            right = int(w * img_width)
            bottom = int(h * img_height)

            left = max(0, left)
            top = max(0, top)
            right = min(img_width, right)
            bottom = min(img_height, bottom)

            cropped_img = img.crop((left, top, right, bottom))


            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(cropped_img)
            input_batch = input_tensor.unsqueeze(0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            input_batch = input_batch.to(device)

            with torch.no_grad():
                output = self.model(input_batch)

            probs = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class_index = torch.max(probs, 0)

            predicted_class = self.vocab[predicted_class_index.item()]
            return predicted_class, predicted_class_index.item(), confidence.item()

        except Exception as e:
            print(f"Error processing image {image_path} for region '{region_name}': {e}")
            return None, None, None

    def detect_wire_exits(self, image_path):
        """
        Detects if the 'blue_brown' and 'black' wire exits are present based on model predictions.

        Args:
            image_path (str): Path to the input image.

        Returns:
            bool: True if both 'blue_brown' and 'black' wire exits are predicted correctly, False otherwise.
        """
        blue_brown_pred, _, _ = self.predict_region(image_path, 'blue_brown')
        black_pred, _, _ = self.predict_region(image_path, 'black')


        if blue_brown_pred is not None and black_pred is not None:
            return blue_brown_pred == 'blue_brown' and black_pred == 'black'
        else:
            return False


class MissingWiresDetector:
    """
    Detector for missing wires using the WirePredictor model.
    Detects presence of blue, brown, and black wires in images.
    Works with OpenCV images (numpy arrays).
    """
    
    def __init__(self, model_path: str = None, cropping_coordinates: dict = None):
        """
        Initialize the MissingWiresDetector.
        
        Args:
            model_path: Path to the wire classification model (.pth file).
                       If None, uses default path 'models/wire_classification_model.pth'
        """
        self.logger = logging.getLogger(__name__)
        
        if model_path is None:
            # Get path relative to this file
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'models', 'wire_classification_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.predictor = WirePredictor(model_path, cropping_coordinates)
        self.logger.info(f"Loaded wire classification model from {model_path}")
    
    def set_rois_from_config(self, roi_config: dict):
        """
        Set ROIs from configuration dictionary.
        
        Args:
            roi_config: Dictionary containing 'wires' array with wire ROI definitions.
                       Expected format:
                       {
                           'wires': [
                               {'id': 1, 'start': {'x': ..., 'y': ...}, 'end': {'x': ..., 'y': ...}},
                               {'id': 2, 'start': {'x': ..., 'y': ...}, 'end': {'x': ..., 'y': ...}}
                           ]
                       }
        """
        if 'wires' not in roi_config or len(roi_config['wires']) < 2:
            raise ValueError("ROI config must contain at least 2 wire definitions")
        
        # Wire 1: blue_brown (left side)
        wire1 = roi_config['wires'][0]
        # Wire 2: black (center)
        wire2 = roi_config['wires'][1]
        
        self.predictor.cropping_coordinates = {
            'blue_brown': (
                wire1['start']['x'],
                wire1['start']['y'],
                wire1['end']['x'],
                wire1['end']['y']
            ),
            'black': (
                wire2['start']['x'],
                wire2['start']['y'],
                wire2['end']['x'],
                wire2['end']['y']
            )
        }
        self.logger.info("Updated wire ROIs from config")
    
    def _save_temp_image(self, img):
        """
        Save a cv2 image to a temporary file.
        
        Args:
            img: cv2 image (numpy array)
            
        Returns:
            str: Path to the temporary file
        """
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img)
        return tmp_path
    
    def detect_across_rois(self, img):
        """
        Detect wires in an image.
        
        Args:
            img: cv2 image (numpy array)
            
        Returns:
            dict: Dictionary with wire detection results
                 {'blue': bool, 'brown': bool, 'black': bool}
        """
        tmp_path = None
        try:
            # Save image to temporary file for WirePredictor
            tmp_path = self._save_temp_image(img)
            
            # Predict blue_brown region
            blue_brown_pred, _, blue_brown_conf = self.predictor.predict_region(tmp_path, 'blue_brown')
            
            # Predict black region
            black_pred, _, black_conf = self.predictor.predict_region(tmp_path, 'black')
            
            # Determine if wires are present
            # blue_brown_pred returns 'blue_brown' if both blue and brown wires present
            blue_brown_present = (blue_brown_pred == 'blue_brown')
            black_present = (black_pred == 'black')
            
            return {
                'blue': blue_brown_present,
                'brown': blue_brown_present,
                'black': black_present
            }
        
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    self.logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")
    
    def get_missing_wires(self, img, use_rois=True):
        """
        Get list of missing wires in an image.
        
        Args:
            img: cv2 image (numpy array)
            use_rois: Whether to use ROIs (kept for API compatibility)
            
        Returns:
            list: List of missing wire names (e.g., ['blue', 'brown'] or ['black'])
        """
        detected = self.detect_across_rois(img)
        
        missing = []
        if not detected['blue']:
            missing.append('blue')
        if not detected['brown']:
            missing.append('brown')
        if not detected['black']:
            missing.append('black')
        
        return missing
    
    def is_all_wires_present(self, img):
        """
        Check if all wires are present in the image.
        
        Args:
            img: cv2 image (numpy array)
            
        Returns:
            bool: True if all wires are present, False otherwise
        """
        missing = self.get_missing_wires(img)
        return len(missing) == 0