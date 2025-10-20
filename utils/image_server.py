import cv2
import os

# Needs to be implemented, currenctly we just serve the photos from dataset

class ImageServer:
    def __init__(self, *image_paths):
        """
        Initializes with a list of image paths.
        Args:
            *image_paths: A variable number of paths to the images.
        """
        self.image_paths = image_paths

    def take_photos(self):
        """
        Loads images from the stored paths.
        Returns:
            A list of images (as numpy arrays).
        """
        images = []
        for path in self.image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image path not found at {path}")
                continue
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read image at {path}")
        return images
