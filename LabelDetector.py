import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob


class LabelDetector:
    """
    A class to detect the presence of labels in images based on black cluster analysis.
    
    Detection Logic:
    - If fewer than 2 significant black clusters are found: NO LABEL
    - If 2 or more significant black clusters are found: LABEL PRESENT
    """
    
    def __init__(self, min_area_percentage=1.0, threshold_value=150, verbose=False):
        """
        Initialize the LabelDetector.
        
        Args:
            min_area_percentage (float): Minimum area percentage for significant clusters (default: 1.0%)
            threshold_value (int): Threshold value for binary thresholding (default: 150)
            verbose (bool): Whether to show processing steps and visualization (default: False)
        """
        self.min_area_percentage = min_area_percentage
        self.threshold_value = threshold_value
        self.verbose = verbose
        
    def __call__(self, image):
        """
        Detect whether a label is present in the given image.
        
        Args:
            image: Can be either:
                - numpy array (loaded image)
                - string (path to image file)
            
        Returns:
            bool: True if label detected, False otherwise
        """
        # Handle different input types
        if isinstance(image, str):
            # It's a file path
            img = cv.imread(image)
            image_name = os.path.basename(image)
            if img is None:
                if self.verbose:
                    print(f"ERROR: Could not read image {image}")
                return False
        else:
            # It's already a numpy array
            img = image
            image_name = "input_image"
        
        try:
            # Process the image
            num_clusters = self._process_image(img, image_name)
            
            # Make detection decision
            has_label = num_clusters >= 2
            
            if self.verbose:
                print(f"Result: {'LABEL DETECTED' if has_label else 'NO LABEL'} ({num_clusters} clusters)")
            
            return has_label
            
        except Exception as e:
            if self.verbose:
                print(f"ERROR processing image: {str(e)}")
            return False
    
    def _process_image(self, img, image_name):
        """Internal method to process the image and find black clusters."""
        
        if self.verbose:
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.title(f'Step 1: Original\n{image_name}')
            plt.axis('off')

        # Convert to LAB and extract A-channel
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        a_channel = lab[:,:,1]
        
        # Apply CLAHE for contrast enhancement
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_a_channel = clahe.apply(a_channel.astype(np.uint8))
        
        if self.verbose:
            plt.subplot(1, 4, 2)
            plt.imshow(enhanced_a_channel, cmap='gray')
            plt.title('Step 2: Enhanced A-channel')
            plt.axis('off')

        # Thresholding and morphological operations
        ret, thresh = cv.threshold(enhanced_a_channel, self.threshold_value, 255, cv.THRESH_BINARY_INV)
        
        # Apply erosion
        kernel = np.ones((4,4), np.uint8)
        eroded_mask = cv.erode(thresh, kernel, iterations=3)
        
        # Remove small black dots
        inverted_eroded = cv.bitwise_not(eroded_mask)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_eroded, connectivity=8)

        total_pixels = thresh.size
        min_area = total_pixels * (self.min_area_percentage / 1000.0)
        
        # Create mask of small black components to remove
        small_black_mask = np.zeros_like(eroded_mask)
        for label in range(1, num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if area < min_area:
                # small_black_mask[labels == label] = 255
                pass
        
        # Final mask
        final_mask = eroded_mask.copy()
        final_mask[small_black_mask == 255] = 255

        if self.verbose:
            plt.subplot(1, 4, 3)
            plt.imshow(final_mask, cmap='gray')
            plt.title('Step 3: Cleaned Mask')
            plt.axis('off')

        # Analyze individual black clusters
        num_black_labels, black_labels, black_stats, black_centroids = cv.connectedComponentsWithStats(
            cv.bitwise_not(final_mask), connectivity=8)
        
        significant_clusters = 0
        
        if self.verbose:
            # Create cluster visualization
            cluster_visualization = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            cluster_visualization[:,:] = [255, 255, 255]  # White background
            
            # Generate colors for clusters
            np.random.seed(42)
            colors = []
            for i in range(num_black_labels):
                colors.append(np.random.randint(0, 255, 3))
            colors[0] = [255, 255, 255]  # Background stays white
        
        for label in range(1, num_black_labels):
            area = black_stats[label, cv.CC_STAT_AREA]
            
            if area >= min_area:
                significant_clusters += 1
                if self.verbose:
                    # Color this cluster in visualization
                    cluster_visualization[black_labels == label] = colors[label % len(colors)]

        if self.verbose:
            plt.subplot(1, 4, 4)
            plt.imshow(cluster_visualization)
            plt.title(f'Step 4: {significant_clusters} Clusters')
            plt.axis('off')
            
            # Add cluster numbers
            cluster_count = 0
            for label in range(1, num_black_labels):
                area = black_stats[label, cv.CC_STAT_AREA]
                if area >= min_area:
                    cluster_count += 1
                    cx, cy = black_centroids[label]
                    plt.annotate(f"{cluster_count}", (cx, cy), 
                                color='black', fontsize=8, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            plt.show()
        
        return significant_clusters
    
    def detect_batch(self, input_directory, image_extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']):
        """
        Process multiple images in a directory.
        
        Args:
            input_directory (str): Directory containing images
            image_extensions (list): List of image file extensions to process
            
        Returns:
            list: List of detection results for each image
        """
        # Gather all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_directory, ext)))
            image_files.extend(glob.glob(os.path.join(input_directory, ext.upper())))
        
        if not image_files:
            print(f"No image files found in directory: {input_directory}")
            return []
        
        image_files.sort()
        if self.verbose:
            print(f"Found {len(image_files)} image files to process")
        
        results = []
        labels_detected = 0
        
        for image_path in image_files:
            result = self(image_path)  # Use __call__ method
            results.append({
                'image_path': image_path,
                'has_label': result
            })
            
            if result:
                labels_detected += 1
        
        # Summary
        print(f"\nBatch Processing Complete:")
        print(f"Images processed: {len(image_files)}")
        print(f"Labels detected: {labels_detected}")
        print(f"No labels: {len(image_files) - labels_detected}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Create detector
    detector = LabelDetector(verbose=True)
    
    # Simple usage - just feed it an image and get True/False
    # has_label = detector(cv.imread("rois/Missing label/IMG_1691.jpg"))  # numpy array
    has_label = detector("rois/Missing label/IMG_1698.jpg")     # file path
    print(has_label)  # True or False