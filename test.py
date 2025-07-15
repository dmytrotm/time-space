import cv2
import numpy as np
from glob import glob
import os
from IPreprocessor import IPreprocessor
from custom_markers_finder import MarkerDetector
from WorkspaceExtractor import WorkspaceExtractor  # Import your workspace extractor
import time
from EnhancedPreprocessor import EnhancedPreprocessor
from AdaptivePreprocessor import AdaptivePreprocessor


def test_single_method_with_workspace(dir_name, preprocessor: IPreprocessor|None = None, 
                                     output_dir="output", method_name="baseline",
                                     extract_workspace=True, workspace_size=(800, 600)):
    """
    Test with a single preprocessing method and optional workspace extraction
    """
    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    m = MarkerDetector()
    m.load_dictionaries("custom_dictionaries.yml", dict_names)
    
    # Initialize workspace extractor
    extractor = WorkspaceExtractor(m) if extract_workspace else None
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    failed_dir = f"failed_{method_name}_discovery"
    os.makedirs(failed_dir, exist_ok=True)
    
    if extract_workspace:
        workspace_dir = f"workspace_{method_name}"
        os.makedirs(workspace_dir, exist_ok=True)
        workspace_viz_dir = f"workspace_viz_{method_name}"
        os.makedirs(workspace_viz_dir, exist_ok=True)
    
    image_files = glob(os.path.join(dir_name, "*.jpg"))
    failed_images = {}
    workspace_extracted = {}
    total_time = 0.0
    total_preprocess_time = 0.0
    total_workspace_time = 0.0
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            print(f"Failed to load: {image_file}")
            continue
        
        base_name = os.path.basename(image_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Preprocessing
        start_preprocess_time = time.time()
        if preprocessor:
            processed_image = preprocessor.preprocess(image)
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elapsed_preprocess = time.time() - start_preprocess_time
        total_preprocess_time += elapsed_preprocess
        
        # Marker detection
        start_time = time.time()
        corners, rejected, ids, *_ = m.detect_all_markers(processed_image)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        count = len(corners) if corners else 0
        
        # Draw detected markers
        if corners:
            output = image.copy()
            if len(output.shape) == 2:  # Grayscale
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            cv2.aruco.drawDetectedMarkers(output, corners, ids)
            cv2.imwrite(os.path.join(output_dir, base_name), output)
            
            # Workspace extraction
            if extract_workspace and extractor:
                start_workspace_time = time.time()
                
                # Try automatic rectangle detection
                workspace = extractor.extract_workspace_from_rectangle(
                    image, corners, ids, workspace_size
                )
                
                elapsed_workspace = time.time() - start_workspace_time
                total_workspace_time += elapsed_workspace
                
                if workspace is not None:
                    # Save extracted workspace
                    workspace_path = os.path.join(workspace_dir, f"{name_without_ext}_workspace.jpg")
                    cv2.imwrite(workspace_path, workspace)
                    workspace_extracted[image_file] = True
                    
                    # Save visualization of rectangle detection
                    viz = extractor.visualize_rectangle_detection(image, corners, ids)
                    viz_path = os.path.join(workspace_viz_dir, f"{name_without_ext}_viz.jpg")
                    cv2.imwrite(viz_path, viz)
                else:
                    workspace_extracted[image_file] = False
                    print(f"Failed to extract workspace from: {base_name}")
            
            # Check if expected number of markers found
            if count != 6:
                failed_images[image_file] = count
                cv2.imwrite(os.path.join(failed_dir, base_name), output)
        else:
            failed_images[image_file] = 0
            # Save original image to failed directory for reference
            cv2.imwrite(os.path.join(failed_dir, base_name), image)
            workspace_extracted[image_file] = False
    
    # Calculate averages
    avg_time = total_time / len(image_files) if image_files else 0
    avg_preprocess_time = total_preprocess_time / len(image_files) if image_files else 0
    avg_workspace_time = total_workspace_time / len(image_files) if image_files else 0
    
    # Print results
    print(f'Marker Detection: {len(image_files) - len(failed_images)}/{len(image_files)} successful')
    print(f"Average detection time: {avg_time:.4f} seconds")
    print(f"Average preprocess time: {avg_preprocess_time:.4f} seconds")
    
    if extract_workspace:
        successful_workspaces = sum(1 for success in workspace_extracted.values() if success)
        print(f'Workspace Extraction: {successful_workspaces}/{len(image_files)} successful')
        print(f"Average workspace extraction time: {avg_workspace_time:.4f} seconds")
        print(f"Workspace images saved to: {workspace_dir}")
        print(f"Workspace visualizations saved to: {workspace_viz_dir}")
    
    if len(failed_images) > 0:
        print("Failed marker detection:")
        for name, count in failed_images.items():
            print(f"  {os.path.basename(name)}: {count}/6 markers")
        print(f"Failed images saved to: {failed_dir}")
    
    return {
        'successful_detections': len(image_files) - len(failed_images),
        'total_images': len(image_files),
        'failed_images': failed_images,
        'workspace_extracted': workspace_extracted,
        'avg_detection_time': avg_time,
        'avg_preprocess_time': avg_preprocess_time,
        'avg_workspace_time': avg_workspace_time
    }


def test_workspace_extraction_only(dir_name, output_dir="workspace_only"):
    """
    Test workspace extraction on images that already have detected markers
    """
    dict_names = ["cust_dictionary4", "cust_dictionary5", "cust_dictionary6", "cust_dictionary8"]
    m = MarkerDetector()
    m.load_dictionaries("custom_dictionaries.yml", dict_names)
    extractor = WorkspaceExtractor(m)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    workspace_dir = os.path.join(output_dir, "workspaces")
    os.makedirs(workspace_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    image_files = glob(os.path.join(dir_name, "*.jpg"))
    results = {}
    
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is None:
            continue
            
        base_name = os.path.basename(image_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Detect markers
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, rejected, ids, *_ = m.detect_all_markers(gray)
        
        if corners and len(corners) >= 4:
            # Try different workspace extraction methods
            
            # Method 1: Automatic rectangle detection
            workspace_rect = extractor.extract_workspace_from_rectangle(image, corners, ids, (800, 600))
            
            # Method 2: Auto crop with margin
            workspace_auto = extractor.extract_workspace_auto(image, corners, ids, margin=50)
            
            # Save results
            if workspace_rect is not None:
                cv2.imwrite(os.path.join(workspace_dir, f"{name_without_ext}_rectangle.jpg"), workspace_rect)
                results[f"{name_without_ext}_rectangle"] = True
            
            if workspace_auto is not None:
                cv2.imwrite(os.path.join(workspace_dir, f"{name_without_ext}_auto.jpg"), workspace_auto)
                results[f"{name_without_ext}_auto"] = True
            
            # Save visualization
            viz = extractor.visualize_rectangle_detection(image, corners, ids)
            cv2.imwrite(os.path.join(viz_dir, f"{name_without_ext}_viz.jpg"), viz)
            
            print(f"Processed: {base_name}")
            print(f"  Markers found: {len(corners)}")
            print(f"  Rectangle method: {'✓' if workspace_rect is not None else '✗'}")
            print(f"  Auto method: {'✓' if workspace_auto is not None else '✗'}")
        else:
            print(f"Skipped {base_name}: insufficient markers ({len(corners) if corners else 0})")
            results[name_without_ext] = False
    
    return results


def compare_methods_with_workspace():
    """
    Compare all preprocessing methods with workspace extraction
    """
    methods = [
        ("baseline", None),
        ("adaptive", AdaptivePreprocessor()),
        ("enhanced", EnhancedPreprocessor())
    ]
    
    results = {}
    
    print("=" * 60)
    print("TESTING ALL METHODS WITH WORKSPACE EXTRACTION")
    print("=" * 60)
    
    for method_name, preprocessor in methods:
        print(f"\n{method_name.upper()} METHOD:")
        print("-" * 40)
        
        result = test_single_method_with_workspace(
            "jpg_discovery", 
            preprocessor, 
            output_dir=f"output_{method_name}",
            method_name=method_name,
            extract_workspace=True,
            workspace_size=(800, 600)
        )
        
        results[method_name] = result
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    for method_name, result in results.items():
        successful_workspaces = sum(1 for success in result['workspace_extracted'].values() if success)
        print(f"{method_name.upper()}:")
        print(f"  Marker Detection: {result['successful_detections']}/{result['total_images']}")
        print(f"  Workspace Extraction: {successful_workspaces}/{result['total_images']}")
        print(f"  Avg Detection Time: {result['avg_detection_time']:.4f}s")
        print(f"  Avg Workspace Time: {result['avg_workspace_time']:.4f}s")
        print()
    
    return results


if __name__ == "__main__":
    print("Choose testing mode:")
    print("1. Test all methods with workspace extraction")
    print("2. Test workspace extraction only")
    print("3. Test individual method")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        compare_methods_with_workspace()
    
    elif choice == "2":
        print("\nTesting workspace extraction only:")
        test_workspace_extraction_only("jpg_discovery")
    
    elif choice == "3":
        print("\nTesting individual method:")
        print("Available methods:")
        print("1. Baseline (no preprocessing)")
        print("2. Adaptive preprocessing")
        print("3. Enhanced preprocessing")
        
        method_choice = input("Enter method (1-3): ").strip()
        extract_workspace = input("Extract workspace? (y/n): ").strip().lower() == 'y'
        
        if method_choice == "1":
            test_single_method_with_workspace("jpg_discovery", None, 
                                            method_name="baseline", 
                                            extract_workspace=extract_workspace)
        elif method_choice == "2":
            test_single_method_with_workspace("jpg_discovery", AdaptivePreprocessor(), 
                                            method_name="adaptive", 
                                            extract_workspace=extract_workspace)
        elif method_choice == "3":
            test_single_method_with_workspace("jpg_discovery", EnhancedPreprocessor(), 
                                            method_name="enhanced", 
                                            extract_workspace=extract_workspace)
        else:
            print("Invalid choice!")
    
    else:
        print("Invalid choice!")