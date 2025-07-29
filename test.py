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
                                     extract_workspace=True, use_original_size=True, 
                                     fixed_size=(800, 600)):
    """
    Test with a single preprocessing method and optional workspace extraction
    
    Args:
        use_original_size: If True, uses original calculated size. If False, uses fixed_size
        fixed_size: Size to use when use_original_size is False
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
    workspace_sizes = {}  # Track extracted workspace sizes
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
                
                # Determine output size based on settings
                output_size = None if use_original_size else fixed_size
                
                # Try automatic rectangle detection
                workspace = extractor.extract_workspace_from_rectangle(
                    image, corners, ids, output_size
                )
                
                elapsed_workspace = time.time() - start_workspace_time
                total_workspace_time += elapsed_workspace
                
                if workspace is not None:
                    # Save extracted workspace
                    workspace_path = os.path.join(workspace_dir, f"{name_without_ext}_workspace.jpg")
                    cv2.imwrite(workspace_path, workspace)
                    workspace_extracted[image_file] = True
                    workspace_sizes[image_file] = (workspace.shape[1], workspace.shape[0])  # (width, height)
                    
                    # Save visualization of rectangle detection
                    viz = extractor.visualize_rectangle_detection(image, corners, ids)
                    viz_path = os.path.join(workspace_viz_dir, f"{name_without_ext}_viz.jpg")
                    cv2.imwrite(viz_path, viz)
                    
                    print(f"✓ Workspace extracted from {base_name}: {workspace.shape[1]}x{workspace.shape[0]}")
                else:
                    workspace_extracted[image_file] = False
                    workspace_sizes[image_file] = (0, 0)
                    print(f"✗ Failed to extract workspace from: {base_name}")
            
            # Check if expected number of markers found
            if count != 6:
                failed_images[image_file] = count
                cv2.imwrite(os.path.join(failed_dir, base_name), output)
        else:
            failed_images[image_file] = 0
            # Save original image to failed directory for reference
            cv2.imwrite(os.path.join(failed_dir, base_name), image)
            workspace_extracted[image_file] = False
            workspace_sizes[image_file] = (0, 0)
    
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
        print(f"Size mode: {'Original size' if use_original_size else f'Fixed size {fixed_size}'}")
        
        # Show workspace size statistics
        if successful_workspaces > 0:
            valid_sizes = [size for size in workspace_sizes.values() if size != (0, 0)]
            if valid_sizes:
                widths = [size[0] for size in valid_sizes]
                heights = [size[1] for size in valid_sizes]
                print(f"Workspace sizes - Width: {min(widths)}-{max(widths)}, Height: {min(heights)}-{max(heights)}")
        
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
        'workspace_sizes': workspace_sizes,
        'avg_detection_time': avg_time,
        'avg_preprocess_time': avg_preprocess_time,
        'avg_workspace_time': avg_workspace_time
    }


def test_workspace_extraction_only(dir_name, output_dir="workspace_only", use_original_size=True):
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
    
    print(f"Testing workspace extraction with {'original size' if use_original_size else 'fixed size'}")
    print("-" * 50)
    
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
            
            # Method 1: Automatic rectangle detection with original or fixed size
            output_size = None if use_original_size else (800, 600)
            workspace_rect = extractor.extract_workspace_from_rectangle(image, corners, ids, output_size)
            
            # Method 2: Auto crop with margin (always uses original size)
            workspace_auto = extractor.extract_workspace_auto(image, corners, ids, margin=50)
            
            # Save results
            if workspace_rect is not None:
                rect_path = os.path.join(workspace_dir, f"{name_without_ext}_rectangle.jpg")
                cv2.imwrite(rect_path, workspace_rect)
                results[f"{name_without_ext}_rectangle"] = True
                rect_size = f"{workspace_rect.shape[1]}x{workspace_rect.shape[0]}"
            else:
                results[f"{name_without_ext}_rectangle"] = False
                rect_size = "failed"
            
            if workspace_auto is not None:
                auto_path = os.path.join(workspace_dir, f"{name_without_ext}_auto.jpg")
                cv2.imwrite(auto_path, workspace_auto)
                results[f"{name_without_ext}_auto"] = True
                auto_size = f"{workspace_auto.shape[1]}x{workspace_auto.shape[0]}"
            else:
                results[f"{name_without_ext}_auto"] = False
                auto_size = "failed"
            
            # Save visualization
            viz = extractor.visualize_rectangle_detection(image, corners, ids)
            cv2.imwrite(os.path.join(viz_dir, f"{name_without_ext}_viz.jpg"), viz)
            
            print(f"Processed: {base_name}")
            print(f"  Markers found: {len(corners)}")
            print(f"  Rectangle method: {'✓' if workspace_rect is not None else '✗'} ({rect_size})")
            print(f"  Auto method: {'✓' if workspace_auto is not None else '✗'} ({auto_size})")
            print()
        else:
            print(f"Skipped {base_name}: insufficient markers ({len(corners) if corners else 0})")
            results[name_without_ext] = False
    
    return results


def compare_methods_with_workspace(use_original_size=True):
    """
    Compare all preprocessing methods with workspace extraction
    """
    methods = [
        ("baseline", None),
        ("adaptive", AdaptivePreprocessor()),
        ("enhanced", EnhancedPreprocessor())
    ]
    
    results = {}
    
    size_mode = "ORIGINAL SIZE" if use_original_size else "FIXED SIZE"
    print("=" * 60)
    print(f"TESTING ALL METHODS WITH WORKSPACE EXTRACTION ({size_mode})")
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
            use_original_size=use_original_size,
            fixed_size=(800, 600)
        )
        
        results[method_name] = result
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    for method_name, result in results.items():
        successful_workspaces = sum(1 for success in result['workspace_extracted'].values() if success)
        
        # Calculate workspace size statistics
        valid_sizes = [size for size in result['workspace_sizes'].values() if size != (0, 0)]
        if valid_sizes:
            widths = [size[0] for size in valid_sizes]
            heights = [size[1] for size in valid_sizes]
            size_info = f"W:{min(widths)}-{max(widths)}, H:{min(heights)}-{max(heights)}"
        else:
            size_info = "No valid sizes"
        
        print(f"{method_name.upper()}:")
        print(f"  Marker Detection: {result['successful_detections']}/{result['total_images']}")
        print(f"  Workspace Extraction: {successful_workspaces}/{result['total_images']}")
        print(f"  Workspace Sizes: {size_info}")
        print(f"  Avg Detection Time: {result['avg_detection_time']:.4f}s")
        print(f"  Avg Workspace Time: {result['avg_workspace_time']:.4f}s")
        print()
    
    return results


def test_size_comparison():
    """
    Compare original size vs fixed size extraction
    """
    print("=" * 60)
    print("COMPARING ORIGINAL SIZE VS FIXED SIZE EXTRACTION")
    print("=" * 60)
    
    # Test with original size
    print("\n1. TESTING WITH ORIGINAL SIZE:")
    print("-" * 40)
    result_original = test_single_method_with_workspace(
        "jpg_discovery", 
        None,  # No preprocessor
        output_dir="output_original_size",
        method_name="original_size",
        extract_workspace=True,
        use_original_size=True
    )
    
    # Test with fixed size
    print("\n2. TESTING WITH FIXED SIZE (800x600):")
    print("-" * 40)
    result_fixed = test_single_method_with_workspace(
        "jpg_discovery", 
        None,  # No preprocessor
        output_dir="output_fixed_size",
        method_name="fixed_size",
        extract_workspace=True,
        use_original_size=False,
        fixed_size=(800, 600)
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("SIZE COMPARISON SUMMARY")
    print("=" * 60)
    
    for name, result in [("Original Size", result_original), ("Fixed Size", result_fixed)]:
        successful_workspaces = sum(1 for success in result['workspace_extracted'].values() if success)
        valid_sizes = [size for size in result['workspace_sizes'].values() if size != (0, 0)]
        
        if valid_sizes:
            widths = [size[0] for size in valid_sizes]
            heights = [size[1] for size in valid_sizes]
            size_info = f"W:{min(widths)}-{max(widths)}, H:{min(heights)}-{max(heights)}"
        else:
            size_info = "No valid sizes"
        
        print(f"{name}:")
        print(f"  Successful Extractions: {successful_workspaces}/{result['total_images']}")
        print(f"  Size Range: {size_info}")
        print(f"  Avg Workspace Time: {result['avg_workspace_time']:.4f}s")
        print()


if __name__ == "__main__":
    print("Choose testing mode:")
    print("1. Test all methods with workspace extraction (original size)")
    print("2. Test all methods with workspace extraction (fixed size)")
    print("3. Test workspace extraction only")
    print("4. Test individual method")
    print("5. Compare original size vs fixed size")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        compare_methods_with_workspace(use_original_size=True)
    
    elif choice == "2":
        compare_methods_with_workspace(use_original_size=False)
    
    elif choice == "3":
        use_original = input("Use original size? (y/n): ").strip().lower() == 'y'
        print(f"\nTesting workspace extraction only ({'original size' if use_original else 'fixed size'}):")
        test_workspace_extraction_only("jpg_discovery", use_original_size=use_original)
    
    elif choice == "4":
        print("\nTesting individual method:")
        print("Available methods:")
        print("1. Baseline (no preprocessing)")
        print("2. Adaptive preprocessing")
        print("3. Enhanced preprocessing")
        
        method_choice = input("Enter method (1-3): ").strip()
        extract_workspace = input("Extract workspace? (y/n): ").strip().lower() == 'y'
        
        if extract_workspace:
            use_original = input("Use original size? (y/n): ").strip().lower() == 'y'
        else:
            use_original = True
        
        if method_choice == "1":
            test_single_method_with_workspace("jpg_discovery", None, 
                                            method_name="baseline", 
                                            extract_workspace=extract_workspace,
                                            use_original_size=use_original)
        elif method_choice == "2":
            test_single_method_with_workspace("jpg_discovery", AdaptivePreprocessor(), 
                                            method_name="adaptive", 
                                            extract_workspace=extract_workspace,
                                            use_original_size=use_original)
        elif method_choice == "3":
            test_single_method_with_workspace("jpg_discovery", EnhancedPreprocessor(), 
                                            method_name="enhanced", 
                                            extract_workspace=extract_workspace,
                                            use_original_size=use_original)
        else:
            print("Invalid choice!")
    
    elif choice == "5":
        test_size_comparison()
    
    else:
        print("Invalid choice!")