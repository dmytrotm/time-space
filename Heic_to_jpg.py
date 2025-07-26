import os
from PIL import Image
import pillow_heif
import tarfile
import glob

def convert_heic_to_jpg(input_path, output_dir="jpg_discovery"):
    try:
        # Register HEIC support
        pillow_heif.register_heif_opener()
        
        # Load image
        image = Image.open(input_path)
        
        # Get base name without extension
        name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, name + ".jpg")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to RGB if necessary (HEIC might be in different color mode)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save the image
        image.save(output_path, "JPEG", quality=95)
        print(f"Converted: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return False

if __name__ == "__main__":
    archive_path = 'discovery.tar'  # Note: Check if this should be 'discovery.tar'
    extract_path = '.'
    
    # Extract tar file
    try:
        with tarfile.open(archive_path, 'r:*') as tar:
            tar.extractall(path=extract_path)
            print(f"Extracted to {extract_path}")
    except Exception as e:
        print(f"Error extracting archive: {str(e)}")
        exit(1)
    
    # Find HEIC files (case insensitive)
    heic_files = []
    
    # Check for both .heic and .HEIC extensions
    heic_files.extend(glob.glob(os.path.join(extract_path, "**", "*.heic"), recursive=True))
    heic_files.extend(glob.glob(os.path.join(extract_path, "**", "*.HEIC"), recursive=True))
    
    print(f"Found {len(heic_files)} HEIC files:")
    for heic_file in heic_files:
        print(f"  - {heic_file}")
    
    if not heic_files:
        print("No HEIC files found. Checking what files are in the extracted directory...")
        all_files = glob.glob(os.path.join(extract_path, "**", "*.*"), recursive=True)
        print("All files found:")
        for file in all_files[:10]:  # Show first 10 files
            print(f"  - {file}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
    else:
        # Convert each HEIC file
        successful_conversions = 0
        for heic in heic_files:
            if convert_heic_to_jpg(heic):
                successful_conversions += 1
        
        print(f"\nConversion complete: {successful_conversions}/{len(heic_files)} files converted successfully")