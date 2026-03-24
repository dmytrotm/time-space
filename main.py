from core import (
    ImageServer,
    UIManager,
    VerificationManager,
)
from utils.constants import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)
import multiprocessing
import argparse

if __name__ == "__main__":
    """Main entry point"""
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Run the verification system with cameras')
    parser.add_argument('--camera-ids', type=int, nargs=2, default=[0, 1],
                        help='Camera IDs to use (default: 0 1)')
    
    args = parser.parse_args()
    
    try:
        print("Scanning for available cameras...")
        import cv2
        available_cameras = []
        for i in range(10):  
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    print(f"  Found camera at index {i}")
                cap.release()
        
        if not available_cameras:
            print("ERROR: No cameras detected on the system!")
            print("Please check:")
            print("  - Cameras are connected")
            print("  - Camera permissions (try: sudo usermod -a -G video $USER)")
            print("  - V4L2 drivers are installed")
            exit(1)
        
        print(f"\nAvailable cameras: {available_cameras}")
        
        if args.camera_ids[0] not in available_cameras or args.camera_ids[1] not in available_cameras:
            print(f"Warning: Requested cameras {args.camera_ids} not available")
            if len(available_cameras) >= 2:
                args.camera_ids = available_cameras[:2]
                print(f"Using first two available cameras: {args.camera_ids}")
            else:
                print(f"ERROR: Need at least 2 cameras, only found {len(available_cameras)}")
                exit(1)
        
        print(f"\nInitializing cameras: {args.camera_ids}")
        camera_paths = [
    "/dev/v4l/by-path/platform-xhci-hcd.0-usb-0:1:1.0-video-index0",
    "/dev/v4l/by-path/platform-xhci-hcd.1-usb-0:1.4:1.0-video-index0"
]
        cameras = ImageServer(use_cameras=True, camera_ids=camera_paths)
        #cameras = ImageServer("Z1_0_1.png","Z2_0_1.png")

        verification_manager = VerificationManager()
        verification_manager.start()
        
        ui_manager = UIManager(verification_manager, cameras, WINDOW_WIDTH, WINDOW_HEIGHT)
        ui_manager.main_loop()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'verification_manager' in locals():
            verification_manager.stop()
        if 'cameras' in locals():
            cameras.release()
        try:
            import sdl2
            import sdl2.ext
            import sdl2.sdlttf as sdlttf
            
            sdlttf.TTF_Quit()
            sdl2.ext.quit() 
        except Exception:
            pass
        print("Cleanup complete")