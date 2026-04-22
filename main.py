from core import (
    ImageServer,
    UIManager,
    VerificationManager,
    ResourceMonitor,
    init_profiler,
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
    parser.add_argument('--no-resource-monitor', action='store_true',
                        help='Disable resource monitoring (default: enabled)')
    parser.add_argument('--no-performance-profiler', action='store_true',
                        help='Disable performance profiling (default: enabled)')
    
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
        cameras = ImageServer(use_cameras=True, camera_ids=args.camera_ids)
        #cameras = ImageServer("Z1_0_1.png","Z2_0_1.png")

        verification_manager = VerificationManager()
        verification_manager.start()
        
        # Initialize performance profiler if not disabled
        performance_profiler = None
        if not args.no_performance_profiler:
            performance_profiler = init_profiler(enabled=True, log_file="performance_profile.json")
            print("Performance profiling enabled")
        else:
            print("Performance profiling disabled")
        
        # Initialize resource monitor if not disabled
        resource_monitor = None
        if not args.no_resource_monitor:
            resource_monitor = ResourceMonitor(sampling_interval=1.0, log_file="resource_monitor.log")
            resource_monitor.start_monitoring()
            print("Resource monitoring enabled")
        else:
            print("Resource monitoring disabled")
        
        ui_manager = UIManager(verification_manager, cameras, WINDOW_WIDTH, WINDOW_HEIGHT, resource_monitor, performance_profiler)
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
        if 'resource_monitor' in locals():
            resource_monitor.stop_monitoring()
            print("Resource monitoring data saved to resource_monitor.json and resource_monitor_summary.txt")
        if 'performance_profiler' in locals():
            performance_profiler.save_data()
            print("Performance profiling data saved to performance_profile.json")
        try:
            import sdl2
            import sdl2.ext
            import sdl2.sdlttf as sdlttf
            
            sdlttf.TTF_Quit()
            sdl2.ext.quit() 
        except Exception:
            pass
        print("Cleanup complete")