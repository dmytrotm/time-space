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
    # Ensure multiprocessing works correctly
    multiprocessing.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the verification system with cameras')
    parser.add_argument('--camera-ids', type=int, nargs=2, default=[0, 1],
                        help='Camera IDs to use (default: 0 1)')
    
    args = parser.parse_args()
    
    try:
        # Initialize ImageServer with cameras
        print(f"Initializing cameras: {args.camera_ids}")
        cameras = ImageServer(use_cameras=True, camera_ids=args.camera_ids)
        
        # Setup cameras and detect zones
        setup_success = cameras.setup()
        if not setup_success:
            print("Warning: Zone detection failed, continuing with default order")
        
        # Initialize Verification Manager (starts worker process)
        verification_manager = VerificationManager()
        verification_manager.start()
        
        # Create and run UI
        ui_manager = UIManager(verification_manager, cameras, WINDOW_WIDTH, WINDOW_HEIGHT)
        ui_manager.main_loop()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if 'verification_manager' in locals():
            verification_manager.stop()
        if 'cameras' in locals():
            cameras.release()
        print("Cleanup complete")