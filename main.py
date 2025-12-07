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

if __name__ == "__main__":
    """Main entry point"""
    
    # Ensure multiprocessing works correctly
    multiprocessing.set_start_method('spawn', force=True)

    DEFAULT_Z1_IMAGE_PATH = "Z1_0_1.png"
    DEFAULT_Z2_IMAGE_PATH = "Z2_0_1.png"

    try:
        # Initialize components
        cameras = ImageServer(DEFAULT_Z1_IMAGE_PATH, DEFAULT_Z2_IMAGE_PATH)
        
        # Initialize Verification Manager (starts worker process)
        verification_manager = VerificationManager()
        verification_manager.start()

        # Create and run UI
        ui_manager = UIManager(verification_manager, cameras, WINDOW_WIDTH, WINDOW_HEIGHT)
        ui_manager.main_loop()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if 'verification_manager' in locals():
            verification_manager.stop()
