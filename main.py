from core import (
    ImageServer,
    UIManager,
    VerificationManager,
    ResourceMonitor,
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
    parser.add_argument('--save-all', action='store_true',
                        help='Save all photos (both success and error)')
    parser.add_argument('--save-errors', action='store_true',
                        help='Save only problematic error crop images')
    
    args = parser.parse_args()
    
    try:
        print("\nStarting TIME&SPACE Multi-Workspace System...")
        
        cameras = ImageServer(use_cameras=True, init_on_start=False)

        verification_manager = VerificationManager()
        # Hailo має апаратний лок на пристрій, тому ми використовуємо 1 воркер-процес
        # (який обробляє чергу з обох робочих місць)
        verification_manager.start(num_workers=1)
        
        # Initialize resource monitor if not disabled
        resource_monitor = None
        if not args.no_resource_monitor:
            resource_monitor = ResourceMonitor(sampling_interval=1.0, log_file="resource_monitor.log")
            resource_monitor.start_monitoring()
            print("Resource monitoring enabled")
        else:
            print("Resource monitoring disabled")
        
        ui_manager = UIManager(verification_manager, cameras, resource_monitor=resource_monitor, save_all=args.save_all, save_errors=args.save_errors)
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
        try:
            import sdl2
            import sdl2.ext
            import sdl2.sdlttf as sdlttf
            
            sdlttf.TTF_Quit()
            sdl2.ext.quit() 
        except Exception:
            pass
        print("Cleanup complete")