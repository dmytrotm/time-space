"""
Entry point that runs main.py + workspace_server.py as parallel processes.

Usage:
    python start_web.py --camera-ids 0 2 [--port 8765]
"""
import multiprocessing
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def run_workspace_server(shared_state, shared_frames, cmd_queue, port: int):
    import uvicorn
    from utils.workspace_server import app, init
    init(shared_state, shared_frames, cmd_queue)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def run_main(shared_state, shared_frames, cmd_queue, save_all: bool, save_errors: bool):
    from core import ImageServer, UIManager, VerificationManager, ResourceMonitor

    try:
        print("\nStarting TIME&SPACE Multi-Workspace System...")

        cameras = ImageServer(use_cameras=True, init_on_start=False)
        vm = VerificationManager()
        vm.start(num_workers=1)

        resource_monitor = ResourceMonitor(sampling_interval=1.0, log_file="resource_monitor.log")
        resource_monitor.start_monitoring()

        ui = UIManager(
            vm, cameras,
            resource_monitor=resource_monitor,
            save_all=save_all,
            save_errors=save_errors,
            shared_state=shared_state,
            shared_frames=shared_frames,
            cmd_queue=cmd_queue,
        )
        ui.main_loop()

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'vm' in locals():
            vm.stop()
        if 'cameras' in locals():
            cameras.release()
        if 'resource_monitor' in locals():
            resource_monitor.stop_monitoring()
        try:
            import sdl2
            import sdl2.ext
            import sdl2.sdlttf as sdlttf
            sdlttf.TTF_Quit()
            sdl2.ext.quit()
        except Exception:
            pass
        print("Cleanup complete")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Run TIME&SPACE with web monitor')
    parser.add_argument('--camera-ids', type=int, nargs=2, default=[0, 2],
                        help='Camera IDs (default: 0 2)')
    parser.add_argument('--port', type=int, default=8765,
                        help='Workspace server port (default: 8765)')
    parser.add_argument('--save-all', action='store_true')
    parser.add_argument('--save-errors', action='store_true')
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    shared_state = manager.dict({
        'ws1_state': 'INIT',
        'ws2_state': 'INIT',
        'ws1_error': '',
        'ws2_error': '',
        'cameras': [],
    })
    shared_frames = manager.dict()
    cmd_queue = manager.Queue()

    server_proc = multiprocessing.Process(
        target=run_workspace_server,
        args=(shared_state, shared_frames, cmd_queue, args.port),
        daemon=True,
        name="workspace-server",
    )
    server_proc.start()
    print(f"Workspace server started on port {args.port}")

    try:
        run_main(shared_state, shared_frames, cmd_queue,
                 save_all=args.save_all, save_errors=args.save_errors)
    finally:
        server_proc.terminate()
        server_proc.join(timeout=3)
        manager.shutdown()
