import multiprocessing
from core.worker import worker_logic
from configs.config import (
    CONFIG_ROI_Z1_PATH,
    CONFIG_ROI_Z2_PATH,
    CONFIG_POSITIONS_PATH,
    ENVIRONMENT_CONFIG
)

class VerificationManager:
    def __init__(self):
        self.command_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.processes = []
        self.config_paths = {
            "roi_z1": CONFIG_ROI_Z1_PATH,
            "roi_z2": CONFIG_ROI_Z2_PATH,
            "positions": CONFIG_POSITIONS_PATH,
            "env": ENVIRONMENT_CONFIG
        }

    def start(self, num_workers=2):
        """Start the worker processes."""
        for i in range(num_workers):
            p = multiprocessing.Process(
                target=worker_logic,
                args=(self.command_queue, self.result_queue, self.config_paths),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            print(f"Verification Worker Process {i+1} Started.")

    def stop(self):
        """Stop all worker processes."""
        for _ in self.processes:
            self.command_queue.put({"command": "STOP"})
        
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        self.processes.clear()
        print("All Verification Worker Processes Stopped.")

    def trigger_verification(self, workspace_id, images, save_errors=False):
        """
        Trigger a verification cycle.
        
        Args:
            workspace_id: ID of the workspace (1 or 2).
            images: List of images (numpy arrays) to process.
            save_errors: If True, worker will collect error images for saving.
        """
        self.command_queue.put({"command": "TRIGGER", "workspace_id": workspace_id, "images": images, "save_errors": save_errors})

    def check_results(self):
        """
        Check for results from the worker.
        
        Returns:
            dict or None: Result dictionary if available, else None.
        """
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
