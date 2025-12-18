import multiprocessing
from core.worker import worker_logic
from utils.constants import (
    CONFIG_ROI_Z1_PATH,
    CONFIG_ROI_Z2_PATH,
    CONFIG_POSITIONS_PATH,
    ENVIRONMENT_CONFIG
)

class VerificationManager:
    def __init__(self):
        self.command_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.process = None
        self.config_paths = {
            "roi_z1": CONFIG_ROI_Z1_PATH,
            "roi_z2": CONFIG_ROI_Z2_PATH,
            "positions": CONFIG_POSITIONS_PATH,
            "env":ENVIRONMENT_CONFIG
        }

    def start(self):
        """Start the worker process."""
        if self.process is None or not self.process.is_alive():
            self.process = multiprocessing.Process(
                target=worker_logic,
                args=(self.command_queue, self.result_queue, self.config_paths),
                daemon=True
            )
            self.process.start()
            print("Verification Worker Process Started.")

    def stop(self):
        """Stop the worker process."""
        if self.process and self.process.is_alive():
            self.command_queue.put({"command": "STOP"})
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
            print("Verification Worker Process Stopped.")

    def trigger_verification(self, images):
        """
        Trigger a verification cycle.
        
        Args:
            images: List of images (numpy arrays) to process.
        """
        self.command_queue.put({"command": "TRIGGER", "images": images})

    def check_results(self):
        """
        Check for results from the worker.
        
        Returns:
            dict or None: Result dictionary if available, else None.
        """
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
