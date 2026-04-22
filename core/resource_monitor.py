import time
import threading
import psutil
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
import subprocess


class ResourceMonitor:
    """Monitors system resources (CPU, memory, GPU) during application runtime."""
    
    def __init__(self, sampling_interval: float = 1.0, log_file: Optional[str] = None):
        """
        Initialize the resource monitor.
        
        Args:
            sampling_interval: Time between samples in seconds
            log_file: Optional file path to save resource logs
        """
        self.sampling_interval = sampling_interval
        self.log_file = log_file or "resource_monitor.log"
        self.is_monitoring = False
        self.monitor_thread = None
        self.resource_data: List[Dict] = []
        self.start_time = None
        self.process = psutil.Process()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def start_monitoring(self):
        """Start resource monitoring in a separate thread."""
        if self.is_monitoring:
            self.logger.warning("Resource monitoring is already running")
            return
            
        self.is_monitoring = True
        self.start_time = time.time()
        self.resource_data.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring and save data."""
        if not self.is_monitoring:
            self.logger.warning("Resource monitoring is not running")
            return
            
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        self._save_data()
        self.logger.info("Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                sample = self._collect_sample()
                self.resource_data.append(sample)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error collecting resource sample: {e}")
                
    def _collect_sample(self) -> Dict:
        """Collect current resource usage sample."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_temp = self._get_cpu_temperature()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        # GPU metrics (if available)
        gpu_info = self._get_gpu_info()
        
        sample = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "cpu": {
                "percent_total": cpu_percent,
                "count": cpu_count,
                "process_percent": self.process.cpu_percent(),
                "temperature_c": cpu_temp
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent,
                "process_rss_mb": process_memory.rss / (1024**2),
                "process_vms_mb": process_memory.vms / (1024**2)
            },
            "gpu": gpu_info,
            "disk": self._get_disk_info(),
            "platform": "raspberry_pi" if cpu_temp is not None else "unknown"
        }
        
        return sample
        
    def _get_gpu_info(self) -> Dict:
        """Get GPU information if available."""
        gpu_info = {"available": False}
        
        # Try Hailo 8L first (Raspberry Pi specific)
        hailo_info = self._get_hailo_info()
        if hailo_info["available"]:
            return hailo_info
            
        # Fallback to standard GPU monitoring
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_info = {
                    "available": True,
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "load_percent": gpu.load * 100,
                    "temperature_c": gpu.temperature
                }
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"Could not get GPU info: {e}")
            
        return gpu_info
        
    def _get_hailo_info(self) -> Dict:
        """Get Hailo 8L accelerator information."""
        hailo_info = {"available": False, "type": "hailo"}
        
        # Method 1: Check for Hailo device files
        if self._check_hailo_device_files():
            hailo_info.update({
                "available": True,
                "name": "Hailo 8L",
                "status": "connected",
                "detection_method": "device_files"
            })
            
        # Method 2: Try hailort CLI
        if not hailo_info["available"]:
            try:
                result = subprocess.run(['hailort', 'query'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    hailo_info.update({
                        "available": True,
                        "name": "Hailo 8L",
                        "status": "connected",
                        "detection_method": "hailort_cli"
                    })
                    
            except FileNotFoundError:
                self.logger.debug("hailort CLI not found")
            except subprocess.TimeoutExpired:
                self.logger.debug("Hailo query timeout")
            except Exception as e:
                self.logger.debug(f"hailort query failed: {e}")
        
        # Method 3: Try Python Hailo RT library
        if not hailo_info["available"]:
            if self._check_hailo_python():
                hailo_info.update({
                    "available": True,
                    "name": "Hailo 8L",
                    "status": "connected",
                    "detection_method": "python_library"
                })
        
        # If Hailo is detected, try to get more info
        if hailo_info["available"]:
            self._get_hailo_detailed_info(hailo_info)
            
        return hailo_info
        
    def _check_hailo_device_files(self) -> bool:
        """Check for Hailo device files in /dev/."""
        import glob
        try:
            # Look for Hailo device files
            hailo_devices = glob.glob('/dev/hailo*')
            if hailo_devices:
                self.logger.debug(f"Found Hailo devices: {hailo_devices}")
                return True
                
            # Also check for PCI device
            pci_devices = glob.glob('/sys/bus/pci/devices/*/hw/hailo*')
            if pci_devices:
                self.logger.debug(f"Found Hailo PCI devices: {pci_devices}")
                return True
                
        except Exception as e:
            self.logger.debug(f"Error checking Hailo device files: {e}")
            
        return False
        
    def _check_hailo_python(self) -> bool:
        """Check if Hailo Python library is available."""
        try:
            import hailo
            self.logger.debug("Hailo Python library found")
            return True
        except ImportError:
            self.logger.debug("Hailo Python library not found")
            return False
        except Exception as e:
            self.logger.debug(f"Error importing Hailo library: {e}")
            return False
            
    def _get_hailo_detailed_info(self, hailo_info: Dict):
        """Get detailed Hailo information."""
        # Try to get firmware version
        try:
            fw_result = subprocess.run(['hailort', 'fw-control', '--get-fw-version'], 
                                     capture_output=True, text=True, timeout=5)
            if fw_result.returncode == 0:
                hailo_info["firmware_version"] = fw_result.stdout.strip()
        except:
            pass
            
        # Try to get device info
        try:
            info_result = subprocess.run(['hailort', 'info'], 
                                       capture_output=True, text=True, timeout=5)
            if info_result.returncode == 0:
                hailo_info["device_info"] = info_result.stdout.strip()
        except:
            pass
            
        # Try to get utilization monitoring capability
        try:
            util_result = subprocess.run(['hailort', 'utilization-monitor', '--timeout-ms', '1000'], 
                                       capture_output=True, text=True, timeout=5)
            if util_result.returncode == 0:
                hailo_info["utilization_available"] = True
                hailo_info["utilization_output"] = util_result.stdout.strip()
            else:
                hailo_info["utilization_available"] = False
        except:
            hailo_info["utilization_available"] = False
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature for Raspberry Pi."""
        try:
            # Try to read CPU temperature from Raspberry Pi specific location
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_millidegrees = float(f.read().strip())
                return temp_millidegrees / 1000.0  # Convert to Celsius
        except FileNotFoundError:
            # Not a Raspberry Pi or thermal zone not available
            pass
        except Exception as e:
            self.logger.debug(f"Could not read CPU temperature: {e}")
            
        return None
        
    def _get_disk_info(self) -> Dict:
        """Get disk usage information."""
        try:
            disk = psutil.disk_usage('/')
            return {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": (disk.used / disk.total) * 100
            }
        except Exception as e:
            self.logger.debug(f"Could not get disk info: {e}")
            return {"error": str(e)}
            
    def _save_data(self):
        """Save collected resource data to file."""
        if not self.resource_data:
            return
            
        try:
            # Save as JSON
            json_file = self.log_file.replace('.log', '.json')
            with open(json_file, 'w') as f:
                json.dump({
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                    "total_samples": len(self.resource_data),
                    "sampling_interval": self.sampling_interval,
                    "data": self.resource_data
                }, f, indent=2)
                
            # Save summary statistics
            self._save_summary()
            
        except Exception as e:
            self.logger.error(f"Error saving resource data: {e}")
            
    def _save_summary(self):
        """Save summary statistics of resource usage."""
        if not self.resource_data:
            return
            
        summary_file = self.log_file.replace('.log', '_summary.txt')
        
        # Calculate statistics
        cpu_values = [sample["cpu"]["percent_total"] for sample in self.resource_data]
        memory_values = [sample["memory"]["percent_used"] for sample in self.resource_data]
        process_memory_values = [sample["memory"]["process_rss_mb"] for sample in self.resource_data]
        cpu_temps = [sample["cpu"]["temperature_c"] for sample in self.resource_data if sample["cpu"]["temperature_c"] is not None]
        
        total_duration = self.resource_data[-1]["elapsed_time"] if self.resource_data else 0
        
        summary = f"""
Resource Monitoring Summary
==========================
Duration: {total_duration:.2f} seconds
Samples collected: {len(self.resource_data)}
Sampling interval: {self.sampling_interval} seconds

CPU Usage:
- Average: {sum(cpu_values) / len(cpu_values):.2f}%
- Maximum: {max(cpu_values):.2f}%
- Minimum: {min(cpu_values):.2f}%

System Memory:
- Average usage: {sum(memory_values) / len(memory_values):.2f}%
- Maximum usage: {max(memory_values):.2f}%
- Minimum usage: {min(memory_values):.2f}%

Process Memory:
- Average RSS: {sum(process_memory_values) / len(process_memory_values):.2f} MB
- Maximum RSS: {max(process_memory_values):.2f} MB
- Minimum RSS: {min(process_memory_values):.2f} MB
"""

        # Add CPU temperature if available
        if cpu_temps:
            summary += f"""
CPU Temperature:
- Average: {sum(cpu_temps) / len(cpu_temps):.1f}°C
- Maximum: {max(cpu_temps):.1f}°C
- Minimum: {min(cpu_temps):.1f}°C
"""

        summary += """
GPU Information:
"""
        
        # Add GPU info if available
        if self.resource_data and self.resource_data[0]["gpu"]["available"]:
            gpu_info = self.resource_data[0]["gpu"]
            
            if gpu_info.get("type") == "hailo":
                # Hailo 8L specific information
                summary += f"""- Accelerator: {gpu_info.get("name", "Hailo 8L")}
- Status: {gpu_info.get("status", "Unknown")}
- Detection method: {gpu_info.get("detection_method", "Unknown")}
- Firmware: {gpu_info.get("firmware_version", "Not available")}
- Utilization monitoring: {"Available" if gpu_info.get("utilization_available") else "Not available"}
"""
                if gpu_info.get("device_info"):
                    summary += f"- Device info: {gpu_info.get('device_info')}\n"
            else:
                # Standard GPU information
                gpu_loads = [sample["gpu"]["load_percent"] for sample in self.resource_data if sample["gpu"]["available"]]
                gpu_memory_usage = [sample["gpu"]["memory_used_mb"] for sample in self.resource_data if sample["gpu"]["available"]]
                
                if gpu_loads:
                    summary += f"""- GPU Name: {gpu_info.get("name", "Unknown")}
- Average load: {sum(gpu_loads) / len(gpu_loads):.2f}%
- Maximum load: {max(gpu_loads):.2f}%
- Average memory usage: {sum(gpu_memory_usage) / len(gpu_memory_usage):.2f} MB
- Maximum memory usage: {max(gpu_memory_usage):.2f} MB
"""
        else:
            summary += "No GPU/Hailo information available\n"
            
        try:
            with open(summary_file, 'w') as f:
                f.write(summary)
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
            
    def debug_hailo_detection(self) -> Dict:
        """Debug method to check Hailo detection status."""
        debug_info = {
            "device_files_check": False,
            "hailort_cli_check": False,
            "python_library_check": False,
            "hailort_output": None,
            "device_files_found": [],
            "errors": []
        }
        
        # Check device files
        try:
            import glob
            hailo_devices = glob.glob('/dev/hailo*')
            pci_devices = glob.glob('/sys/bus/pci/devices/*/hw/hailo*')
            debug_info["device_files_found"] = hailo_devices + pci_devices
            debug_info["device_files_check"] = len(debug_info["device_files_found"]) > 0
        except Exception as e:
            debug_info["errors"].append(f"Device file check error: {e}")
        
        # Check hailort CLI
        try:
            result = subprocess.run(['which', 'hailort'], 
                                  capture_output=True, text=True, timeout=5)
            debug_info["hailort_cli_check"] = result.returncode == 0
            
            if result.returncode == 0:
                # Try to run hailort query
                query_result = subprocess.run(['hailort', 'query'], 
                                            capture_output=True, text=True, timeout=5)
                debug_info["hailort_output"] = {
                    "returncode": query_result.returncode,
                    "stdout": query_result.stdout,
                    "stderr": query_result.stderr
                }
        except Exception as e:
            debug_info["errors"].append(f"hailort CLI check error: {e}")
        
        # Check Python library
        try:
            import hailo
            debug_info["python_library_check"] = True
        except ImportError:
            debug_info["python_library_check"] = False
        except Exception as e:
            debug_info["errors"].append(f"Python library check error: {e}")
        
        return debug_info
        
    def get_current_stats(self) -> Dict:
        """Get current resource statistics."""
        return self._collect_sample()
        
    def get_average_stats(self) -> Optional[Dict]:
        """Get average statistics from collected data."""
        if not self.resource_data:
            return None
            
        cpu_values = [sample["cpu"]["percent_total"] for sample in self.resource_data]
        memory_values = [sample["memory"]["percent_used"] for sample in self.resource_data]
        process_memory_values = [sample["memory"]["process_rss_mb"] for sample in self.resource_data]
        
        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "memory_avg": sum(memory_values) / len(memory_values),
            "process_memory_avg_mb": sum(process_memory_values) / len(process_memory_values),
            "samples_count": len(self.resource_data),
            "duration_seconds": self.resource_data[-1]["elapsed_time"] if self.resource_data else 0
        }
