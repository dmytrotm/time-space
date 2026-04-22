import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os


class ResourceVisualizer:
    """Convert resource monitoring JSON data to visual charts and images."""
    
    def __init__(self, json_file: str, output_dir: str = "resource_charts"):
        """
        Initialize the visualizer.
        
        Args:
            json_file: Path to the resource monitor JSON file
            output_dir: Directory to save generated charts
        """
        self.json_file = json_file
        self.output_dir = output_dir
        self.data = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def load_data(self) -> bool:
        """Load resource monitoring data from JSON file."""
        try:
            with open(self.json_file, 'r') as f:
                self.data = json.load(f)
            return True
        except FileNotFoundError:
            print(f"Error: JSON file {self.json_file} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return False
            
    def create_cpu_chart(self, save_path: Optional[str] = None) -> str:
        """Create CPU usage chart."""
        if not self.data or not self.data.get('data'):
            raise ValueError("No data available")
            
        samples = self.data['data']
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples]
        cpu_usage = [s['cpu']['percent_total'] for s in samples]
        cpu_process = [s['cpu']['process_percent'] for s in samples]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Total CPU usage
        ax1.plot(timestamps, cpu_usage, 'b-', linewidth=2, label='Total CPU')
        ax1.fill_between(timestamps, cpu_usage, alpha=0.3)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('Total CPU Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Process CPU usage
        ax2.plot(timestamps, cpu_process, 'r-', linewidth=2, label='Process CPU')
        ax2.fill_between(timestamps, cpu_process, alpha=0.3, color='red')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_xlabel('Time')
        ax2.set_title('Process CPU Usage Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'cpu_usage.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_memory_chart(self, save_path: Optional[str] = None) -> str:
        """Create memory usage chart."""
        if not self.data or not self.data.get('data'):
            raise ValueError("No data available")
            
        samples = self.data['data']
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples]
        memory_usage = [s['memory']['percent_used'] for s in samples]
        memory_rss = [s['memory']['process_rss_mb'] for s in samples]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # System memory usage
        ax1.plot(timestamps, memory_usage, 'g-', linewidth=2, label='System Memory')
        ax1.fill_between(timestamps, memory_usage, alpha=0.3, color='green')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.set_title('System Memory Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Process memory usage
        ax2.plot(timestamps, memory_rss, 'm-', linewidth=2, label='Process RSS')
        ax2.fill_between(timestamps, memory_rss, alpha=0.3, color='magenta')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_xlabel('Time')
        ax2.set_title('Process Memory Usage (RSS) Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'memory_usage.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_temperature_chart(self, save_path: Optional[str] = None) -> Optional[str]:
        """Create CPU temperature chart if temperature data is available."""
        if not self.data or not self.data.get('data'):
            raise ValueError("No data available")
            
        samples = self.data['data']
        temperatures = [s['cpu']['temperature_c'] for s in samples if s['cpu']['temperature_c'] is not None]
        
        if not temperatures:
            print("No temperature data available")
            return None
            
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples if s['cpu']['temperature_c'] is not None]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(timestamps, temperatures, 'r-', linewidth=2, label='CPU Temperature')
        ax.fill_between(timestamps, temperatures, alpha=0.3, color='red')
        ax.set_ylabel('Temperature (°C)')
        ax.set_xlabel('Time')
        ax.set_title('CPU Temperature Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add warning line at 80°C
        ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Warning (80°C)')
        ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Critical (85°C)')
        
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'cpu_temperature.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_dashboard(self, save_path: Optional[str] = None) -> str:
        """Create a comprehensive dashboard with all metrics."""
        if not self.data or not self.data.get('data'):
            raise ValueError("No data available")
            
        samples = self.data['data']
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resource Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # CPU Usage
        cpu_usage = [s['cpu']['percent_total'] for s in samples]
        ax1.plot(timestamps, cpu_usage, 'b-', linewidth=2)
        ax1.fill_between(timestamps, cpu_usage, alpha=0.3)
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('Total CPU Usage')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Memory Usage
        memory_usage = [s['memory']['percent_used'] for s in samples]
        ax2.plot(timestamps, memory_usage, 'g-', linewidth=2)
        ax2.fill_between(timestamps, memory_usage, alpha=0.3, color='green')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.set_title('System Memory Usage')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Process Memory
        memory_rss = [s['memory']['process_rss_mb'] for s in samples]
        ax3.plot(timestamps, memory_rss, 'm-', linewidth=2)
        ax3.fill_between(timestamps, memory_rss, alpha=0.3, color='magenta')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xlabel('Time')
        ax3.set_title('Process Memory (RSS)')
        ax3.grid(True, alpha=0.3)
        
        # Temperature (if available)
        temperatures = [s['cpu']['temperature_c'] for s in samples if s['cpu']['temperature_c'] is not None]
        if temperatures:
            temp_timestamps = [datetime.fromisoformat(s['timestamp']) for s in samples if s['cpu']['temperature_c'] is not None]
            ax4.plot(temp_timestamps, temperatures, 'r-', linewidth=2)
            ax4.fill_between(temp_timestamps, temperatures, alpha=0.3, color='red')
            ax4.set_ylabel('Temperature (°C)')
            ax4.set_xlabel('Time')
            ax4.set_title('CPU Temperature')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning')
            ax4.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Critical')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No Temperature\nData Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('CPU Temperature')
        
        # Format all x-axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'resource_dashboard.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_all_charts(self) -> Dict[str, str]:
        """Generate all available charts and return paths."""
        if not self.load_data():
            return {}
            
        charts = {}
        
        try:
            charts['cpu'] = self.create_cpu_chart()
            print(f"✓ CPU chart saved: {charts['cpu']}")
        except Exception as e:
            print(f"✗ Error creating CPU chart: {e}")
            
        try:
            charts['memory'] = self.create_memory_chart()
            print(f"✓ Memory chart saved: {charts['memory']}")
        except Exception as e:
            print(f"✗ Error creating memory chart: {e}")
            
        try:
            temp_chart = self.create_temperature_chart()
            if temp_chart:
                charts['temperature'] = temp_chart
                print(f"✓ Temperature chart saved: {charts['temperature']}")
        except Exception as e:
            print(f"✗ Error creating temperature chart: {e}")
            
        try:
            charts['dashboard'] = self.create_dashboard()
            print(f"✓ Dashboard saved: {charts['dashboard']}")
        except Exception as e:
            print(f"✗ Error creating dashboard: {e}")
            
        return charts


def convert_json_to_images(json_file: str, output_dir: str = "resource_charts") -> Dict[str, str]:
    """
    Convenience function to convert resource monitoring JSON to images.
    
    Args:
        json_file: Path to the resource monitor JSON file
        output_dir: Directory to save generated charts
        
    Returns:
        Dictionary mapping chart types to file paths
    """
    visualizer = ResourceVisualizer(json_file, output_dir)
    return visualizer.create_all_charts()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python resource_visualizer.py <json_file> [output_dir]")
        sys.exit(1)
        
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "resource_charts"
    
    charts = convert_json_to_images(json_file, output_dir)
    
    print(f"\nGenerated {len(charts)} charts:")
    for chart_type, path in charts.items():
        print(f"  {chart_type}: {path}")
