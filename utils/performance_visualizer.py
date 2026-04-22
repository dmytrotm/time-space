import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import os


class PerformanceVisualizer:
    """Visualize operation-level performance profiling data."""
    
    def __init__(self, json_file: str, output_dir: str = "performance_charts"):
        """
        Initialize the performance visualizer.
        
        Args:
            json_file: Path to the performance profile JSON file
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
        """Load performance profiling data from JSON file."""
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
            
    def create_operation_duration_chart(self, save_path: Optional[str] = None) -> str:
        """Create chart showing operation durations over time."""
        if not self.data or not self.data.get('operations'):
            raise ValueError("No performance data available")
            
        operations = self.data['operations']
        
        # Group by operation name
        op_groups = {}
        for op in operations:
            name = op['operation_name']
            if name not in op_groups:
                op_groups[name] = []
            op_groups[name].append(op)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot each operation type
        colors = plt.cm.tab10(np.linspace(0, 1, len(op_groups)))
        
        for i, (op_name, ops) in enumerate(op_groups.items()):
            timestamps = [datetime.fromisoformat(op['timestamp']) for op in ops]
            durations = [op['duration_ms'] for op in ops]
            
            ax.scatter(timestamps, durations, label=op_name, alpha=0.7, 
                      color=colors[i], s=50)
            
            # Add trend line if enough data points
            if len(durations) > 1:
                z = np.polyfit(range(len(timestamps)), durations, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(range(len(timestamps))), 
                       color=colors[i], linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Duration (ms)')
        ax.set_title('Operation Duration Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'operation_durations.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_operation_summary_chart(self, save_path: Optional[str] = None) -> str:
        """Create summary chart showing average operation performance."""
        if not self.data or not self.data.get('summary'):
            raise ValueError("No performance summary data available")
            
        summary = self.data['summary']
        
        # Prepare data for plotting
        op_names = list(summary.keys())
        avg_durations = [summary[op]['duration_ms']['avg'] for op in op_names]
        max_durations = [summary[op]['duration_ms']['max'] for op in op_names]
        execution_counts = [summary[op]['total_executions'] for op in op_names]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Operation Performance Summary', fontsize=16, fontweight='bold')
        
        # Average duration bar chart
        bars1 = ax1.bar(op_names, avg_durations, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Average Duration (ms)')
        ax1.set_title('Average Operation Duration')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_durations):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}ms', ha='center', va='bottom')
        
        # Max duration bar chart
        bars2 = ax2.bar(op_names, max_durations, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Max Duration (ms)')
        ax2.set_title('Maximum Operation Duration')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, max_durations):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}ms', ha='center', va='bottom')
        
        # Execution count pie chart
        ax3.pie(execution_counts, labels=op_names, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Execution Count Distribution')
        
        # Memory usage chart
        avg_memory = [summary[op]['memory_delta_mb']['avg'] for op in op_names]
        bars4 = ax4.bar(op_names, avg_memory, color='lightgreen', alpha=0.7)
        ax4.set_ylabel('Average Memory Delta (MB)')
        ax4.set_title('Average Memory Usage Change')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars4, avg_memory):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}MB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'operation_summary.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_slowest_operations_chart(self, save_path: Optional[str] = None, limit: int = 10) -> str:
        """Create chart showing the slowest operations."""
        if not self.data or not self.data.get('operations'):
            raise ValueError("No performance data available")
            
        operations = self.data['operations']
        
        # Sort by duration and get top N
        sorted_ops = sorted(operations, key=lambda x: x['duration_ms'], reverse=True)[:limit]
        
        op_names = [f"{op['operation_name']}\n({op['timestamp'].split('T')[1][:8]})" 
                   for op in sorted_ops]
        durations = [op['duration_ms'] for op in sorted_ops]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(op_names, durations, color='orangered', alpha=0.7)
        ax.set_xlabel('Duration (ms)')
        ax.set_title(f'Top {limit} Slowest Operations')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, durations):
            ax.text(bar.get_width() + max(durations) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}ms', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'slowest_operations.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_memory_usage_chart(self, save_path: Optional[str] = None) -> str:
        """Create chart showing memory usage patterns."""
        if not self.data or not self.data.get('operations'):
            raise ValueError("No performance data available")
            
        operations = self.data['operations']
        
        # Group by operation name
        op_groups = {}
        for op in operations:
            name = op['operation_name']
            if name not in op_groups:
                op_groups[name] = []
            op_groups[name].append(op)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Memory delta over time
        colors = plt.cm.tab10(np.linspace(0, 1, len(op_groups)))
        
        for i, (op_name, ops) in enumerate(op_groups.items()):
            timestamps = [datetime.fromisoformat(op['timestamp']) for op in ops]
            memory_deltas = [op['memory_after_mb'] - op['memory_before_mb'] for op in ops]
            
            ax1.plot(timestamps, memory_deltas, label=op_name, alpha=0.7, 
                    color=colors[i], linewidth=2, marker='o', markersize=3)
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory Delta (MB)')
        ax1.set_title('Memory Usage Change Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Peak memory usage
        for i, (op_name, ops) in enumerate(op_groups.items()):
            timestamps = [datetime.fromisoformat(op['timestamp']) for op in ops]
            peak_memory = [op['memory_peak_mb'] for op in ops]
            
            ax2.plot(timestamps, peak_memory, label=op_name, alpha=0.7, 
                    color=colors[i], linewidth=2, marker='s', markersize=3)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Peak Memory Usage Over Time')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'memory_usage.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
        
    def create_all_charts(self) -> Dict[str, str]:
        """Generate all available performance charts and return paths."""
        if not self.load_data():
            return {}
            
        charts = {}
        
        try:
            charts['durations'] = self.create_operation_duration_chart()
            print(f"✓ Operation duration chart saved: {charts['durations']}")
        except Exception as e:
            print(f"✗ Error creating duration chart: {e}")
            
        try:
            charts['summary'] = self.create_operation_summary_chart()
            print(f"✓ Operation summary chart saved: {charts['summary']}")
        except Exception as e:
            print(f"✗ Error creating summary chart: {e}")
            
        try:
            charts['slowest'] = self.create_slowest_operations_chart()
            print(f"✓ Slowest operations chart saved: {charts['slowest']}")
        except Exception as e:
            print(f"✗ Error creating slowest operations chart: {e}")
            
        try:
            charts['memory'] = self.create_memory_usage_chart()
            print(f"✓ Memory usage chart saved: {charts['memory']}")
        except Exception as e:
            print(f"✗ Error creating memory usage chart: {e}")
            
        return charts


def convert_performance_json_to_images(json_file: str, output_dir: str = "performance_charts") -> Dict[str, str]:
    """
    Convenience function to convert performance profiling JSON to images.
    
    Args:
        json_file: Path to the performance profile JSON file
        output_dir: Directory to save generated charts
        
    Returns:
        Dictionary mapping chart types to file paths
    """
    visualizer = PerformanceVisualizer(json_file, output_dir)
    return visualizer.create_all_charts()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python performance_visualizer.py <json_file> [output_dir]")
        sys.exit(1)
        
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "performance_charts"
    
    charts = convert_performance_json_to_images(json_file, output_dir)
    
    print(f"\nGenerated {len(charts)} performance charts:")
    for chart_type, path in charts.items():
        print(f"  {chart_type}: {path}")
