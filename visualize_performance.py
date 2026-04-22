#!/usr/bin/env python3
"""
Script to convert performance profiling JSON data to visual charts.
Usage: python visualize_performance.py [json_file] [output_dir]
"""

import sys
import os
from utils.performance_visualizer import convert_performance_json_to_images


def main():
    # Default JSON file
    default_json = "performance_profile.json"
    default_output = "performance_charts"
    
    # Parse arguments
    json_file = sys.argv[1] if len(sys.argv) > 1 else default_json
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        print(f"Please run the application first to generate performance data, or specify a different file.")
        print(f"Usage: python {sys.argv[0]} [json_file] [output_dir]")
        sys.exit(1)
    
    print(f"Converting {json_file} to performance charts...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Generate charts
    charts = convert_performance_json_to_images(json_file, output_dir)
    
    if charts:
        print(f"\n✓ Successfully generated {len(charts)} performance charts:")
        for chart_type, path in charts.items():
            print(f"  • {chart_type.capitalize()}: {path}")
        
        print(f"\nAll performance charts saved to: {output_dir}/")
        print("These charts show operation-level performance metrics for your detectors and processors.")
    else:
        print("✗ No charts were generated. Please check the JSON file format.")


if __name__ == "__main__":
    main()
