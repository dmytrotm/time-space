#!/usr/bin/env python3
"""
Script to convert resource monitoring JSON data to visual charts.
Usage: python visualize_resources.py [json_file] [output_dir]
"""

import sys
import os
from utils.resource_visualizer import convert_json_to_images


def main():
    # Default JSON file
    default_json = "resource_monitor.json"
    default_output = "resource_charts"
    
    # Parse arguments
    json_file = sys.argv[1] if len(sys.argv) > 1 else default_json
    output_dir = sys.argv[2] if len(sys.argv) > 2 else default_output
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        print(f"Please run the application first to generate resource data, or specify a different file.")
        print(f"Usage: python {sys.argv[0]} [json_file] [output_dir]")
        sys.exit(1)
    
    print(f"Converting {json_file} to charts...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Generate charts
    charts = convert_json_to_images(json_file, output_dir)
    
    if charts:
        print(f"\n✓ Successfully generated {len(charts)} charts:")
        for chart_type, path in charts.items():
            print(f"  • {chart_type.capitalize()}: {path}")
        
        print(f"\nAll charts saved to: {output_dir}/")
        print("You can open these image files to view the resource monitoring visualizations.")
    else:
        print("✗ No charts were generated. Please check the JSON file format.")


if __name__ == "__main__":
    main()
