# Time-Space Analysis

This project is a computer vision application for analyzing images of a workspace to detect the presence of grounding wires, tape, and labels within predefined regions of interest (ROIs). It also estimates location and size of the tapes.

## Prerequisites

* Python **3.10+**

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dmytrotm/time-space.git
   ```
2. (Optional, yet recommended) Create and activate virtual environment:
   1) Creating venv:
      Windows
      ```bash
      python -m venv .venv
      ```

      Linux and MacOS
      ```bash
      python3 -m venv .venv
      ```

      Or create it with uv (if available)
      ```bash
      uv venv
      ```
   
   2) Activating venv:
      Windows
      ```bash
      venv\Scripts\activate
      ```

      Linux and MacOS
      ```bash
      source .venv/bin/activate
      ```

2. Install the required dependencies:
   Windows
   ```bash
   pip install -r requirements.txt
   ```

   Linux and MacOS
   ```bash
   pip3 install -r requirements.txt
   ```

   Or install with uv (if available)
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

To run the analysis, execute the `main.py` script:

Windows
```bash
python main.py
```

Linux and MacOS
```bash
python3 main.py
```

The script will:
1. Load images from the `dataset` directory (unpack shared archive in this folder).
2. Extract the workspace from the images.
3. Crop the ROIs based on the configured coordinates.
4. Perform the following detections:
   - Presence of grounding wires.
   - Presence of tape and labels.
   - Deviation of the tape from its expected position.
5. Print the results to the console (for now).

## Configuration

The project can be configured through the files in the `configs` directory:

- `rois_z1.json` and `rois_z2.json`: Define the coordinates of the ROIs for two different zones.
- `positions.json`: Defines the expected positions of the tape for deviation calculation.
- `custom_markers.yaml`: Defines the custom markers used for workspace extraction.

## Dependencies

The project uses the following major libraries:

- [OpenCV](https://opencv.org/) for image processing.
- [PyTorch](https://pytorch.org/) for deep learning.
- [Ultralytics](https://ultralytics.com/) for YOLO object detection.
- [NumPy](https://numpy.org/) for numerical operations.
- [Pandas](https://pandas.pydata.org/) for data manipulation.

For a complete list of dependencies, see the `requirements.txt` file.
