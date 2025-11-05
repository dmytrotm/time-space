import cv2

# Logging Configuration
LOG_SEPARATOR = "=" * 60
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480

RELATIVE_HALF_SIZE_RASNET = 0.052
TAPE_CLASS_ID = 1
LABEL_CLASS_ID = 0
TAPE_DEVIATION_TOO_FAR = -1
TAPE_DEVIATION_WRONG_LENGTH = 1
TAPE_DETECTOR_CONF_THRESHOLD = 0.8

KEY_MAPPING = {
    13: "start",  # enter
    3: "next",  # right arrow
    2: "previous",  # left arrow
    32: "finish",  # space
    27: "exit",  # esc
}

# BGR colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
GRAY = (80, 80, 80)
YELLOW = (0, 255, 255)

# Text Settings
LOADING_TEXT = "Loading... Please wait."
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICKNESS = 2

# Configuration File Paths
CONFIG_ROI_Z1_PATH = "configs/rois_z1.json"
CONFIG_ROI_Z2_PATH = "configs/rois_z2.json"
CONFIG_POSITIONS_PATH = "configs/positions.json"
WORKSPACE_EXTRACTOR_CONFIG = "configs/custom_markers.yaml"

# Inspection Window Constants
NAV_ZONE_INFO_OFFSET_Y = 10

# Image Scaling
INTERPOLATION_METHOD = cv2.INTER_LINEAR

# UI Layout Constants
LINE_HEIGHT = 80
BUTTON_HEIGHT = 100
BUTTON_WIDTH = 500
BUTTON_SPACING = 20
