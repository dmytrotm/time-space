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
DARK_GRAY = (40, 40, 40)
YELLOW = (0, 255, 255)
LIGHTER_GRAY = (100, 100, 100)
ORANGE = (0, 100, 255)

# Text Settings
LOADING_TEXT = "Loading... Please wait."
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICKNESS = 2
TITLE_FONT_SCALE = 0.8

# Configuration File Paths
CONFIG_ROI_Z1_PATH = "configs/rois_z1.json"
CONFIG_ROI_Z2_PATH = "configs/rois_z2.json"
CONFIG_POSITIONS_PATH = "configs/positions.json"
WORKSPACE_EXTRACTOR_CONFIG = "configs/custom_markers.yaml"

# Inspection Window Constants
NAV_BUTTON_WIDTH = 80
NAV_TEXT_OFFSET_Y = 30
NAV_ZONE_INFO_OFFSET_Y = 10

# Image Scaling
INTERPOLATION_METHOD = cv2.INTER_LINEAR

# UI Layout Constants
CONTENT_WIDTH = 600
CONTENT_HEIGHT = 400
TITLE_OFFSET_Y = 60
LINE_HEIGHT = 80
BUTTON_HEIGHT = 50
BUTTON_WIDTH = 500
BUTTON_OFFSET_Y = 10
TITLE_LINE_OFFSET = 30
BUTTON_START_OFFSET = 120
CONTENT_PADDING = 50
DIVIDER_WIDTH = 250

# Other Constants
WAIT_KEY_TIMEOUT = 50
KEY_MASK = 0xFF
INVALID_KEY = 255