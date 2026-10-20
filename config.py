"""
Configuration settings for AI-Based Accident Detection System.
Provides customizable thresholds, model parameters, and UI constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# YOLO Model Configuration
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Target Vehicle Classes in COCO Dataset:
# Class 2: car
# Class 3: motorcycle
# Class 5: bus
# Class 7: truck
TARGET_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Default Detection & Collision Thresholds
DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU_THRESHOLD = 0.18           # Spatial overlap ratio indicating physical contact
DEFAULT_PERSISTENCE_FRAMES = 4        # Consecutive frames of collision overlap required for confirmation
DEFAULT_SPEED_DROP_PERCENT = 40.0     # Sudden deceleration threshold (%)
DEFAULT_PROXIMITY_MARGIN = 20         # Pixel margin for near-miss proximity warning

# Color Palette for Annotations (BGR for OpenCV)
CLASS_COLORS = {
    "car": (255, 178, 50),         # Sky Blue
    "motorcycle": (50, 205, 50),   # Lime Green
    "bus": (255, 140, 0),          # Amber / Orange
    "truck": (205, 92, 92),        # Purple / Indian Red
}
COLLISION_BOX_COLOR = (0, 0, 255)       # Red for collision bounding box
WARNING_BOX_COLOR = (0, 165, 255)       # Orange for near-miss

# Supported Video Extensions
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]

# Application Metadata
APP_TITLE = "AI-Based Accident Detection System"
APP_SUBTITLE = "Computer Vision & Deep Learning Road Traffic Safety Prototype"
APP_VERSION = "1.0.0"
ACADEMIC_LEVEL = "Second-Year Engineering Mini Project"
