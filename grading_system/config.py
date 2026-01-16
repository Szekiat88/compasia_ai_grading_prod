# config.py
class Config:
    # Default device dimensions if not provided
    DEFAULT_DEVICE_WIDTH_MM = 70
    DEFAULT_DEVICE_HEIGHT_MM = 140

    # Size thresholds in mm
    DENT_SMALL_THRESHOLD = 0.5
    DENT_MEDIUM_THRESHOLD = 2.0
    DENT_LARGE_THRESHOLD = 3.0

    # Model input size
    MODEL_INPUT_SIZE = 960