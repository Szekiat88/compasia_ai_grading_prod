from pathlib import Path
import torch
from ultralytics import YOLO
from typing import Dict, Optional, Any
import sys
import logging
import cv2
import numpy as np
import sys
import os

# Add the project directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

from core.enums import Grade, DevicePosition
from grading.screen import ScreenGrading
from grading.housing_grading import HousingGrading
from grading.housing_processor import HousingProcessor
from utils.device_dimensions import DeviceDimensions


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_model(model_path: str) -> YOLO:
    """Load YOLOv8 model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return YOLO(model_path)


def validate_images(image_paths: Dict[DevicePosition, str]):
    """Validate that all required images exist"""
    for position, path in image_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found for {position.value}: {path}")


def process_images(model, image_paths: Dict[DevicePosition, str], target_size=(960, 960)) -> Dict[DevicePosition, Any]:
    """Process all device images through YOLOv8 model"""
    results = {}
    for position, img_path in image_paths.items():
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or invalid: {img_path}")

        # if position == DevicePosition.DEVICE_FRONT_INACTIVE:
            # detection_results = model.predict(image, conf=0.34, iou=0.3, imgsz=960)
        # else:
            # detection_results = model.predict(image, conf=0.34, iou=0.3, imgsz=960)

        detection_results = model.predict(image, conf=0.34, iou=0.3, imgsz=960)
        if len(detection_results) > 0:
            results[position] = detection_results[0]
        else:
            results[position] = None

    return results


def grade_device(
        model_path: str,
        image_paths: Dict[DevicePosition, str],
        device_model: Optional[str] = None,
        dimensions_csv: Optional[str] = None
) -> dict:
    """Grade a device using all its images"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate inputs
        validate_images(image_paths)

        # Load model and process images
        logger.info("Loading YOLOv8 model...")
        model = load_model(model_path)

        logger.info("Processing images through model...")
        detection_results = process_images(model, image_paths)

        # Load device dimensions
        logger.info(f"Loading device dimensions for model: {device_model or 'default'}")
        dimensions = DeviceDimensions(dimensions_csv)
        height_mm, width_mm, thickness_mm = dimensions.get_dimensions(device_model)
        logger.info(f"Device dimensions: {width_mm}x{height_mm}x{thickness_mm}mm")

        # Initialize components
        screen_grader = ScreenGrading(height_mm, width_mm, thickness_mm)
        housing_processor = HousingProcessor(height_mm, width_mm, thickness_mm)
        housing_grader = HousingGrading()

        # Process screen
        logger.info("Grading screen...")
        screen_result = screen_grader.calculate_grade(
            detection_results[DevicePosition.DEVICE_FRONT_INACTIVE]
        )
        logger.info(f"Screen Grade: {screen_result.grade.value}")

        # Process housing sections
        logger.info("Processing housing sections...")
        processed_sections = {}

        # Process back
        logger.info("Processing back...")
        processed_sections['back'] = housing_processor.process_back(
            detection_results[DevicePosition.DEVICE_BACK]
        )

        # Process sides
        logger.info("Processing sides...")
        processed_sections['sides'] = housing_processor.process_sides([
            detection_results[DevicePosition.DEVICE_LEFT],
            detection_results[DevicePosition.DEVICE_RIGHT]
        ])

        # Process top
        logger.info("Processing top...")
        processed_sections['top'] = housing_processor.process_top(
            detection_results[DevicePosition.DEVICE_TOP]
        )

        # Process bottom
        logger.info("Processing bottom...")
        processed_sections['bottom'] = housing_processor.process_bottom(
            detection_results[DevicePosition.DEVICE_BOTTOM]
        )

        # Calculate overall housing grade
        logger.info("Calculating housing grade...")
        housing_result = housing_grader.calculate_grade(processed_sections)

        # Prepare results
        results = {
            'screen': {
                'grade': screen_result.grade.value,
                'defects': screen_result.defect_count,
                'details': screen_result.details
            },
            'housing': {
                'overall_grade': housing_result.grade.value,
                'defect_count': housing_result.defect_count,
                'details': housing_result.details
            },
            'device_info': {
                'model': device_model,
                'dimensions': {
                    'width': width_mm,
                    'height': height_mm,
                    'thickness': thickness_mm
                }
            }
        }

        return results

    except Exception as e:
        logger.error(f"Error during grading: {str(e)}")
        raise

# Function to fetch image paths for a given session ID
def get_image_paths(session_id, base_dir):
    # Path to the session's folder
    session_dir = os.path.join(base_dir, session_id)
    # Dictionary to store the image paths
    image_paths = {}
    # Check if the session folder exists
    if not os.path.exists(session_dir):
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    # Map filenames to device positions
    for file_name in os.listdir(session_dir):
        if "INPUT_FRONT" in file_name:
            image_paths[DevicePosition.DEVICE_FRONT_INACTIVE] = os.path.join(session_dir, file_name)
        elif "INPUT_BACK" in file_name:
            image_paths[DevicePosition.DEVICE_BACK] = os.path.join(session_dir, file_name)
        elif "INPUT_BOTTOM" in file_name:
            image_paths[DevicePosition.DEVICE_BOTTOM] = os.path.join(session_dir, file_name)
        elif "INPUT_LEFT" in file_name:
            image_paths[DevicePosition.DEVICE_LEFT] = os.path.join(session_dir, file_name)
        elif "INPUT_RIGHT" in file_name:
            image_paths[DevicePosition.DEVICE_RIGHT] = os.path.join(session_dir, file_name)
        elif "INPUT_TOP" in file_name:
            image_paths[DevicePosition.DEVICE_TOP] = os.path.join(session_dir, file_name)
    return image_paths

def main():
    """Main entry point"""
    MODEL_PATH = "/Users/sabainaharoon/Documents/compasia/ai_grading/defect_detection/training/runs/detect/yolov8_960/best_5Dec_960.pt"
    session_id = "4fb94a16-fb78-4b9e-9e89-e5374ac949fa"
    session_id = "320b3c33-b06a-480c-b50b-42d53bd73baf"
    base_dir = "/Users/sabainaharoon/Documents/compasia/ai_grading/grading_system/testing_grading_accuracy/organized_cleaned_images"
    # Example usage
    session_id = "e7282c59-ee53-492f-bb9d-3b4b1cf199a9"
    # session_id = "76a7cd2e-13a3-4020-9133-1a9f20cc4976"
    IMAGE_PATHS = get_image_paths(session_id, base_dir)

    try:
        results = grade_device(
            MODEL_PATH,
            IMAGE_PATHS,
            device_model=None,
            dimensions_csv=None
        )

        # Print final results
        print("\nFinal Grading Results:")
        print("=" * 50)
        print(f"Device Model: {results['device_info']['model'] or 'Unknown'}")
        print(f"Dimensions: {results['device_info']['dimensions']}mm")
        print(f"\nScreen Grade: {results['screen']['grade']}")
        print(f"Screen Details: {results['screen']['details']}")
        print(f"\nHousing Grade: {results['housing']['overall_grade']}")
        print(f"Housing Details: {results['housing']['details']}")

    except Exception as e:
        print(f"Error during grading: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()