# convert_yolo.py
from typing import Dict, Any, Optional
from pathlib import Path
from ultralytics import YOLO


def yolo_result_to_dict(result) -> Optional[Dict[str, Any]]:
    """
    Convert a YOLO result to a dictionary format suitable for storage/transmission

    Args:
        result: YOLO prediction result object

    Returns:
        Dictionary containing essential detection information
    """
    if result is None:
        return None

    # Convert boxes to list of dicts
    boxes = []
    for box in result.boxes:
        boxes.append({
            'cls': int(box.cls[0].item()),  # Convert tensor to int
            'conf': float(box.conf[0].item()),  # Convert tensor to float
            'xywh': [float(x) for x in box.xywh[0].tolist()]  # Convert tensor to list of floats
        })

    return {
        'boxes': boxes,
        'orig_shape': result.orig_shape,
        'names': result.names
    }


def process_images_to_dict(
        model_path: str,
        image_paths: Dict[str, str],
        conf: float = 0.2,
        imgsz: int = 960
) -> Dict[str, Any]:
    """
    Process images through YOLO model and return dictionary format results

    Args:
        model_path: Path to YOLO model file
        image_paths: Dictionary mapping positions to image file paths
        conf: Confidence threshold for detections
        imgsz: Image size for processing

    Returns:
        Dictionary mapping positions to detection results in storable format
    """
    # Load model
    model = YOLO(model_path)

    # Process each image
    detection_results = {}
    for position, img_path in image_paths.items():
        if not Path(img_path).exists():
            print(f"Warning: Image not found at {img_path}")
            detection_results[position] = None
            continue

        results = model.predict(img_path, conf=conf, imgsz=imgsz)
        if len(results) > 0:
            detection_results[position] = yolo_result_to_dict(results[0])
        else:
            detection_results[position] = None

    return detection_results


# # Example usage
# if __name__ == "__main__":
#     # Model and image paths
#     MODEL_PATH = "/Users/sabainaharoon/Documents/compasia/ai_grading/defect_detection/training/runs/detect/yolov8_960/best_5Dec_960.pt"
#     IMAGE_PATHS = {
#         'front': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_front_1734574854771.png",
#         'back': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_back_1734575013796.png",
#         'bottom': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_bottom_1734574940230.png",
#         'right': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_right_1734574994050.png",
#         'top': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_top_1734574874469.png",
#         'left': "test_images/357290099511186_heavyscratches/IMEI_357290099511186_left_1734574932293.png"
#     }

#     # Process images
#     detection_results = process_images_to_dict(
#         model_path=MODEL_PATH,
#         image_paths=IMAGE_PATHS,
#         conf=0.2,
#         imgsz=960
#     )

#     # Can now use these results with get_ai_grade
#     from get_ai_grade import get_ai_grade

#     grading_results = get_ai_grade(
#         detection_results=detection_results,
#         device_model=None,
#         dimensions_csv=None
#     )

#     # Print results
#     print("\nFinal Grading Results:")
#     print("=" * 50)
#     print(f"Screen Grade: {grading_results['screen']['grade']}")
#     print("\nHousing Grades:")
#     for section, grade in grading_results['housing']['section_grades'].items():
#         print(f"- {section.capitalize()}: {grade}")
#     print(f"\nOverall Housing Grade: {grading_results['housing']['overall_grade']}")

#     # Example of what the dictionary format looks like for one position
#     if detection_results['front']:
#         print("\nExample Detection Result Structure:")
#         print("Number of boxes:", len(detection_results['front']['boxes']))
#         if detection_results['front']['boxes']:
#             print("First detection:", detection_results['front']['boxes'][0])