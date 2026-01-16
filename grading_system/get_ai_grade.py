# get_ai_grade.py
from dataclasses import dataclass
import os
import torch

import logging

import ast
from typing import Dict, Optional, Any, List, Union, Tuple
from typing import Union
# Add the project directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

from .core.enums import Grade, DevicePosition, DeviceArea
from .grading.screen import ScreenGrading
from .grading.housing_grading import HousingGrading
from .grading.housing_processor import HousingProcessor
from .utils.device_dimensions import DeviceDimensions
from .grading.housing_grading_ML import HousingGradePredictor


@dataclass
class MockBox:
    """Box class for YOLO detections"""
    cls: Union[torch.Tensor, int, List[int]]
    conf: Union[torch.Tensor, float, List[float]]
    xywh: Union[torch.Tensor, List[float], Tuple[float, float, float, float]]

    def __post_init__(self):
        # Convert to tensors if needed
        if not isinstance(self.cls, torch.Tensor):
            self.cls = torch.tensor([self.cls if isinstance(self.cls, int) else self.cls[0]])
        if not isinstance(self.conf, torch.Tensor):
            self.conf = torch.tensor([self.conf if isinstance(self.conf, float) else self.conf[0]])
        if not isinstance(self.xywh, torch.Tensor):
            self.xywh = torch.tensor([self.xywh])


@dataclass
class YOLODetectionResult:
    """Clean detection result format"""
    boxes: List[MockBox]
    orig_shape: Tuple[int, int]
    names: Dict[int, str]


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )




def convert_detection_results(raw_results: Union[str, Dict[str, Any]]) -> Dict[
    DevicePosition, Optional[YOLODetectionResult]]:
    """
    Convert raw detection results into internal format with proper enums.

    Args:
        raw_results: Either a Python dict or a string representation of a Python dict
    """
    # Parse input if it's a string
    if isinstance(raw_results, str):
        try:
            # Use ast.literal_eval for safely evaluating Python literals
            raw_results = ast.literal_eval(raw_results)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input string format: {e}")

    position_mapping = {
        'front': DevicePosition.DEVICE_FRONT_INACTIVE,
        'back': DevicePosition.DEVICE_BACK,
        'left': DevicePosition.DEVICE_LEFT,
        'right': DevicePosition.DEVICE_RIGHT,
        'top': DevicePosition.DEVICE_TOP,
        'bottom': DevicePosition.DEVICE_BOTTOM
    }

    converted_results = {}

    for pos_str, result_list in raw_results.items():
        if pos_str.lower() not in position_mapping:
            raise ValueError(f"Unknown position: {pos_str}")

        position = position_mapping[pos_str.lower()]

        if not result_list:  # Handle empty or None cases
            converted_results[position] = None
            continue

        # Take first result if it's a list
        result = result_list[0] if isinstance(result_list, list) else result_list

        # Convert names dictionary - ensure keys are integers
        names_dict = {
            int(k): v for k, v in result['names'].items()
        }

        # Create boxes
        boxes = []
        for box_data in result['boxes']:
            box = MockBox(
                cls=box_data['cls'],
                conf=box_data['conf'],
                xywh=box_data['xywh']
            )
            boxes.append(box)

        # Ensure orig_shape is a tuple
        orig_shape = tuple(result['orig_shape'])

        # Create YOLODetectionResult
        converted_results[position] = YOLODetectionResult(
            boxes=boxes,
            orig_shape=orig_shape,
            names=names_dict
        )

    return converted_results


def serialize_area_defects_with_counts(area_defects: Dict) -> dict:
    return {
        area.name: {
            "micro_scratches": {
                "count": len(defects.micro_scratches),
                "sizes": list(map(float, defects.micro_scratches))
            },
            "minor_scratches": {
                "count": len(defects.minor_scratches),
                "sizes": list(map(float, defects.minor_scratches))
            },
            "major_scratches": {
                "count": len(defects.major_scratches),
                "sizes": list(map(float, defects.major_scratches))
            }
        }
        for area, defects in area_defects.items()
    }

def serialize_flat_defects(area_defects: Dict) -> dict:
    all_micro = []
    all_minor = []
    all_major = []

    for defects in area_defects.values():
        all_micro.extend(defects.micro_scratches)
        all_minor.extend(defects.minor_scratches)
        all_major.extend(defects.major_scratches)

    return {
        "micro_scratches": {
            "count": len(all_micro),
            "sizes": list(map(float, all_micro))
        },
        "minor_scratches": {
            "count": len(all_minor),
            "sizes": list(map(float, all_minor))
        },
        "major_scratches": {
            "count": len(all_major),
            "sizes": list(map(float, all_major))
        }
    }


def get_ai_grade(
        detection_results: Union[str, Dict[str, Any]],
        use_ml_housing: bool = True,
        device_model: Optional[str] = None,
        dimensions_csv: Optional[str] = None
) -> dict:
    """getde
    Calculate device grade from detection results

    Args:
        detection_results: Either a JSON string or dictionary mapping positions to detections
            Format can be either:
            - Dictionary mapping positions to detection results
            - JSON string containing the same structure
        device_model: Optional device model name for dimensions
        dimensions_csv: Optional path to device dimensions CSV

    Returns:
        Dictionary containing comprehensive grading results
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Convert detection results to internal format
        logger.info("Processing detection results...")
        try:
            converted_results = convert_detection_results(detection_results)
        except ValueError as e:
            logger.error(f"Error parsing detection results: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing detection results: {e}")
            raise

        # Load device dimensions
        logger.info(f"Loading device dimensions for model: {device_model or 'default'}")
        dimensions = DeviceDimensions(dimensions_csv)
        height_mm, width_mm, thickness_mm = dimensions.get_dimensions(device_model)
        logger.info(f"Device dimensions: {width_mm}x{height_mm}x{thickness_mm}mm")

        # Initialize grading components
        screen_grader = ScreenGrading(height_mm, width_mm, thickness_mm)
        housing_processor = HousingProcessor(height_mm, width_mm, thickness_mm)
        housing_grader = HousingGrading()

        # Process screen
        logger.info("Grading screen...")
        screen_result = screen_grader.calculate_grade(
            converted_results[DevicePosition.DEVICE_FRONT_INACTIVE]
        )
        logger.info(f"Screen Grade: {screen_result.grade.value}")

        # Check for R2 conditions in back side detections
        if DevicePosition.DEVICE_BACK in converted_results and converted_results[
            DevicePosition.DEVICE_BACK] is not None:
            back_result = converted_results[DevicePosition.DEVICE_BACK]

            # Check for hairline_crack or dense_crack in the back
            for box in back_result.boxes:
                class_id = int(box.cls)
                class_name = back_result.names.get(class_id, "unknown")

                if "hairline_crack" in class_name.lower() or "dense_crack" in class_name.lower() or "housing_cracked" in class_name.lower():
                    logger.info(f"R2 condition detected: {class_name} found on back side")

                    logger.info("Housing Grade: R2 (immediate rejection)")

                    # Skip further housing grading
                    return {

                        # Direct grade access for backwards compatibility
            "ai_screen": screen_result.grade.value,
            "ai_housing": "R2",
            "details": {
                "screen": {
                    "grade": screen_result.grade.value,
                    "assessment": screen_result.details,
                    'defects':serialize_flat_defects(screen_result.defect_count.area_defects)
                    # "defects": screen_result.defect_count.__dict__
                }},


                        'screen': {
                            'grade': screen_result.grade.value,
                            'defects': screen_result.defect_count,
                            'details': screen_result.details
                        },
                        'housing': {
                            'overall_grade': "R2",
                            'defect_count': None,
                            'details': "Cracked back"
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

        if use_ml_housing:
            # Use the new ML-based housing grader
            logger.info("Using ML-based housing grader...")
            housing_grader = HousingGradePredictor()
            housing_grade, housing_confidence, cs_count = housing_grader.predict({k: v for k, v in converted_results.items() if k != DevicePosition.DEVICE_FRONT_INACTIVE})

            # Create a housing result dictionary in the same format as expected by the rest of the code
            housing_output = {
                'overall_grade': housing_grade,  # Convert string grade to Grade enum
                'cover_spot_count': cs_count,
                'details': {
                    'confidence': housing_confidence
                }
            }

            logger.info(f"Housing Grade (ML): {housing_grade} with confidence {housing_confidence:.4f}")

        else:
            # Process housing sections
            logger.info("Processing housing sections...")
            processed_sections = {}

            # Process back
            logger.info("Processing back...")
            processed_sections['back'] = housing_processor.process_back(
                converted_results[DevicePosition.DEVICE_BACK]
            )

            # Process sides
            logger.info("Processing sides...")
            processed_sections['sides'] = housing_processor.process_sides([
                converted_results[DevicePosition.DEVICE_LEFT],
                converted_results[DevicePosition.DEVICE_RIGHT]
            ])

            # Process top
            logger.info("Processing top...")
            processed_sections['top'] = housing_processor.process_top(
                converted_results[DevicePosition.DEVICE_TOP]
            )

            # Process bottom
            logger.info("Processing bottom...")
            processed_sections['bottom'] = housing_processor.process_bottom(
                converted_results[DevicePosition.DEVICE_BOTTOM]
            )

            # Calculate overall housing grade
            logger.info("Calculating housing grade...")
            housing_result = housing_grader.calculate_grade(processed_sections)
            housing_output = {
                'overall_grade': housing_result.grade.value,
                'defect_count': housing_result.defect_count,
                'details': housing_result.details,
                "cover_spot_count": -100
            }

        # Restructure results to maintain backwards compatibility
        results = {
            # Direct grade access for backwards compatibility
            "ai_screen": screen_result.grade.value,
            "ai_housing": housing_output['overall_grade'],
            "cover_spot_count": housing_output["cover_spot_count"],

            # Detailed information in separate section
            "details": {
                "screen": {
                    "grade": screen_result.grade.value,
                    "assessment": screen_result.details,
                    'defects':serialize_flat_defects(screen_result.defect_count.area_defects)
                    # "defects": screen_result.defect_count.__dict__
                },
                'housing': housing_output,
                "device": {
                    "model": device_model,
                    "dimensions": {
                        "width": width_mm,
                        "height": height_mm,
                        "thickness": thickness_mm
                    }
                }
            }
        }
        logger.info(f"Grading results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error during grading: {str(e)}")
        raise


# # Example usage
# if __name__ == "__main__":
#     # Example with dictionary format
#     # detection_results = '''{'top': [{'boxes': [{'cls': 4, 'conf': 0.37433338165283203, 'xywh': [269.129150390625, 330.0013427734375, 12.7384033203125, 13.52392578125]}],
#     #                               'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                               'orig_shape': [800, 448]}],
#     #                      'right': [{'boxes': [{'cls': 9, 'conf': 0.5158535838127136, 'xywh': [219.66445922851562, 123.40404510498047, 15.11810302734375, 16.067276000976562]},
#     #                                           {'cls': 0, 'conf': 0.4720025062561035, 'xywh': [198.99729919433594, 701.10400390625, 21.552703857421875, 87.138427734375]},
#     #                                           {'cls': 7, 'conf': 0.436643123626709, 'xywh': [204.28311157226562, 254.33880615234375, 14.7786865234375, 15.74127197265625]},
#     #                                           {'cls': 9, 'conf': 0.31643879413604736, 'xywh': [218.50730895996094, 121.5130386352539, 19.016326904296875, 21.618209838867188]}],
#     #                                 'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                                 'orig_shape': [800, 448]}],
#     #                      'front': [{'boxes': [{'cls': 0, 'conf': 0.8558123111724854, 'xywh': [118.41397094726562, 712.606689453125, 123.37725830078125, 91.959716796875]},
#     #                                           {'cls': 2, 'conf': 0.6189518570899963, 'xywh': [379.9136657714844, 235.4759521484375, 29.53607177734375, 18.2882080078125]},
#     #                                           {'cls': 3, 'conf': 0.5429285764694214, 'xywh': [234.20474243164062, 445.35699462890625, 15.903961181640625, 14.31201171875]},
#     #                                           {'cls': 3, 'conf': 0.470856249332428, 'xywh': [160.22183227539062, 211.13526916503906, 24.421142578125, 16.7626953125]},
#     #                                           {'cls': 3, 'conf': 0.27346524596214294, 'xywh': [121.74200439453125, 178.4034881591797, 17.986297607421875, 14.07366943359375]}],
#     #                                 'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                                 'orig_shape': [800, 448]}],
#     #                      'back': [{'boxes': [{'cls': 2, 'conf': 0.6443317532539368, 'xywh': [309.83270263671875, 277.5762939453125, 58.06854248046875, 41.535400390625]},
#     #                                          {'cls': 2, 'conf': 0.6104090213775635, 'xywh': [271.727783203125, 230.6852569580078, 27.7191162109375, 99.20986938476562]},
#     #                                          {'cls': 5, 'conf': 0.35948097705841064, 'xywh': [166.108154296875, 54.37659454345703, 9.83648681640625, 13.227676391601562]},
#     #                                          {'cls': 3, 'conf': 0.28255370259284973, 'xywh': [116.77203369140625, 714.3328857421875, 13.421356201171875, 10.608642578125]}],
#     #                                'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                                'orig_shape': [800, 448]}],
#     #                      'left': [{'boxes': [{'cls': 8, 'conf': 0.3863007724285126, 'xywh': [222.90707397460938, 573.1998901367188, 13.47552490234375, 17.266357421875]},
#     #                                          {'cls': 8, 'conf': 0.2716498374938965, 'xywh': [215.0440673828125, 624.0682373046875, 10.961029052734375, 38.681884765625]}],
#     #                                'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                                'orig_shape': [800, 448]}],
#     #                      'bottom': [{'boxes': [{'cls': 4, 'conf': 0.5928195118904114, 'xywh': [188.96578979492188, 321.5050964355469, 17.25274658203125, 30.540771484375]},
#     #                                            {'cls': 5, 'conf': 0.44345614314079285, 'xywh': [117.20083618164062, 341.8561096191406, 16.00909423828125, 14.0968017578125]},
#     #                                            {'cls': 4, 'conf': 0.3789197504520416, 'xywh': [260.8472595214844, 314.67327880859375, 21.231201171875, 18.440673828125]},
#     #                                            {'cls': 4, 'conf': 0.36774688959121704, 'xywh': [223.8518829345703, 308.62725830078125, 71.89224243164062, 7.984375]},
#     #                                            {'cls': 4, 'conf': 0.3474479913711548, 'xywh': [259.1634826660156, 325.0404357910156, 17.92181396484375, 36.91143798828125]},
#     #                                            {'cls': 4, 'conf': 0.29864656925201416, 'xywh': [257.58709716796875, 328.866455078125, 15.4774169921875, 37.7149658203125]},
#     #                                            {'cls': 4, 'conf': 0.2818424105644226, 'xywh': [220.4687957763672, 310.0384216308594, 67.38021850585938, 9.98681640625]}],
#     #                                  'names': {'0': 'Screen_cracked', '1': 'Housing_cracked', '2': 'Screen_Major_scratch', '3': 'Screen_Minor_scratch', '4': 'Housing_Major_scratch', '5': 'Housing_Minor_scratch', '6': 'Dent', '7': 'Ding', '8': 'Discoloration', '9': 'Cover_spot'},
#     #                                  'orig_shape': [800, 448]}]}'''




#     results = get_ai_grade(
#         detection_results=detection_results,
#         device_model="iPhone 12",  # Optional
#         dimensions_csv="data/device_dimensions.csv"  # Optional
#     )

#     # Simple grade access (backwards compatible)
#     screen_grade = results.get("ai_screen")  # Returns "A1"
#     housing_grade = results.get("ai_housing")  # Returns "A2"

#     print("Screen Grade:", screen_grade)
#     print("Housing Grade:", housing_grade)