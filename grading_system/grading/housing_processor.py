from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('housing_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

from grading_system.core.enums import DefectType, Grade, DeviceArea, DevicePosition, ScratchType
from grading_system.core.models import AreaDefects
from grading_system.utils.measurements import SizeMeasurement
from grading_system.utils.area_mapping import AreaMapping


@dataclass
class SizedDefect:
    """Helper class to store defect with its size in mm"""
    defect_type: str
    width_mm: float
    height_mm: float
    x_center: float
    y_center: float
    image_size: Tuple[int, int]

    @property
    def max_dimension(self) -> float:
        """Get maximum dimension in mm"""
        return max(self.width_mm, self.height_mm)


@dataclass
class ProcessedDefects:
    """Raw processed defects from a section before any grading decisions"""
    area_defects: Dict[DeviceArea, AreaDefects]  # For area-mapped scratches
    sized_defects: List[SizedDefect]  # Raw defect data with sizes
    has_crack: bool = False  # Only essential flag we need


class HousingProcessor:
    """Processes housing defects and maps them to areas without making grading decisions"""

    def __init__(self, height_mm: float, width_mm: float, thickness_mm: float):
        logger.info(f"Initializing HousingProcessor with dimensions - "
                    f"height: {height_mm}mm, width: {width_mm}mm, thickness: {thickness_mm}mm")

        self.measurement = SizeMeasurement(
            device_width_mm=width_mm,
            device_height_mm=height_mm,
            device_thickness_mm=thickness_mm
        )
        self.MINOR_SCRATCH_THRESHOLD = 4  # ≤4mm for initial scratch classification
        logger.info(f"Set MINOR_SCRATCH_THRESHOLD to {self.MINOR_SCRATCH_THRESHOLD}mm")

    def _process_detections(self, yolo_results, position: DevicePosition) -> List[SizedDefect]:
        """Convert YOLOv8 detections to sized defects"""
        logger.info(f"Processing detections for position: {position}")
        sized_defects = []
        boxes = yolo_results.boxes
        image_size = yolo_results.orig_shape[:2]  # (height, width)

        logger.debug(f"Processing {len(boxes)} detections from image size {image_size}")

        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            defect_type = yolo_results.names[class_id]

            if defect_type == 'camera_lens':
                continue

            # if defect type housing cracked and DevicePosition is left, right , top , bottom, skip including this defect in list
            if (defect_type in ['Housing_cracked', 'Screen_cracked',
                                "dense_crack", "hairline_crack"]) and (position in
                                                       [DevicePosition.DEVICE_LEFT,
                                                        DevicePosition.DEVICE_RIGHT,
                                                        DevicePosition.DEVICE_TOP,
                                                        DevicePosition.DEVICE_BOTTOM]):
                logger.info(f"Skipping defect {i + 1}/{len(boxes)}: "
                             f"type={defect_type}, position={position}")
                continue

            # Get center coordinates from box
            x_center = box.xywh[0][0]
            y_center = box.xywh[0][1]

            logger.debug(f"Processing detection {i + 1}/{len(boxes)}: "
                         f"type={defect_type}, center=({x_center:.2f}, {y_center:.2f})")

            # Get size in mm based on position
            width_mm, height_mm = self.measurement.get_size_mm(
                box.xywh[0],
                (image_size[1], image_size[0]),  # Convert to (width, height)
                position
            )

            logger.debug(f"Converted size: width={width_mm:.2f}mm, height={height_mm:.2f}mm")

            sized_defects.append(SizedDefect(
                defect_type=defect_type,
                width_mm=width_mm,
                height_mm=height_mm,
                x_center=x_center,
                y_center=y_center,
                image_size=image_size
            ))

        logger.info(f"Processed {len(sized_defects)} defects for position {position}")
        return sized_defects

    def _process_scratches(self, sized_defects: List[SizedDefect],
                           position: DevicePosition,
                           area_mapper: AreaMapping) -> Dict[DeviceArea, AreaDefects]:
        """Process scratches using area-based assessment"""
        logger.info(f"Processing scratches for position: {position}")

        # Initialize area defects
        area_defects = {}
        areas = DeviceArea.get_areas_for_position(position)
        for area in areas:
            area_defects[area] = AreaDefects(area=area)

        logger.debug(f"Initialized {len(areas)} areas for position {position}")

        # Process scratches
        scratch_count = 0
        for defect in sized_defects:
            if 'scratch' in defect.defect_type.lower():
                scratch_count += 1
                x_norm, y_norm = area_mapper.pixel_to_normalized(
                    int(defect.x_center),
                    int(defect.y_center)
                )
                area = area_mapper.get_area(x_norm, y_norm, position)

                logger.debug(f"Processing scratch {scratch_count} - "
                             f"size: {defect.max_dimension:.2f}mm, "
                             f"normalized position: ({x_norm:.2f}, {y_norm:.2f}), "
                             f"mapped to area: {area}")

                if defect.max_dimension <= self.MINOR_SCRATCH_THRESHOLD:
                    area_defects[area].minor_scratches.append(defect.max_dimension)
                    logger.debug(f"Classified as minor scratch in {area}")
                else:
                    area_defects[area].major_scratches.append(defect.max_dimension)
                    logger.debug(f"Classified as major scratch in {area}")

        # Log summary of scratch processing
        for area, defects in area_defects.items():
            if defects.minor_scratches or defects.major_scratches:
                logger.info(f"Area {area} - "
                            f"Minor scratches: {len(defects.minor_scratches)}, "
                            f"Major scratches: {len(defects.major_scratches)}")

        return area_defects

    def process_section(self, sized_defects: List[SizedDefect], position: DevicePosition) -> ProcessedDefects:
        """Common processing for any section"""
        logger.info(f"Processing section for position: {position}")

        if not sized_defects:
            logger.info("No defects to process, returning empty ProcessedDefects")
            return ProcessedDefects(area_defects={}, sized_defects=[])

        # Check for cracks
        has_crack = any(d.defect_type in ['Housing_cracked', 'Screen_cracked',
                                          "dense_crack", "hairline_crack"] for d in sized_defects)

        if has_crack:
            logger.warning(f"Crack detected in position {position}")
            return ProcessedDefects(
                area_defects={},
                sized_defects=sized_defects,
                has_crack=True
            )

        # Initialize area mapping
        image_height, image_width = sized_defects[0].image_size
        area_mapper = AreaMapping(image_width, image_height)
        logger.debug(f"Initialized area mapping for image size: {image_width}x{image_height}")

        # Process scratches using area-based assessment
        area_defects = self._process_scratches(sized_defects, position, area_mapper)

        result = ProcessedDefects(
            area_defects=area_defects,
            sized_defects=sized_defects
        )

        logger.info(f"Section processing complete for {position} - "
                    f"Total defects: {len(sized_defects)}")
        return result

    def process_back(self, detection_results) -> ProcessedDefects:
        """Process back section defects"""
        logger.info("Processing back section")
        if not detection_results:
            logger.info("No detection results for back section")
            return ProcessedDefects(area_defects={}, sized_defects=[])

        sized_defects = self._process_detections(detection_results, DevicePosition.DEVICE_BACK)
        return self.process_section(sized_defects, DevicePosition.DEVICE_BACK)

    def process_sides(self, detection_results) -> Dict[str, ProcessedDefects]:
        """Process side section defects (handles both left and right)"""
        logger.info("Processing side sections")
        if not detection_results:
            logger.info("No detection results for sides")
            return {}

        results = {}
        if isinstance(detection_results, list):
            logger.info("Processing multiple side results")
            for result, pos in zip(detection_results,
                                   [DevicePosition.DEVICE_LEFT, DevicePosition.DEVICE_RIGHT]):
                if result is not None:
                    logger.info(f"Processing {pos.value} side")
                    sized_defects = self._process_detections(result, pos)
                    results[pos.value] = self.process_section(sized_defects, pos)
                else:
                    logger.warning(f"No detection results for {pos.value} side")

        else:
            # For single side processing
            logger.info("Processing single side result")
            pos = DevicePosition.DEVICE_LEFT  # Default to left if position not specified
            sized_defects = self._process_detections(detection_results, pos)
            results[pos.value] = self.process_section(sized_defects, pos)

        logger.info(f"Completed processing sides. Processed positions: {list(results.keys())}")
        return results

    def process_top(self, detection_results) -> ProcessedDefects:
        """Process top section defects"""
        logger.info("Processing top section")
        if not detection_results:
            logger.info("No detection results for top section")
            return ProcessedDefects(area_defects={}, sized_defects=[])

        sized_defects = self._process_detections(detection_results, DevicePosition.DEVICE_TOP)
        return self.process_section(sized_defects, DevicePosition.DEVICE_TOP)

    def process_bottom(self, detection_results) -> ProcessedDefects:
        """Process bottom section defects"""
        logger.info("Processing bottom section")
        if not detection_results:
            logger.info("No detection results for bottom section")
            return ProcessedDefects(area_defects={}, sized_defects=[])

        sized_defects = self._process_detections(detection_results, DevicePosition.DEVICE_BOTTOM)
        return self.process_section(sized_defects, DevicePosition.DEVICE_BOTTOM)