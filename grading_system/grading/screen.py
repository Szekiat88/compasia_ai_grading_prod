# grading/screen.py
from typing import List, Dict
from dataclasses import dataclass
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('screen_grading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

from grading_system.core.enums import DefectType, Grade, DeviceArea, DevicePosition, ScratchType
from grading_system.core.models import GradingResult, DefectCount, AreaDefects
from grading_system.utils.area_mapping import AreaMapping
from grading_system.utils.measurements import SizeMeasurement


class ScreenGrading:
    """
    Screen grading implementation using area-based assessment.
    Follows new grading criteria with area-based scratch assessment.

    Scratch Size Classifications:
    - Micro scratches: ≤5mm
    - Minor scratches: 5mm-10mm
    - Major scratches: 10mm-20mm

    Grading Criteria:
    A1 (Excellent):
    - Micro scratches: ≤2 areas
    - No minor/major scratches

    A2 (Good):
    - Micro scratches: ≤3 areas
    - Minor scratches: ≤1 area
    - No major scratches

    A3 (Fair):
    - Micro scratches: ≤4 areas
    - Minor scratches: ≤2 areas
    - Major scratches: ≤1 area

    C1 (Acceptable):
    - Heavy scratches (scratches in ≥75% areas)
    - Or exceeds A3 criteria
    """

    def __init__(self, height_mm: float, width_mm: float, thickness_mm: float):
        logger.info(f"Initializing ScreenGrading with dimensions - "
                    f"height: {height_mm}mm, width: {width_mm}mm, thickness: {thickness_mm}mm")

        self.measurement = SizeMeasurement(
            device_width_mm=width_mm,
            device_height_mm=height_mm,
            device_thickness_mm=thickness_mm
        )
        # Size thresholds in millimeters
        self.MICRO_SCRATCH_THRESHOLD = 5    # ≤5mm
        self.MINOR_SCRATCH_THRESHOLD = 10   # 5-10mm
        self.MAJOR_SCRATCH_THRESHOLD = 20   # 10-20mm
        self.HEAVY_SCRATCH_COVERAGE_THRESHOLD = 75  # percentage of areas affected

        logger.info(f"Initialized thresholds - "
                    f"Micro: ≤{self.MICRO_SCRATCH_THRESHOLD}mm, "
                    f"Minor: ≤{self.MINOR_SCRATCH_THRESHOLD}mm, "
                    f"Major: ≤{self.MAJOR_SCRATCH_THRESHOLD}mm, "
                    f"Heavy coverage: {self.HEAVY_SCRATCH_COVERAGE_THRESHOLD}%")

    def _classify_scratch_by_size(self, size_mm: float, original_type: str) -> ScratchType:
        """Classify scratch based on size, considering original YOLO classification"""
        logger.debug(f"Classifying scratch - Size: {size_mm:.2f}mm, Original type: {original_type}")

        if size_mm <= self.MICRO_SCRATCH_THRESHOLD:
            scratch_type = ScratchType.MICRO
        elif size_mm <= self.MINOR_SCRATCH_THRESHOLD:
            scratch_type = ScratchType.MINOR
        else:
            scratch_type = ScratchType.MAJOR

        logger.debug(f"Classified as: {scratch_type}")
        return scratch_type

    def calculate_grade(self, yolo_results) -> GradingResult:
        """Calculate screen grade based on YOLOv8 detection results."""
        logger.info("Starting screen grade calculation")

        if not yolo_results or not yolo_results.boxes:
            logger.info("No detection results found, returning A1 grade")
            return GradingResult(
                grade=Grade.A1,
                defect_count=DefectCount(),
                details="No defects detected"
            )

        # Initialize area mapping
        image_height, image_width = yolo_results.orig_shape[:2]
        area_mapper = AreaMapping(image_width, image_height)
        logger.info(f"Initialized area mapping for image size: {image_width}x{image_height}")

        # Track defects by area
        area_defects: Dict[DeviceArea, AreaDefects] = {}
        for area in DeviceArea.get_areas_for_position(DevicePosition.DEVICE_FRONT_INACTIVE):
            area_defects[area] = AreaDefects(area=area)

        logger.debug(f"Initialized {len(area_defects)} areas for tracking")

        # Process each detection
        has_cracks = False
        detection_count = len(yolo_results.boxes)
        logger.info(f"Processing {detection_count} detections")

        for i, box in enumerate(yolo_results.boxes):
            class_id = int(box.cls[0])
            original_defect_type = yolo_results.names[class_id]

            logger.debug(f"Processing detection {i + 1}/{detection_count}: {original_defect_type}")

            if (original_defect_type == 'Screen_cracked') or (original_defect_type == 'Housing_cracked'):
                has_cracks = True
                logger.warning(f"Crack detected: {original_defect_type}")
                continue

            # Get center point and area
            x_center, y_center = box.xywh[0][0], box.xywh[0][1]
            x_norm, y_norm = area_mapper.pixel_to_normalized(
                int(x_center), int(y_center)
            )
            area = area_mapper.get_area(x_norm, y_norm, DevicePosition.DEVICE_FRONT_INACTIVE)

            logger.debug(f"Mapped to area {area} at normalized position ({x_norm:.2f}, {y_norm:.2f})")

            # Calculate size and classify
            width_mm, height_mm = self.measurement.get_size_mm(
                box.xywh[0],
                (image_width, image_height),
                DevicePosition.DEVICE_FRONT_INACTIVE
            )
            scratch_size = max(width_mm, height_mm)
            logger.debug(f"Defect size: {scratch_size:.2f}mm (width: {width_mm:.2f}mm, height: {height_mm:.2f}mm)")

            # Classify based on both size and original type
            scratch_type = self._classify_scratch_by_size(
                scratch_size,
                original_defect_type
            )

            # Add to appropriate list
            if scratch_type == ScratchType.MICRO:
                area_defects[area].micro_scratches.append(scratch_size)
            elif scratch_type == ScratchType.MINOR:
                area_defects[area].minor_scratches.append(scratch_size)
            else:
                area_defects[area].major_scratches.append(scratch_size)

            logger.debug(f"Added {scratch_type} scratch to area {area}")

        # Count areas with defects
        total_areas = len(area_defects)
        areas_with_scratches = sum(1 for area in area_defects.values()
                                   if (area.micro_scratches or
                                       area.minor_scratches or
                                       area.major_scratches))

        # Calculate coverage percentage
        coverage_percentage = (areas_with_scratches / total_areas) * 100
        logger.info(
            f"Screen coverage: {coverage_percentage:.1f}% ({areas_with_scratches}/{total_areas} areas affected)")

        # Check for heavy scratches (high area coverage)
        has_heavy_scratches = coverage_percentage >= self.HEAVY_SCRATCH_COVERAGE_THRESHOLD
        if has_heavy_scratches:
            logger.warning(f"Heavy scratch coverage detected: {coverage_percentage:.1f}%")

        areas_with_micro = sum(1 for area in area_defects.values()
                               if area.micro_scratches)
        areas_with_minor = sum(1 for area in area_defects.values()
                               if area.minor_scratches)
        areas_with_major = sum(1 for area in area_defects.values()
                               if area.major_scratches)

        logger.info(f"Areas affected - "
                    f"Micro: {areas_with_micro}, "
                    f"Minor: {areas_with_minor}, "
                    f"Major: {areas_with_major}")

        # Create defect count
        defect_count = DefectCount(
            minor_scratches=sum(len(area.micro_scratches) + len(area.minor_scratches)
                                for area in area_defects.values()),
            major_scratches=sum(len(area.major_scratches)
                                for area in area_defects.values()),
            area_defects=area_defects
        )

        logger.info(f"Total defect count - "
                    f"Minor scratches: {defect_count.minor_scratches}, "
                    f"Major scratches: {defect_count.major_scratches}")

        # Grade determination
        if has_cracks:
            logger.warning("Assigning R2 grade due to crack")
            return GradingResult(
                grade=Grade.R2,
                defect_count=defect_count,
                details="Screen is cracked"
            )

        # C1 Grade - Heavy scratch coverage or exceeding A3
        if (has_heavy_scratches or
                areas_with_micro > 4 or
                areas_with_minor > 2 or
                areas_with_major > 1):
            details = ("Heavy scratch coverage across screen" if has_heavy_scratches else
                       f"Exceeds A3 criteria: micro scratches in {areas_with_micro} areas, "
                       f"minor scratches in {areas_with_minor} areas, "
                       f"major scratches in {areas_with_major} areas")
            logger.info(f"Assigning C1 grade: {details}")
            return GradingResult(
                grade=Grade.C1,
                defect_count=defect_count,
                details=details
            )

        # Check from best to worst grade
        # A1 Grade (Best condition)
        if (areas_with_micro <= 2 and
                areas_with_minor == 0 and
                areas_with_major == 0):
            logger.info("Assigning A1 grade - Excellent condition")
            return GradingResult(
                grade=Grade.A1,
                defect_count=defect_count,
                details="Excellent condition"
            )

        # A2 Grade
        elif (areas_with_micro <= 3 and
              areas_with_minor <= 1 and
              areas_with_major == 0):
            logger.info("Assigning A2 grade - Good condition")
            return GradingResult(
                grade=Grade.A2,
                defect_count=defect_count,
                details="Good condition"
            )

        # A3 Grade
        elif (areas_with_micro <= 4 and
              areas_with_minor <= 2 and
              areas_with_major <= 1):
            logger.info("Assigning A3 grade - Fair condition")
            return GradingResult(
                grade=Grade.A3,
                defect_count=defect_count,
                details="Fair condition"
            )

        # Default case
        logger.warning("No specific grade condition met, defaulting to A2")
        return GradingResult(
            grade=Grade.A2,
            defect_count=defect_count,
            details="No condition got fulfilled"
        )

    def process_image(self, image_path: str, model) -> GradingResult:
        """Process a single screen image and return grading result."""
        logger.info(f"Processing image: {image_path}")

        # Run inference
        results = model(image_path)
        if not results or len(results) == 0:
            logger.info("No detection results from model, returning A1 grade")
            return GradingResult(
                grade=Grade.A1,
                defect_count=DefectCount(),
                details="No defects detected"
            )

        # Grade first result
        logger.info("Model detection successful, calculating grade")
        return self.calculate_grade(results[0])