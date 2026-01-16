from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
import os
import logging
from requests import session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('housing_grading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

from grading_system.core.enums import Grade, DeviceArea, DevicePosition
from grading_system.core.models import AreaDefects, DefectCount
from grading_system.grading.housing_processor import ProcessedDefects, SizedDefect


@dataclass
class HousingGradingResult:
    """Final housing grading result"""
    grade: Grade
    defect_count: DefectCount
    details: str


class HousingGrading:
    """Handles grading logic for housing based on processed defects from all sections"""

    def __init__(self):
        # Size thresholds for different defect types (in mm)
        self.MINOR_SCRATCH_THRESHOLD = 4
        self.HEAVY_SCRATCH_COVERAGE = 70  # percentage
        logger.info("Initialized HousingGrading with thresholds: "
                    f"MINOR_SCRATCH_THRESHOLD={self.MINOR_SCRATCH_THRESHOLD}mm, "
                    f"HEAVY_SCRATCH_COVERAGE={self.HEAVY_SCRATCH_COVERAGE}%")

    def _count_defects_by_size(self, sized_defects: List[SizedDefect],
                               defect_type: str, max_size: float) -> int:
        """Count defects of given type below size threshold"""
        logger.debug(f"Counting {defect_type} defects with max size {max_size}mm")
        count = 0

        for d in sized_defects:
            if d.defect_type == defect_type:
                if defect_type == 'Discoloration':
                    min_dimension = min(d.width_mm, d.height_mm)
                    if min_dimension <= max_size:
                        count += 1
                        logger.debug(f"Counted {defect_type} defect: width={d.width_mm}mm, height={d.height_mm}mm")
                else:
                    if d.max_dimension <= max_size:
                        count += 1
                        logger.debug(f"Counted {defect_type} defect: max_dimension={d.max_dimension}mm")

        logger.info(f"Total {defect_type} defects under {max_size}mm: {count}")
        return count

    def _count_defects_in_range(self, sized_defects: List[SizedDefect],
                                defect_type: str, min_size: float,
                                max_size: float) -> int:
        """Count defects of given type within size range"""
        logger.debug(f"Counting {defect_type} defects between {min_size}mm and {max_size}mm")
        count = 0

        for d in sized_defects:
            if d.defect_type == defect_type and min_size < d.max_dimension <= max_size:
                count += 1
                logger.debug(f"Counted {defect_type} defect: max_dimension={d.max_dimension}mm")

        logger.info(f"Total {defect_type} defects between {min_size}mm and {max_size}mm: {count}")
        return count

    def _process_section_defects(self, processed_defects: ProcessedDefects) -> Dict[str, int]:
        """Process non-scratch defects from a section"""
        logger.info(f"Processing section with {len(processed_defects.sized_defects)} defects")
        sized_defects = processed_defects.sized_defects

        result = {
            'dents_small': self._count_defects_by_size(sized_defects, 'DentDing', 0.5),
            'dents_medium': self._count_defects_in_range(sized_defects, 'DentDing', 0.5, 2),
            'dents_large': self._count_defects_in_range(sized_defects, 'DentDing', 2, 3),
            'dents_oversize': sum(1 for d in sized_defects
                                  if d.defect_type == 'DentDing' and d.max_dimension > 5),

            'discoloration_3mm': self._count_defects_by_size(sized_defects, 'Discoloration', 1.5),
            'discoloration_oversize': sum(1 for d in sized_defects
                                          if d.defect_type == 'Discoloration'
                                          and min(d.width_mm, d.height_mm) > 5),

            'cover_spots_2mm': self._count_defects_by_size(sized_defects, 'cover_spot', 2),
            'cover_spots_3mm': self._count_defects_in_range(sized_defects, 'cover_spot', 2, 3),
            'cover_spots_7mm': self._count_defects_in_range(sized_defects, 'cover_spot', 3, 7),
            'cover_spots_oversize': sum(1 for d in sized_defects
                                        if d.defect_type == 'cover_spot'
                                        and d.max_dimension > 7),
            'total_cover_spots': sum(1 for d in sized_defects if d.defect_type == 'cover_spot')

        }

        logger.info(f"Section processing results: {result}")
        return result

    def _combine_area_defects(self, sections: Dict[str, ProcessedDefects]) -> Dict[DeviceArea, AreaDefects]:
        """Combine area defects from all sections"""
        logger.info(f"Combining defects from {len(sections)} sections")
        combined_areas = {}

        for section_name, section_data in sections.items():
            logger.info(f"Processing section: {section_name}")

            # For sides (which contain a nested dictionary of left and right ProcessedDefects)
            if isinstance(section_data, dict):
                logger.info(f"Processing side section with {len(section_data)} positions")
                for side_position, side_defects in section_data.items():
                    logger.info(f"Processing side position: {side_position}")
                    for area, defects in side_defects.area_defects.items():
                        if area not in combined_areas:
                            combined_areas[area] = AreaDefects(area=area)
                            logger.debug(f"Created new AreaDefects for {area}")
                        combined_areas[area].minor_scratches.extend(defects.minor_scratches)
                        combined_areas[area].major_scratches.extend(defects.major_scratches)
                        logger.debug(f"Added {len(defects.minor_scratches)} minor and "
                                     f"{len(defects.major_scratches)} major scratches to {area}")

            # For regular sections (back, top, bottom) which are single ProcessedDefects
            else:
                for area, defects in section_data.area_defects.items():
                    if area not in combined_areas:
                        combined_areas[area] = AreaDefects(area=area)
                        logger.debug(f"Created new AreaDefects for {area}")
                    combined_areas[area].minor_scratches.extend(defects.minor_scratches)
                    combined_areas[area].major_scratches.extend(defects.major_scratches)
                    logger.debug(f"Added {len(defects.minor_scratches)} minor and "
                                 f"{len(defects.major_scratches)} major scratches to {area}")

            logger.info(f"Section {section_name} processed. Current combined areas: {list(combined_areas.keys())}")

        return combined_areas

    def _determine_grade(self,
                         area_defects: Dict[DeviceArea, AreaDefects],
                         defect_counts: Dict[str, int],
                         sections: Dict[str, ProcessedDefects]) -> Grade:
        """Determine grade based on all defects"""
        logger.info("Starting grade determination")
        logger.info(f"Input defect counts: {defect_counts}")

        # R2 check remains the same - any crack is R2
        has_crack = any(isinstance(section, dict) and any(s.has_crack for s in section.values())
                        or (not isinstance(section, dict) and section.has_crack)
                        for section in sections.values())
        if has_crack:
            logger.warning("Found crack in device - assigning R2 grade")
            return Grade.R2

        # Count areas affected by scratches
        areas_with_minor = sum(1 for area in area_defects.values() if area.minor_scratches)
        areas_with_major = sum(1 for area in area_defects.values() if area.major_scratches)
        logger.info(f"Areas affected - Minor: {areas_with_minor}, Major: {areas_with_major}")

        # Calculate bezel (top/bottom) scratches
        bezel_areas = [area for area in area_defects.keys()
                       if area in [DeviceArea.TOP_AREA, DeviceArea.BOTTOM_AREA]]
        # Left side scratches
        left_side_areas = [area for area in area_defects.keys()
                           if area in [DeviceArea.LEFT_SIDE_AREA_1, DeviceArea.LEFT_SIDE_AREA_2]]

        # Right side scratches
        right_side_areas = [area for area in area_defects.keys()
                            if area in [DeviceArea.RIGHT_SIDE_AREA_1, DeviceArea.RIGHT_SIDE_AREA_2]]

        bezel_minor_count = 0
        bezel_major_count = 0
        left_side_minor_count = 0
        left_side_major_count = 0
        right_side_minor_count = 0
        right_side_major_count = 0

        # Count bezel scratches
        for area in bezel_areas:
            if area in area_defects:
                bezel_minor_count += len(area_defects[area].minor_scratches)
                bezel_major_count += len(area_defects[area].major_scratches)
        # Count scratches in left side areas
        for area in left_side_areas:
            if area in area_defects:
                left_side_minor_count += len(area_defects[area].minor_scratches)
                left_side_major_count += len(area_defects[area].major_scratches)

        # Count scratches in right side areas
        for area in right_side_areas:
            if area in area_defects:
                right_side_minor_count += len(area_defects[area].minor_scratches)
                right_side_major_count += len(area_defects[area].major_scratches)

        # Calculate total edge scratches (bezel + sides)
        total_edge_minor_scratches = bezel_minor_count + left_side_minor_count + right_side_minor_count
        total_edge_major_scratches = bezel_major_count + left_side_major_count + right_side_major_count

        logger.info(f"Edge defect counts - "
                    f"Bezel minor: {bezel_minor_count}, "
                    f"Bezel major: {bezel_major_count}, "
                    f"Total edge minor: {total_edge_minor_scratches}, "
                    f"Total edge major: {total_edge_major_scratches}")

        # # switch off discoloration defect to see impact on accuracy as this has very high false positive
        # defect_counts['discoloration_oversize'] = 0
        # defect_counts['discoloration_3mm'] = 0

        # C1 conditions (worse than A3)
        if (areas_with_minor > 5 or
                areas_with_major > 3 or
                defect_counts['dents_oversize'] > 5 or
                defect_counts['discoloration_oversize'] > 5 or
                defect_counts['total_cover_spots'] > 12):
            logger.info("C1 condition met due to excessive defects")
            return Grade.C1

        # Special case: Concentrated bezel damage
        if (bezel_minor_count >= 4 or bezel_major_count >= 1) and (areas_with_minor <= 2):
            logger.info("A3 condition met due to concentrated bezel damage")
            return Grade.A3

        # A1 conditions (best condition)
        if (areas_with_minor <= 3 and
                areas_with_major <= 1 and
                defect_counts['dents_small'] <= 2 and
                defect_counts['discoloration_3mm'] <= 3 and
                defect_counts['total_cover_spots'] <= 5):
            logger.info("A1 condition met - device in best condition")
            return Grade.A1

        # # A2 conditions (good condition)
        # if (areas_with_minor == 4 and  # Exactly 4 areas
        #         areas_with_major == 2 and  # Exactly 2 areas
        #         ((defect_counts['dents_medium'] <= 3) or (defect_counts['dents_large'] <= 2)) and
        #         defect_counts['discoloration_3mm'] <= 8 and
        #         (3 < defect_counts['total_cover_spots'] <= 5)):
        #     logger.info("A2 condition met - device in good condition")
        #     return Grade.A2

        # # A3 conditions
        # if (areas_with_minor == 5 and  # Exactly 5 areas
        #         areas_with_major == 3 and  # Exactly 3 areas
        #         ((defect_counts['dents_medium'] <= 5) or (defect_counts['dents_large'] <= 3)) and
        #         defect_counts['discoloration_3mm'] <= 10 and
        #         (5 < defect_counts['total_cover_spots'] <= 8)):
        #     logger.info("A3 condition met exactly")
        #     return Grade.A3

        # If defects are within A2 range but not exact A2 criteria
        if (areas_with_minor <= 4 and
                areas_with_major <= 2 and
                ((defect_counts['dents_medium'] <= 3) or (defect_counts['dents_large'] <= 2)) and
                defect_counts['discoloration_3mm'] <= 10 and
                (5 < defect_counts['total_cover_spots'] <= 7)):
            logger.info("A2 condition met within range")
            return Grade.A2

        # If defects are within A3 range but not exact A3 criteria
        if (areas_with_minor <= 5 and
                areas_with_major <= 3 and
                ((defect_counts['dents_medium'] <= 5) or (defect_counts['dents_large'] <= 3)) and
                defect_counts['discoloration_3mm'] <= 10 and
                (7 < defect_counts['total_cover_spots'] <= 12)):
            logger.info("A3 condition met within range")
            return Grade.A3

        logger.info("No specific grade conditions met - defaulting to C1")
        return Grade.C1


    def calculate_grade(self, sections: Dict[str, ProcessedDefects]) -> HousingGradingResult:
        """Calculate final housing grade considering all sections"""
        logger.info("Starting housing grade calculation")

        # Combine area defects from all sections
        combined_areas = self._combine_area_defects(sections)
        logger.info(f"Combined areas processed: {list(combined_areas.keys())}")

        # Process other defects from all sections
        total_defect_counts = {
            'dents_small': 0, 'dents_medium': 0, 'dents_large': 0, 'dents_oversize': 0,
            'discoloration_3mm': 0, 'discoloration_oversize': 0,
            'cover_spots_2mm': 0, 'cover_spots_3mm': 0, 'cover_spots_7mm': 0,
            'cover_spots_oversize': 0, 'total_cover_spots': 0
        }

        logger.info(f"Initial defect counts: {total_defect_counts}")

        # Sum up defect counts across all sections
        for section_name, section in sections.items():
            logger.info(f"Processing section: {section_name}")
            if isinstance(section, dict):  # Handle side sections
                logger.debug(f"Processing side section {section_name} with {len(section)} subsections")
                for side_position, side_section in section.items():
                    logger.debug(f"Processing side position: {side_position}")
                    section_counts = self._process_section_defects(side_section)
                    for key in total_defect_counts:
                        total_defect_counts[key] += section_counts[key]
                    logger.debug(f"Updated counts after {side_position}: {total_defect_counts}")
            else:  # Handle regular sections
                logger.debug(f"Processing regular section: {section_name}")
                section_counts = self._process_section_defects(section)
                for key in total_defect_counts:
                    total_defect_counts[key] += section_counts[key]
                logger.debug(f"Updated counts after {section_name}: {total_defect_counts}")

        logger.info(f"Final defect counts: {total_defect_counts}")

        # Determine grade based on combined defects
        grade = self._determine_grade(combined_areas, total_defect_counts, sections)
        logger.info(f"Determined grade: {grade}")

        # Create defect count for result
        defect_count = DefectCount(
            minor_scratches=sum(len(area.minor_scratches)
                                for area in combined_areas.values()),
            major_scratches=sum(len(area.major_scratches)
                                for area in combined_areas.values()),
            area_defects=combined_areas
        )

        logger.info(f"Final defect count - Minor scratches: {defect_count.minor_scratches}, "
                    f"Major scratches: {defect_count.major_scratches}")

        result = HousingGradingResult(
            grade=grade,
            defect_count=defect_count,
            details=f"Grade {grade.value} condition"
        )

        logger.info(f"Final grading result: {result}")
        return result