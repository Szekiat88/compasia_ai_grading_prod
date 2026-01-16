# grading_system/core/enums.py
from enum import Enum
from typing import List


# Existing enums stay the same
class DevicePosition(Enum):
    """Device positions for image capture"""
    DEVICE_FRONT_INACTIVE = "deviceFrontInactive"
    DEVICE_BACK = "deviceBack"
    DEVICE_LEFT = "deviceLeft"
    DEVICE_RIGHT = "deviceRight"
    DEVICE_TOP = "deviceTop"
    DEVICE_BOTTOM = "deviceBottom"


class DefectType(Enum):
    """Types of defects that can be detected"""
    SCREEN_CRACKED = "Screen_cracked"
    SCREEN_MAJOR_SCRATCH = "Screen_Major_scratch"
    SCREEN_MINOR_SCRATCH = "Screen_Minor_scratch"
    HOUSING_CRACKED = "Housing_cracked"
    HOUSING_MAJOR_SCRATCH = "Housing_Major_scratch"
    HOUSING_MINOR_SCRATCH = "Housing_Minor_scratch"
    DENT_DING = "DentDing"
    DISCOLORATION = "Discoloration"
    COVER_SPOT = "cover_spot"


class Grade(Enum):
    """Possible grades for device condition"""
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    C1 = "C1"
    R1 = "R1"
    R2 = "R2"

    def __lt__(self, other):
        # Order: A1 < A2 < A3 < C1 < R1 < R2
        order = ["A1", "A2", "A3", "C1", "R1", "R2"]
        return order.index(self.value) < order.index(other.value)

    def __gt__(self, other):
        # This enables > comparison
        order = ["A1", "A2", "A3", "C1", "R1", "R2"]
        return order.index(self.value) > order.index(other.value)

    def __eq__(self, other):
        # This enables == comparison
        if not isinstance(other, Grade):
            return NotImplemented
        return self.value == other.value


class ScratchType(Enum):
    """
    Types of scratches with size thresholds in mm
    Used for new area-based grading system
    """
    MICRO = "micro_scratch"  # ≤5mm
    MINOR = "minor_scratch"  # 5mm-10mm
    MAJOR = "major_scratch"  # 10mm-20mm

    @classmethod
    def get_type_by_size(cls, size_mm: float) -> 'ScratchType':
        """Determine scratch type based on size in mm"""
        if size_mm <= 5:
            return cls.MICRO
        elif size_mm <= 10:
            return cls.MINOR
        else:
            return cls.MAJOR


class DeviceArea(Enum):
    """
    Areas for each device face for area-based grading
    """
    # Back/Screen areas (2x3 grid)
    BACK_AREA_1 = "back_area_1"
    BACK_AREA_2 = "back_area_2"
    BACK_AREA_3 = "back_area_3"
    BACK_AREA_4 = "back_area_4"
    BACK_AREA_5 = "back_area_5"
    BACK_AREA_6 = "back_area_6"

    # Left side areas
    LEFT_SIDE_AREA_1 = "left_side_area_1"  # Bottom
    LEFT_SIDE_AREA_2 = "left_side_area_2"  # Top

    # Right side areas
    RIGHT_SIDE_AREA_1 = "right_side_area_1"  # Bottom
    RIGHT_SIDE_AREA_2 = "right_side_area_2"  # Top

    # Top/Bottom (separate areas)
    TOP_AREA = "top_area"
    BOTTOM_AREA = "bottom_area"

    @classmethod
    def get_areas_for_position(cls, position: DevicePosition) -> list['DeviceArea']:
        """Get list of areas for a given device position"""
        if position in [DevicePosition.DEVICE_FRONT_INACTIVE, DevicePosition.DEVICE_BACK]:
            return [cls.BACK_AREA_1, cls.BACK_AREA_2, cls.BACK_AREA_3,
                    cls.BACK_AREA_4, cls.BACK_AREA_5, cls.BACK_AREA_6]
        elif position == DevicePosition.DEVICE_LEFT:
            return [cls.LEFT_SIDE_AREA_1, cls.LEFT_SIDE_AREA_2]
        elif position == DevicePosition.DEVICE_RIGHT:
            return [cls.RIGHT_SIDE_AREA_1, cls.RIGHT_SIDE_AREA_2]
        elif position == DevicePosition.DEVICE_TOP:
            return [cls.TOP_AREA]
        else:  # Bottom
            return [cls.BOTTOM_AREA]