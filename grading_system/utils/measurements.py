# grading_system/utils/measurements.py
from enum import Enum
from typing import Tuple
import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))
from grading_system.core.enums import DevicePosition


class SizeMeasurement:
    def __init__(self, device_width_mm: float, device_height_mm: float, device_thickness_mm: float):
        """
        Initialize with all device dimensions

        Args:
            device_width_mm: Width of device (front/back width)
            device_height_mm: Height of device (front/back height)
            device_thickness_mm: Thickness of device
        """
        self.device_width_mm = device_width_mm
        self.device_height_mm = device_height_mm
        self.device_thickness_mm = device_thickness_mm

    def get_dimensions_for_position(self, position: DevicePosition) -> Tuple[float, float]:
        """
        Get relevant dimensions based on device position/orientation

        Args:
            position: DevicePosition enum indicating which face we're measuring

        Returns:
            Tuple of (width_mm, height_mm) for that face
        """
        if position in [DevicePosition.DEVICE_FRONT_INACTIVE, DevicePosition.DEVICE_BACK]:
            # Front and back use normal width and height
            return self.device_width_mm, self.device_height_mm

        elif position in [DevicePosition.DEVICE_LEFT, DevicePosition.DEVICE_RIGHT]:
            # Sides use thickness as width, height as height
            return self.device_thickness_mm, self.device_height_mm

        elif position in [DevicePosition.DEVICE_TOP, DevicePosition.DEVICE_BOTTOM]:
            # Top and bottom use width as width, thickness as height
            return self.device_width_mm, self.device_thickness_mm

        else:
            raise ValueError(f"Unknown device position: {position}")

    def get_size_mm(self, box, image_size: Tuple[int, int], position: DevicePosition) -> Tuple[float, float]:
        """
        Convert YOLO box dimensions to mm based on device position

        Args:
            box: YOLO detection box (xywh format)
            image_size: Tuple of (width, height) of the image
            position: DevicePosition enum indicating which face we're measuring
        """
        width_px, height_px = float(box[2]), float(box[3])

        # Get appropriate dimensions for this face
        face_width_mm, face_height_mm = self.get_dimensions_for_position(position)

        # if position in [DevicePosition.DEVICE_LEFT, DevicePosition.DEVICE_RIGHT]:
        #     ratio = face_height_mm / face_width_mm
        #     calculated_image_width = image_size[1] / ratio
        #     height_scale = face_height_mm / image_size[1]
        #     width_scale = face_width_mm / calculated_image_width
        #
        # if position in  [DevicePosition.DEVICE_TOP, DevicePosition.DEVICE_BOTTOM]:
        #     ratio = face_width_mm / face_height_mm
        #     calculated_image_height = image_size[0] / ratio
        #     width_scale = face_width_mm / image_size[0]
        #     height_scale = face_height_mm / calculated_image_height
        #
        # elif position in [DevicePosition.DEVICE_FRONT_INACTIVE, DevicePosition.DEVICE_BACK]:
        #     # Calculate mm per pixel for this face
        #     # width_scale = face_width_mm / image_size[0]
        #     height_scale = face_height_mm / image_size[1]
        #     width_scale = height_scale

        # Calculate mm per pixel for this face
        width_scale = face_width_mm / image_size[0]
        height_scale = face_height_mm / image_size[1]

        return width_px * width_scale, height_px * height_scale