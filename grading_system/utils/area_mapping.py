# utils/area_mapping.py
from typing import Tuple
from ..core.enums import DevicePosition, DeviceArea


class AreaMapping:
    """Maps physical coordinates to device areas based on position"""

    def __init__(self, image_width: int, image_height: int):
        """
        Initialize with image dimensions

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
        """
        self.image_width = image_width
        self.image_height = image_height

    def pixel_to_normalized(self, x_pixel: int, y_pixel: int) -> Tuple[float, float]:
        """Convert pixel coordinates to normalized coordinates (0-1)"""
        x_norm = x_pixel / self.image_width
        y_norm = y_pixel / self.image_height
        return (x_norm, y_norm)

    def get_area_from_pixels(self, x_pixel: int, y_pixel: int, position: DevicePosition) -> DeviceArea:
        """
        Get area for a point using pixel coordinates

        Args:
            x_pixel: X coordinate in pixels
            y_pixel: Y coordinate in pixels
            position: Device position being examined
        """
        x_norm, y_norm = self.pixel_to_normalized(x_pixel, y_pixel)
        return self.get_area(x_norm, y_norm, position)

    def get_area_divisions_pixels(self) -> Tuple[list, list]:
        """Get the pixel values where areas are divided"""
        width_divisions = [self.image_width // 2]  # Middle cut
        height_divisions = [
            self.image_height // 3,  # First third
            (self.image_height * 2) // 3  # Second third
        ]
        return width_divisions, height_divisions

    def get_area(self, x: float, y: float, position: DevicePosition) -> DeviceArea:
        """
        Get area for a point given device position.
        Coordinates should be normalized (0-1).

        Args:
            x: Normalized x coordinate (0-1)
            y: Normalized y coordinate (0-1)
            position: Device position/face being examined

        Returns:
            DeviceArea enum indicating which area the point belongs to
        """
        if position in [DevicePosition.DEVICE_FRONT_INACTIVE, DevicePosition.DEVICE_BACK]:
            return self._get_back_area(x, y)
        elif position in [DevicePosition.DEVICE_LEFT, DevicePosition.DEVICE_RIGHT]:
            return self._get_side_area(x, y, position)
        elif position == DevicePosition.DEVICE_TOP:
            return DeviceArea.TOP_AREA
        else:  # Bottom
            return DeviceArea.BOTTOM_AREA

    def _get_back_area(self, x: float, y: float) -> DeviceArea:
        """Get area for back/front face (2x3 grid)"""
        # Split into 2 columns
        col = 1 if x < 0.5 else 2

        # Split into 3 rows
        if y < 0.333:
            row = 1
        elif y < 0.666:
            row = 2
        else:
            row = 3

        # Map to area enum
        area_map = {
            (1, 1): DeviceArea.BACK_AREA_1,
            (2, 1): DeviceArea.BACK_AREA_2,
            (1, 2): DeviceArea.BACK_AREA_3,
            (2, 2): DeviceArea.BACK_AREA_4,
            (1, 3): DeviceArea.BACK_AREA_5,
            (2, 3): DeviceArea.BACK_AREA_6
        }
        return area_map[(col, row)]

    def _get_side_area(self, x: float, y: float, position: DevicePosition) -> DeviceArea:
        """Get area for side face (2 vertical sections)"""
        # Split into 2 vertical sections based on position
        if position == DevicePosition.DEVICE_LEFT:
            return DeviceArea.LEFT_SIDE_AREA_1 if y <= 0.5 else DeviceArea.LEFT_SIDE_AREA_2
        else:  # Right side
            return DeviceArea.RIGHT_SIDE_AREA_1 if y <= 0.5 else DeviceArea.RIGHT_SIDE_AREA_2

    def get_area_bounds(self, area: DeviceArea) -> Tuple[float, float, float, float]:
        """
        Get normalized bounds (x1, y1, x2, y2) for an area

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in normalized coordinates
        """
        if area in [DeviceArea.BACK_AREA_1, DeviceArea.BACK_AREA_2]:
            row = 0
            col = 0 if area == DeviceArea.BACK_AREA_1 else 1
            return (col * 0.5, row * 0.333, (col + 1) * 0.5, (row + 1) * 0.333)

        elif area in [DeviceArea.BACK_AREA_3, DeviceArea.BACK_AREA_4]:
            row = 1
            col = 0 if area == DeviceArea.BACK_AREA_3 else 1
            return (col * 0.5, row * 0.333, (col + 1) * 0.5, (row + 1) * 0.333)

        elif area in [DeviceArea.BACK_AREA_5, DeviceArea.BACK_AREA_6]:
            row = 2
            col = 0 if area == DeviceArea.BACK_AREA_5 else 1
            return (col * 0.5, row * 0.333, (col + 1) * 0.5, (row + 1) * 0.333)

        elif area == DeviceArea.LEFT_SIDE_AREA_1:
            return (0, 0.5, 1, 1)
        elif area == DeviceArea.LEFT_SIDE_AREA_2:
            return (0, 0, 1, 0.5)
        elif area == DeviceArea.RIGHT_SIDE_AREA_1:
            return (0, 0.5, 1, 1)
        elif area == DeviceArea.RIGHT_SIDE_AREA_2:
            return (0, 0, 1, 0.5)
        elif area == DeviceArea.TOP_AREA:
            return (0, 0, 1, 1)
        elif area == DeviceArea.BOTTOM_AREA:
            return (0, 0, 1, 1)
        else:
            raise ValueError(f"Unknown area: {area}")