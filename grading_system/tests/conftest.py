# tests/conftest.py
import pytest
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class MockBox:
    """Mock YOLOv8 detection box"""
    cls: torch.Tensor
    conf: torch.Tensor
    xywh: torch.Tensor

    def __post_init__(self):
        # Ensure tensors are properly formatted
        if not isinstance(self.cls, torch.Tensor):
            self.cls = torch.tensor([self.cls])
        if not isinstance(self.conf, torch.Tensor):
            self.conf = torch.tensor([self.conf])
        if not isinstance(self.xywh, torch.Tensor):
            self.xywh = torch.tensor([[self.xywh[0], self.xywh[1], self.xywh[2], self.xywh[3]]])

@dataclass
class MockResults:
    """Mock YOLOv8 detection results"""
    boxes: List[MockBox]
    orig_shape: tuple
    names: dict

    def __post_init__(self):
        # Ensure boxes is a list
        if not isinstance(self.boxes, list):
            self.boxes = [self.boxes]

def create_mock_box(class_id: int, confidence: float, x: float, y: float, w: float, h: float) -> MockBox:
    """Helper to create a mock detection box with default values"""
    return MockBox(
        cls=torch.tensor([class_id]),
        conf=torch.tensor([confidence]),
        xywh=torch.tensor([[x, y, w, h]])
    )

@pytest.fixture
def yolo_class_names():
    """YOLOv8 class mapping fixture"""
    return {
        0: 'Screen_cracked',
        1: 'Housing_cracked',
        2: 'Screen_Major_scratch',
        3: 'Screen_Minor_scratch',
        4: 'Housing_Major_scratch',
        5: 'Housing_Minor_scratch',
        6: 'DentDing',
        7: 'Discoloration',
        8: 'Cover_spot'
    }

@pytest.fixture
def image_size():
    """Standard image size fixture"""
    return (960, 960)