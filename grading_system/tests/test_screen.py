# tests/test_screen.py
import pytest
import torch

import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))


from grading_system.core.enums import Grade
from grading_system.grading.screen import ScreenGrading
from grading_system.tests.conftest import MockResults, create_mock_box


class TestScreenGrading:
    def test_no_defects(self, yolo_class_names, image_size):
        """Test screen with no defects should get A1 grade"""
        grader = ScreenGrading()
        detection = MockResults(
            boxes=[],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        result = grader.calculate_grade(detection)
        assert result.grade == Grade.A1
        assert result.defect_count.minor_scratches == 0
        assert result.defect_count.major_scratches == 0

    def test_minor_scratches_within_a1(self, yolo_class_names, image_size):
        """Test screen with 2 minor scratches (A1 threshold)"""
        detection = MockResults(
            boxes=[
                create_mock_box(3, 0.95, 480, 480, 50, 5),  # Minor scratch 1
                create_mock_box(3, 0.90, 500, 500, 40, 4)  # Minor scratch 2
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.A1
        assert result.defect_count.minor_scratches == 2
        assert result.defect_count.major_scratches == 0

    def test_minor_scratches_a2_threshold(self, yolo_class_names, image_size):
        """Test screen with 4 minor scratches (A2 threshold)"""
        detection = MockResults(
            boxes=[
                create_mock_box(3, 0.95, 480, 480, 50, 5),
                create_mock_box(3, 0.94, 500, 500, 40, 4),
                create_mock_box(3, 0.93, 520, 520, 45, 4),
                create_mock_box(3, 0.92, 540, 540, 42, 5)
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.A2
        assert result.defect_count.minor_scratches == 4
        assert result.defect_count.major_scratches == 0

    def test_major_scratches_a3_threshold(self, yolo_class_names, image_size):
        """Test screen with 2 major scratches (A3 threshold)"""
        detection = MockResults(
            boxes=[
                create_mock_box(2, 0.95, 480, 480, 100, 10),  # Major scratch 1
                create_mock_box(2, 0.94, 520, 520, 95, 12)  # Major scratch 2
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.A3
        assert result.defect_count.major_scratches == 2
        assert result.defect_count.minor_scratches == 0

    def test_c1_grade_combined_defects(self, yolo_class_names, image_size):
        """Test screen with combination of defects triggering C1"""
        detection = MockResults(
            boxes=[
                create_mock_box(2, 0.95, 480, 480, 100, 10),  # Major scratch 1
                create_mock_box(2, 0.94, 520, 520, 95, 12),  # Major scratch 2
                create_mock_box(2, 0.93, 560, 560, 98, 11),  # Major scratch 3
                create_mock_box(3, 0.92, 600, 600, 40, 4)  # Minor scratch
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.C1
        assert result.defect_count.major_scratches == 3
        assert "Exceeds A3 criteria" in result.details.lower()

    def test_cracked_screen_with_scratches(self, yolo_class_names, image_size):
        """Test cracked screen should be R2 regardless of other defects"""
        detection = MockResults(
            boxes=[
                create_mock_box(0, 0.98, 480, 480, 200, 200),  # Screen crack
                create_mock_box(3, 0.95, 520, 520, 40, 4),  # Minor scratch
                create_mock_box(2, 0.94, 560, 560, 100, 10)  # Major scratch
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.R2
        assert "cracked" in result.details.lower()

    def test_low_confidence_detections(self, yolo_class_names, image_size):
        """Test handling of low confidence detections"""
        detection = MockResults(
            boxes=[
                create_mock_box(3, 0.35, 480, 480, 40, 4),  # Low confidence minor scratch
                create_mock_box(2, 0.45, 520, 520, 95, 12)  # Low confidence major scratch
            ],
            orig_shape=(image_size[0], image_size[1], 3),
            names=yolo_class_names
        )

        grader = ScreenGrading()
        result = grader.calculate_grade(detection)

        assert result.grade == Grade.A1
        assert result.defect_count.minor_scratches == 0
        assert result.defect_count.major_scratches == 0


def test_relative_box_sizes():
    """Test that box dimensions are being handled correctly relative to image size"""
    IMAGE_SIZE = 960
    grader = ScreenGrading()

    detection = MockResults(
        boxes=[create_mock_box(2, 0.95, 480, 480, IMAGE_SIZE * 0.1, IMAGE_SIZE * 0.01)],
        orig_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        names={2: 'Screen_Major_scratch'}
    )

    result = grader.calculate_grade(detection)
    assert result.grade == Grade.A2
    assert result.defect_count.major_scratches == 1