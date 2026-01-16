import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from scipy.spatial.distance import pdist
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/sabainaharoon/Documents/compasia/ai_grading')))

# Define necessary mock classes if they aren't already imported
class DevicePosition(Enum):
    DEVICE_FRONT_INACTIVE = "front"
    DEVICE_BACK = "back"
    DEVICE_LEFT = "left"
    DEVICE_RIGHT = "right"
    DEVICE_TOP = "top"
    DEVICE_BOTTOM = "bottom"


def calculate_size_features(boxes, image_shape):
    """Calculate size-based features for defects, avoiding NaN values."""

    if not boxes:
        return {
            'avg_defect_size': 0,
            'size_variance': 0,
            'total_affected_area': 0
        }

    areas = []
    confidences = []
    total_image_area = image_shape[0] * image_shape[1]

    for box in boxes:
        x, y, w, h = box.xywh[0]
        area = w * h
        conf = float(box.conf)  # Extract confidence score

        areas.append(area / total_image_area)
        confidences.append(conf)

    areas = np.array(areas)
    confidences = np.array(confidences)

    # Avoid division by zero
    if np.sum(confidences) == 0:
        return {
            'avg_defect_size': 0,
            'size_variance': 0,
            'total_affected_area': np.sum(areas)
        }

    return {
        'avg_defect_size': np.average(areas, weights=confidences),
        'size_variance': np.average((areas - np.average(areas, weights=confidences)) ** 2, weights=confidences),
        'total_affected_area': np.sum(areas)
    }


def calculate_spatial_features(boxes, image_shape):
    """Calculate spatial distribution features for defects, handling all edge cases."""

    if len(boxes) < 2:
        return {
            'avg_distance_between_defects': 0,
            'clustering_coefficient': 0,
            'center_vs_edge_ratio': 0
        }

    # Get centers of all bounding boxes
    centers = []
    confidences = []  # Store confidence scores for weighting calculations

    for box in boxes:
        x, y, w, h = box.xywh[0]
        conf = float(box.conf)  # Extract confidence score
        centers.append([x,y])
        confidences.append(conf)

    centers = np.array(centers)
    confidences = np.array(confidences)  # Convert to NumPy array

    # Calculate pairwise distances
    distances = pdist(centers)

    # Handle the case with exactly 2 defects (distances will be a single value)
    if len(boxes) == 2:
        avg_distance = distances[0] / np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)  # Normalize
        # For 2 defects, the std is undefined with ddof=1, so set clustering_coeff to a default
        clustering_coeff = 0
    else:
        # Original code for 3+ defects
        avg_distance = np.mean(distances) / np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)
        clustering_coeff = np.std(distances, ddof=1) / np.sqrt(image_shape[0] ** 2 + image_shape[1] ** 2)

    # Calculate center vs edge ratio
    image_center = np.array([image_shape[1] / 2, image_shape[0] / 2])
    distances_to_center = np.linalg.norm(centers - image_center, axis=1)

    # Normalize distances using image diagonal
    max_possible_distance = np.linalg.norm(image_center)

    center_defects = np.sum(distances_to_center < max_possible_distance / 2)
    edge_defects = len(centers) - center_defects

    # Avoid division errors and apply log transformation to smooth the scale
    center_edge_ratio = np.log1p(center_defects) / (np.log1p(edge_defects) + 1e-6)

    return {
        'avg_distance_between_defects': avg_distance,
        'clustering_coefficient': clustering_coeff,
        'center_vs_edge_ratio': center_edge_ratio
    }


class HousingGradePredictor:
    def __init__(self):
        """
        Initialize the housing grade predictor

        Args:
            model_path: Path to the trained model file
            scaler_path: Path to the feature scaler
            label_encoder_path: Path to the label encoder
            cat_encoder_path: Path to the categorical encoder
        """
        model_path = '/home/ubuntu/ai-grading-uat/grading_system/tree_models/rf_housing_classifier.joblib'
        scaler_path = '/home/ubuntu/ai-grading-uat/grading_system/tree_models/feature_scaler.joblib'
        label_encoder_path = '/home/ubuntu/ai-grading-uat/grading_system/tree_models/label_encoder.joblib'
        cat_encoder_path = '/home/ubuntu/ai-grading-uat/grading_system/tree_models/cat_encoder.joblib'
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.cat_encoder = joblib.load(cat_encoder_path)

    def extract_features(self, detection_results: Dict[DevicePosition, None]) -> Tuple[Dict, int]:
        """
        Extract features from detection results

        Args:
            detection_results: Dictionary of YOLOv8 results for each side

        Returns:
            Dictionary of features for classification
        """
        # Map enum to string for consistent feature naming
        position_mapping = {
            "DEVICE_BACK": 'back',
            "DEVICE_LEFT": 'left',
            "DEVICE_RIGHT": 'right',
            "DEVICE_TOP": 'top',
            "DEVICE_BOTTOM": 'bottom',
        }
        class_mapping = {0: 'DentDing',
 1: 'Discoloration',
 10: 'cover_spot',
 12: 'dense_scratch',
 14: 'heavy_discoloration',
 15: 'light_discoloration',
 19: 'scratch',
 21: 'sticker_residual',
 22: 'unk_defect'}

        # Initialize defect counts for all defects per side
        # defect_counts = {}

        defect_counts = {
            'scratch_back': 0, 'dense_scratch_back': 0, 'cover_spot_back': 0, "unk_defect_back": 0,
            "Discoloration_back": 0,
            'scratch_left': 0, 'dense_scratch_left': 0, 'cover_spot_left': 0, 'light_discoloration_left': 0,
            'Discoloration_left': 0, 'heavy_discoloration_left': 0, 'DentDing_left': 0, "unk_defect_left": 0,
            'scratch_right': 0, 'dense_scratch_right': 0, 'cover_spot_right': 0, 'light_discoloration_right': 0,
            'Discoloration_right': 0, 'heavy_discoloration_right': 0, 'DentDing_right': 0, "unk_defect_right": 0,
            'scratch_top': 0, 'dense_scratch_top': 0, 'cover_spot_top': 0, 'light_discoloration_top': 0,
            'Discoloration_top': 0, 'heavy_discoloration_top': 0, 'DentDing_top': 0, "unk_defect_top": 0,
            'scratch_bottom': 0, 'dense_scratch_bottom': 0, 'cover_spot_bottom': 0, 'light_discoloration_bottom': 0,
            'Discoloration_bottom': 0, 'heavy_discoloration_bottom': 0, 'DentDing_bottom': 0, "unk_defect_bottom": 0,
            'sticker_residual_back': 0, 'sticker_residual_left': 0, 'sticker_residual_right': 0,
            'sticker_residual_top': 0, 'sticker_residual_bottom': 0
        }
        cs_count = 0
        # Additional spatial and quadrant features
        for side in ['back', 'left', 'right', 'top', 'bottom']:
            defect_counts.update({
                f'avg_distance_between_defects_{side}': 0,
                f'clustering_coefficient_{side}': 0,
                f'center_vs_edge_ratio_{side}': 0,
                f'avg_defect_size_{side}': 0,
                f'size_variance_{side}': 0,
                f'total_affected_area_{side}': 0,
                f'top_left_defects_{side}': 0,
                f'top_right_defects_{side}': 0,
                f'bottom_left_defects_{side}': 0,
                f'bottom_right_defects_{side}': 0,
                f'most_affected_quadrant_{side}': '',  # Store most affected quadrant
            })

        # Process each side's detection results
        for position, result in detection_results.items():
            if result is None:
                continue

            # Get side string
            side = position_mapping[position.name]

            # Calculate spatial and size features
            spatial_features = calculate_spatial_features(result.boxes, result.orig_shape)
            size_features = calculate_size_features(result.boxes, result.orig_shape)

            # Update features for this side
            for feature_name, value in spatial_features.items():
                defect_counts[f'{feature_name}_{side}'] = value

            for feature_name, value in size_features.items():
                defect_counts[f'{feature_name}_{side}'] = value

            # Calculate quadrant-based distribution
            height, width = result.orig_shape
            quadrants = {'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0}

            # Process each detected defect
            for box in result.boxes:
                class_id = int(box.cls)

                x, y, w, h = box.xywh[0]
                area = w * h
                confidence = float(box.conf)

                # class_name = model_back.names[class_id]
                # find class mapping , if not found continue
                class_name = class_mapping.get(class_id)

                if class_name is None:
                    continue

                defect_key = f"{class_name}_{side}"

                if defect_key == 'cover_spot_back':
                    cs_count += 1


                # # Normalize defect count by area percentage
                normalized_defect = (area.item() / (width * height)) * 100

                # # Weight by confidence score
                defect_counts[defect_key] += normalized_defect * confidence

                # Quadrant calculation
                center_x, center_y = x , y
                if center_x < width / 2:
                    if center_y < height / 2:
                        quadrants['top_left'] += 1
                    else:
                        quadrants['bottom_left'] += 1
                else:
                    if center_y < height / 2:
                        quadrants['top_right'] += 1
                    else:
                        quadrants['bottom_right'] += 1
            # Update quadrant counts & determine most affected quadrant
            if sum(quadrants.values()) > 0:
                max_quadrant = max(quadrants, key=quadrants.get)
                for quadrant, count in quadrants.items():
                    defect_counts[f'{quadrant}_defects_{side}'] = count
                defect_counts[f'most_affected_quadrant_{side}'] = max_quadrant

        # # Compute defect density for each side
        # for side in ['back', 'left', 'right', 'top', 'bottom']:
        #     total_area = defect_counts[f'total_affected_area_{side}']
        #     max_area = 1.0  # Normalized max area
        #     defect_counts[f'defect_density_{side}'] = total_area / max_area
        #
        # # Set max_defect_density
        # defect_counts['max_defect_density'] = max([
        #     defect_counts[f'defect_density_{side}']
        #     for side in ['back', 'left', 'right', 'top', 'bottom', 'front']
        # ])

        return defect_counts, cs_count

    def predict(self, detection_results: Dict[DevicePosition, None]) -> Tuple[str, float, int]:
        """
        Predict housing grade from detection results

        Args:
            detection_results: Dictionary of YOLOv8 results for each side

        Returns:
            Tuple of (predicted grade, confidence)
        """
        # Extract features
        features, cs_count = self.extract_features(detection_results)

        # Convert to DataFrame
        features_df = pd.DataFrame([features])

        # Prepare feature data by encoding categorical variables
        quadrant_columns = [col for col in features_df.columns if 'most_affected_quadrant' in col]
        for col in quadrant_columns:
            if col in features_df.columns:
                # Handle potential unseen categories
                try:
                    features_df[col] = self.cat_encoder.transform(features_df[col])
                except:
                    # If category not seen during training, use a safe default
                    features_df[col] = 0

        # Handle missing values
        features_df.fillna(0, inplace=True)

        # Ensure all expected features are present
        missing_features = set(self.scaler.feature_names_in_) - set(features_df.columns)
        print(missing_features, " missing features")
        for feature in missing_features:
            features_df[feature] = 0

        # Select only the features expected by the model
        features_df = features_df[self.scaler.feature_names_in_]

        # Standardize features
        features_scaled = self.scaler.transform(features_df)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]

        # Get confidence
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction]

        # Convert prediction to grade
        grade = self.label_encoder.inverse_transform([prediction])[0]

        return grade, confidence, cs_count


