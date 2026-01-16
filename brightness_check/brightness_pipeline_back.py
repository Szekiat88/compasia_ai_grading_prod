import os
import numpy as np
import cv2
import joblib
from PIL import Image
from typing import Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrightnessFeatureExtractor:
    """
    Feature extractor for brightness classification.
    Extracts the same features used during training.
    """
    
    @staticmethod
    def perceived_luminance(np_rgb: np.ndarray) -> float:
        """Calculate perceived luminance using standard weights"""
        return (0.2126 * np_rgb[..., 0] + 0.7152 * np_rgb[..., 1] + 0.0722 * np_rgb[..., 2]).mean()
    
    @staticmethod
    def bright_pixel_ratio(np_gray: np.ndarray, threshold: int = 200) -> float:
        """Calculate ratio of bright pixels above threshold"""
        return np.sum(np_gray > threshold) / np_gray.size
    
    @staticmethod
    def highlight_pixel_ratio(np_gray: np.ndarray, threshold: int = 240) -> float:
        """Calculate ratio of highlight pixels above threshold"""
        return np.sum(np_gray > threshold) / np_gray.size
    
    @staticmethod
    def top_bottom_half_means(np_gray: np.ndarray) -> Tuple[float, float, float]:
        """Calculate top half, bottom half means and their absolute difference"""
        H = np_gray.shape[0]
        top = np_gray[:H//2, :].mean()
        bottom = np_gray[H//2:, :].mean()
        return top, bottom, abs(top - bottom)
    
    @staticmethod
    def avg_rgb_intensity(np_rgb: np.ndarray) -> float:
        """Calculate average RGB intensity"""
        return (np_rgb[..., 0].mean() + np_rgb[..., 1].mean() + np_rgb[..., 2].mean()) / 3
    
    @staticmethod
    def std_gray(np_gray: np.ndarray) -> float:
        """Calculate standard deviation of grayscale values"""
        return np.std(np_gray)
    
    @staticmethod
    def midrange_ratio(np_gray: np.ndarray, low: int = 150, high: int = 200) -> float:
        """Calculate ratio of pixels in midrange"""
        return np.sum((np_gray >= low) & (np_gray <= high)) / np_gray.size
    
    @staticmethod
    def highlight_cluster_count(np_gray: np.ndarray, threshold: int = 240) -> int:
        """Count number of highlight clusters using contour detection"""
        binary = (np_gray > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    
    def extract_features(self, image: Union[str, np.ndarray, Image.Image]) -> list:
        """
        Extract all features from an image and return as ordered list.
        
        Args:
            image: Can be a file path, numpy array, or PIL Image
            
        Returns:
            List of feature values in the correct order for prediction
        """
        # Handle different input types
        if isinstance(image, str):
            image_rgb = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = Image.fromarray(image)
            elif len(image.shape) == 2:
                image_rgb = Image.fromarray(image).convert("RGB")
            else:
                raise ValueError(f"Unsupported image array shape: {image.shape}")
        elif isinstance(image, Image.Image):
            image_rgb = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to numpy arrays
        np_rgb = np.array(image_rgb)
        np_gray = np.array(image_rgb.convert("L"))
        
        # Extract all features
        lum = self.perceived_luminance(np_rgb)
        bratio = self.bright_pixel_ratio(np_gray)
        hratio = self.highlight_pixel_ratio(np_gray)
        top, bottom, imbalance = self.top_bottom_half_means(np_gray)
        rgb_avg = self.avg_rgb_intensity(np_rgb)
        gray_std = self.std_gray(np_gray)
        midrange = self.midrange_ratio(np_gray)
        hclusters = self.highlight_cluster_count(np_gray)
        
        # Create derived feature (as done in training)
        top_minus_bottom_mean = top - bottom
        
        # Return features in the same order as training data
        return [
            lum,                    # perceived_luminance
            rgb_avg,               # avg_rgb_intensity
            bratio,                # bright_pixel_ratio
            hratio,                # highlight_pixel_ratio
            imbalance,             # shadow_imbalance
            gray_std,              # std_gray
            midrange,              # midrange_ratio
            hclusters,             # highlight_cluster_count
            top_minus_bottom_mean  # top_minus_bottom_mean
        ]


class BrightnessClassifier:
    """
    Brightness classification pipeline that loads model and predicts image brightness quality.
    """
    
    def __init__(self, model_path = "/home/ubuntu/ai-grading-uat/models/xgboost_brightness_model_housing.joblib"):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved XGBoost model (.joblib file)
        """
        self.model = joblib.load(model_path)
        self.feature_extractor = BrightnessFeatureExtractor()
        logger.info(f"Successfully loaded model from {model_path}")
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> bool:
        """
        Predict brightness quality of an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            True if desired brightness, False if underlit
        """
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        # Create feature vector for prediction
        feature_vector = np.array([features])
        
        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        
        # Return boolean (True = desired brightness, False = underlit)
        return bool(prediction)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    # MODEL_PATH = "/home/ubuntu/ai-grading-uat/models/xgboost_brightness_model_housing.joblib"
    classifier = BrightnessClassifier()
    
    # Simple prediction
    image_path = "/home/ubuntu/ai-grading-uat/brightness_check/test_images/02d7385d-c21e-43cb-816f-70d107bedaea_INPUT_BACK_1742206265640.png"
    image_path = "/home/ubuntu/ai-grading-uat/brightness_check/test_images/0ad9f931-4159-4312-87b6-0777aeac7a61_INPUT_BACK_1740467918732.png"
    img = cv2.imread(image_path)
    is_good_brightness = classifier.predict(img)
    
    print(f"Image has good brightness: {is_good_brightness}")