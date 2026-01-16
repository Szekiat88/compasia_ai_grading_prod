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
    Extracts the same features used during training including new robust features.
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
    
    # NEW ROBUST FEATURES FOR GLOSSY SURFACES
    @staticmethod
    def local_contrast_score(np_gray: np.ndarray) -> float:
        """Local contrast using Laplacian - works better for glossy surfaces"""
        try:
            laplacian = cv2.Laplacian(np_gray, cv2.CV_64F)
            return laplacian.var()
        except:
            return 0.0
    
    @staticmethod
    def reflection_normalized_balance(np_gray: np.ndarray) -> float:
        """Top-bottom balance excluding extreme reflections"""
        try:
            # Mask out extreme highlights (reflections)
            mask = np_gray < np.percentile(np_gray, 95)
            masked_gray = np_gray.copy()
            masked_gray[~mask] = np.median(np_gray[mask])
            
            H = masked_gray.shape[0]
            top = masked_gray[:H//2, :].mean()
            bottom = masked_gray[H//2:, :].mean()
            return abs(top - bottom)
        except:
            return abs(np_gray[:np_gray.shape[0]//2, :].mean() - np_gray[np_gray.shape[0]//2:, :].mean())
    
    @staticmethod
    def detail_visibility_score(np_gray: np.ndarray) -> float:
        """Measures how well details are visible despite reflections"""
        try:
            # Use gradient magnitude to assess detail visibility
            grad_x = cv2.Sobel(np_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(np_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            return np.percentile(gradient_magnitude, 75)
        except:
            return 0.0
    
    @staticmethod
    def surface_glossiness_indicator(np_gray: np.ndarray) -> float:
        """Detects if surface is glossy (high dynamic range)"""
        try:
            very_bright = np.sum(np_gray > 250) / np_gray.size
            very_dark = np.sum(np_gray < 50) / np_gray.size
            return very_bright + very_dark
        except:
            return 0.0
    
    @staticmethod
    def reflection_area_ratio(np_gray: np.ndarray, threshold: int = 245) -> float:
        """Ratio of pixels in extreme highlights (reflections)"""
        return np.sum(np_gray > threshold) / np_gray.size
    
    @staticmethod
    def non_reflection_brightness(np_gray: np.ndarray) -> float:
        """Mean brightness excluding reflection areas"""
        try:
            # Exclude extreme highlights
            non_reflection_pixels = np_gray[np_gray <= 245]
            if len(non_reflection_pixels) > 0:
                return non_reflection_pixels.mean()
            else:
                return np_gray.mean()
        except:
            return np_gray.mean()
    
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
        
        # Extract original features
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
        
        # Extract NEW robust features
        local_contrast = self.local_contrast_score(np_gray)
        ref_norm_balance = self.reflection_normalized_balance(np_gray)
        detail_score = self.detail_visibility_score(np_gray)
        glossiness = self.surface_glossiness_indicator(np_gray)
        ref_area = self.reflection_area_ratio(np_gray)
        non_ref_brightness = self.non_reflection_brightness(np_gray)
        
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
            top_minus_bottom_mean, # top_minus_bottom_mean
            local_contrast,        # local_contrast
            ref_norm_balance,      # reflection_normalized_balance
            detail_score,          # detail_visibility_score
            glossiness,            # surface_glossiness
            ref_area,              # reflection_area_ratio
            non_ref_brightness     # non_reflection_brightness
        ]


class BrightnessClassifier:
    """
    Brightness classification pipeline that loads model and predicts image brightness quality.
    """
    
    def __init__(self, model_path="/home/ubuntu/ai-grading-uat/models/xgboost_brightness_model_housing_new.joblib"):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved model (.joblib file)
        """
        self.model = joblib.load(model_path)
        self.feature_extractor = BrightnessFeatureExtractor()
        logger.info(f"Successfully loaded model from {model_path}")
    
    def predict(self, image: Union[str, np.ndarray, Image.Image], debug: bool = False) -> bool:
        """
        Predict brightness quality of an image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            debug: If True, prints feature values for debugging
            
        Returns:
            True if desired brightness, False if underlit
        """
        # Extract features
        features = self.feature_extractor.extract_features(image)
        
        # Create feature vector for prediction
        feature_vector = np.array([features])
        
        if debug:
            feature_names = [
                'perceived_luminance', 'avg_rgb_intensity', 'bright_pixel_ratio',
                'highlight_pixel_ratio', 'shadow_imbalance', 'std_gray',
                'midrange_ratio', 'highlight_cluster_count', 'top_minus_bottom_mean',
                'local_contrast', 'reflection_normalized_balance', 'detail_visibility_score',
                'surface_glossiness', 'reflection_area_ratio', 'non_reflection_brightness'
            ]
            print("Feature values:")
            for name, value in zip(feature_names, features):
                print(f"  {name}: {value:.3f}")
            print(f"Feature vector shape: {feature_vector.shape}")
        
        # Make prediction
        prediction = self.model.predict(feature_vector)[0]
        
        # Get prediction probability if available
        try:
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = max(probabilities)
            if debug:
                print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        except:
            if debug:
                print(f"Prediction: {prediction}")
        
        # Return boolean (True = desired brightness, False = underlit)
        return bool(prediction)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    MODEL_PATH = "/Users/sabainaharoon/Downloads/brightness_check/xgboost_brightness_model_housing.joblib"
    MODEL_PATH = "/Users/sabainaharoon/Downloads/brightness_check/rf_back_brightness_model.joblib"
    MODEL_PATH = "/Users/sabainaharoon/Downloads/brightness_check/xgboost_brightness_model_housing_new.joblib"
    classifier = BrightnessClassifier()
    
    # Test images
    test_images = [
        "/Users/sabainaharoon/Downloads/brightness_check/0b296aa0-91a6-4ab5-9ed0-bcf10c378c60_INPUT_BACK_1745660598522.png",
        "/Users/sabainaharoon/Downloads/brightness_check/undesired/0a74fac8-2028-4a98-9b05-4c9d2487f2eb_INPUT_BACK_1741750359982.png",
        "/Users/sabainaharoon/Downloads/brightness_check/desired/0a88e1e4-6c1b-48d8-bc8c-11719dcb02f0_INPUT_BACK_1740463696693.png",
        "/Users/sabainaharoon/Downloads/db028a72-55db-48fd-b832-be7740d71925_normal_INPUT_BACK_CAM109.21_PROC96.63_1751440548143.png",
        "/Users/sabainaharoon/Downloads/db028a72-55db-48fd-b832-be7740d71925_normal_INPUT_BACK_CAM113.39_PROC94.85_1751440733058.png.jpeg",
        "/Users/sabainaharoon/Downloads/db028a72-55db-48fd-b832-be7740d71925_normal_INPUT_BACK_CAM119.67_PROC92.74_1751440774931.png.jpeg"
    ]
    
    # Test each image
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting: {os.path.basename(image_path)}")
            is_good_brightness = classifier.predict(image_path, debug=True)
            print(f"Result: {'✅ GOOD BRIGHTNESS' if is_good_brightness else '❌ POOR BRIGHTNESS'}")
        else:
            print(f"File not found: {image_path}")