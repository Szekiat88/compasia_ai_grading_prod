import cv2
import numpy as np
import joblib
from skimage import measure
from scipy.stats import entropy
from skimage.filters import sobel
import pandas as pd

# Feature extractor (same as your training)
def extract_combined_features_from_img(img):
    try:
        if isinstance(img, str):  # path case
            img = cv2.imread(img)
        else:
            print(type(img))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_gray = gray / 255.0

        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        min_brightness = np.min(gray)
        max_brightness = np.max(gray)
        brightness_ratio = np.sum(gray > 127) / gray.size
        exposure = mean_brightness / 255.0

        specular_mask = gray > 240
        specular_ratio = np.sum(specular_mask) / gray.size
        glare_labels = measure.label(specular_mask.astype(np.uint8))
        glare_patch_count = len(np.unique(glare_labels)) - 1

        h, w = gray.shape
        center = gray[h//4:h*3//4, w//4:w*3//4]
        std_dark_region = np.std(center)

        sobel_edges = sobel(norm_gray)
        low = (sobel_edges > 0.1) & (sobel_edges <= 0.3)
        high = sobel_edges > 0.3
        edge_contrast_score = np.sum(high) / (np.sum(low) + 1e-6)

        bright_pixels = gray[gray > 200]
        if bright_pixels.size > 0:
            hist = np.histogram(bright_pixels, bins=10, range=(200, 255))[0]
            white_pixel_entropy = entropy(hist + 1e-5)
        else:
            white_pixel_entropy = 0.0

        high_reflection_area = np.sum(gray > 220)

        return [
            mean_brightness, std_brightness, min_brightness, max_brightness,
            brightness_ratio, exposure, specular_ratio, glare_patch_count,
            std_dark_region, edge_contrast_score, white_pixel_entropy,
            high_reflection_area
        ]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0]*12  # fallback dummy values

# Load model and predict
def predict_screen_brightness(image_path_or_array, 
model_path = "/home/ubuntu/ai-grading-uat/models/rf_screen_brightness_model.joblib"):
    model = joblib.load(model_path)
    features = extract_combined_features_from_img(image_path_or_array)

    # Same order as during training
    feature_names = [
    "mean_brightness", "std_brightness", "min_brightness", "max_brightness",
    "brightness_ratio", "exposure", "specular_ratio", "glare_patch_count",
    "std_dark_region", "edge_contrast_score", "white_pixel_entropy",
    "high_reflection_area"
    ]

    features_df = pd.DataFrame([features], columns=feature_names)
    prediction = model.predict(features_df)[0]
    confidence = model.predict_proba(features_df)[0][prediction]

    # prediction = model.predict([features])[0]
    # confidence = model.predict_proba([features])[0][prediction]
    return prediction, confidence

# Example usage
if __name__ == "__main__":
    image_path = "/home/ubuntu/ai-grading-uat/brightness_check/test_images/e96acc25-a927-4e59-b124-baffce85aeb5_INPUT_FRONT_1742272143967.png"
    image_path = "/home/ubuntu/ai-grading-uat/brightness_check/test_images/c14f1d1a-3d36-4b5f-8b60-389185a56fd8_INPUT_FRONT_1748071743644.png"
    image_path = "/home/ubuntu/ai-grading-uat/brightness_check/test_images/b7b9634f-9563-4bfd-a719-9b80060d807e_INPUT_FRONT_1740985650276.png"
    img = cv2.imread(image_path)
    model_path = "/home/ubuntu/ai-grading-uat/models/rf_screen_brightness_model.joblib"
    label, conf = predict_screen_brightness(img)
    print("Prediction:", "DESIRED" if label == 1 else "UNDESIRED")
    print("Confidence:", round(conf * 100, 2), "%")