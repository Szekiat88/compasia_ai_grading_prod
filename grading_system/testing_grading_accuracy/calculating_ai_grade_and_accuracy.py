import pandas as pd
import os
import sys

# Add the path of grading_system to PYTHONPATH
sys.path.append(os.path.abspath("/Users/sabainaharoon/Documents/compasia/ai_grading/grading_system"))
from main import grade_device  # Import the grading pipeline function
from core.enums import DevicePosition

# Define paths
organized_images_folder = './organized_cleaned_images'  # Folder containing session-wise images
cleaned_csv_file = '/Users/sabainaharoon/Documents/compasia/ai_grading/grading_system/data/cleaned_file_14thjan_dbdata_datefix.csv'  # Cleaned CSV file
# model_path = "/Users/sabainaharoon/Documents/compasia/ai_grading/defect_detection/training/runs/detect/yolov8_960/best_5Dec_960.pt"
model_path = "/Users/sabainaharoon/Documents/compasia/ai_grading/defect_detection/training/runs/detect/yolov8s_v4/weights/best.pt"
output_csv = './grading_result_800model.csv'  # Output file for the grading results

# Load the cleaned CSV
df_cleaned = pd.read_csv(cleaned_csv_file)

# Initialize a list to store results
results = []

# Iterate through each session_id
for _, row in df_cleaned.iterrows():
    session_id = row['session_id']
    qc_screen_grade = row['qc_screen_grading']
    qc_housing_grade = row['qc_housing_grading']

    # Locate session folder
    session_folder = os.path.join(organized_images_folder, session_id)
    if not os.path.exists(session_folder):
        print(f"Session folder not found: {session_folder}")
        continue

    # Collect images for grading
    image_paths = {}
    for file_name in os.listdir(session_folder):
        if "front" in file_name.lower():
            image_paths[DevicePosition.DEVICE_FRONT_INACTIVE] = os.path.join(session_folder, file_name)
        elif "back" in file_name.lower():
            image_paths[DevicePosition.DEVICE_BACK] = os.path.join(session_folder, file_name)
        elif "left" in file_name.lower():
            image_paths[DevicePosition.DEVICE_LEFT] = os.path.join(session_folder, file_name)
        elif "right" in file_name.lower():
            image_paths[DevicePosition.DEVICE_RIGHT] = os.path.join(session_folder, file_name)
        elif "top" in file_name.lower():
            image_paths[DevicePosition.DEVICE_TOP] = os.path.join(session_folder, file_name)
        elif "bottom" in file_name.lower():
            image_paths[DevicePosition.DEVICE_BOTTOM] = os.path.join(session_folder, file_name)

    # Ensure all six images are available
    if len(image_paths) < 6:
        print(f"Not all six images are available for session: {session_id}")
        continue

    try:
        # Perform grading
        grading_results = grade_device(
            model_path=model_path,
            image_paths=image_paths,
            device_model=None,  # Optional: Device model
            dimensions_csv=None  # Optional: Path to dimensions CSV
        )

        # Append results to the list
        results.append({
            'session_id': session_id,
            'qc_screen_grading': qc_screen_grade,
            'qc_housing_grading': qc_housing_grade,
            'ai_screen_grading': grading_results['screen']['grade'],
            'ai_housing_grading': grading_results['housing']['overall_grade'],
            # 'screen_grade_match': qc_screen_grade == grading_results['screen']['grade'],
            # 'housing_grade_match': qc_housing_grade == grading_results['housing']['overall_grade'],
        })

    except Exception as e:
        print(f"Error during grading for session_id {session_id}: {str(e)}")

# Save results to a new CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv, index=False)

print(f"Grading process complete. Results saved to: {output_csv}")