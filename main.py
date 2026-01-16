# gunicorn -w 1 -b 0.0.0.0:5001 main:app
# lsof -i :5001
# kill -9 <pid>

from flask import Flask, request, jsonify
import psycopg2
import uuid
import logging
from dotenv import load_dotenv
import os
import json
import boto3
import threading
import asyncio
from ultralytics import YOLO
import io
from PIL import Image
import datetime
import tempfile
import json
import numpy as np
import torchvision.transforms.functional as F
import psycopg2.extras  
from typing import List, Dict, Any, Optional
from grading_system import get_ai_grade
from brightness_check.screen_pipeline import predict_screen_brightness
from brightness_check.brightness_back_pipeline_new import BrightnessClassifier
import torch


# Load environment variables from the .env file
load_dotenv(dotenv_path='/home/ubuntu/ai-grading-uat/db.env')

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@app.route("/api/health", methods=["GET"])
def health():
    """
    Health check endpoint to confirm the API is running.
    """
    return jsonify({"status": "healthy", "message": "API is up and running!"}), 200

# SQS queue URL and ARN (replace these with your actual values)
SQS_QUEUE_URL = 'https://sqs.ap-southeast-1.amazonaws.com/046498959242/ai-grading-s3-prod'
SQS_REGION = 'ap-southeast-1'

# Create an SQS client using boto3
sqs_client = boto3.client('sqs', region_name=SQS_REGION)
s3_client = boto3.client('s3', region_name=SQS_REGION)

# Get database configuration from environment variables
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv('DB_PASSWORD'),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

screen_defect_threshold = 7
back_defect_threshold = 12

# model
imgsz = 960
model_path = "/home/ubuntu/ai-grading-uat/best_5Dec_960.pt"
# model_back_path = "/home/ubuntu/ai-grading-uat/best_333_imagesModel.pt"
model_back_path = "/home/ubuntu/ai-grading-uat/back_6Aug.pt"

# model_tb_path = "/home/ubuntu/ai-grading-uat/tb_1280_300images.pt"
model_tb_path = "/home/ubuntu/ai-grading-uat/tb_best_yolov12.pt"
model_lr_path = "/home/ubuntu/ai-grading-uat/lr_1280_300images.pt"

model_front = YOLO(model_path)
model_topbottom = YOLO(model_tb_path)
model_leftright = YOLO(model_lr_path)
model_back = YOLO(model_back_path)

br_class_back = BrightnessClassifier()

# base url to return the plotted img url
base_url = "https://aigradingfiles.compasia.com/"

# Function to create a database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        return None

# Generic Function: Verify API Key and Return api_key_id
def verify_api_key(api_key, table_name="ai_api_key", column_name="api_key"):
    """
    Verify API Key in the database and return the corresponding api_key_id.

    :param api_key: The API key string to verify
    :param table_name: The table containing the API keys
    :param column_name: The column containing the API key values
    :return: api_key_id if valid, None otherwise
    """
    logger.info("Verifying API Key...")
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cur:
            query = f"SELECT api_key_id FROM {table_name} WHERE {column_name} = %s"
            cur.execute(query, (api_key,))
            result = cur.fetchone()
            if result:
                logger.info("API Key verified successfully.")
                return result[0]  # Return api_key_id
            logger.warning("Invalid API Key provided.")
            return None
    except Exception as e:
        logger.error(f"Error verifying API Key: {e}")
        return None
    finally:
        conn.close()
        logger.info("Database connection closed after API Key verification.")

# API Endpoint: Create New Diagnostics Session
# API Endpoint: Create New Diagnostics Session
@app.route("/api/diagnostics-session", methods=["POST"])
def create_session():
    # 1. Extract Headers and Body
    logger.info("Received request to create a new diagnostics session.")
    api_key = request.headers.get("x-api-key")
    data = request.get_json()
    imei = data.get("imei")

    if not api_key or not imei:
        logger.warning("Missing api_key header or imei in the request.")
        return jsonify({"error": "api_key header and imei are required"}), 400

    # 2. Verify API Key using Generic Function
    api_key_id = verify_api_key(api_key)
    if not api_key_id:
        logger.warning("Unauthorized access with an invalid API Key.")
        return jsonify({"error": "Invalid API Key"}), 401

    # 3. Insert into ai_session table (api_key_id, imei, status)
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to the database for session creation.")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn.cursor() as cur:
            # Generate the session_id and insert data
            insert_query = """
                INSERT INTO ai_session (api_key_id, imei, status, created_at, updated_at)
                VALUES (%s, %s, %s, NOW(), NOW())
                RETURNING session_id
            """
            cur.execute(insert_query, (api_key_id, imei, "Open"))
            session_id = cur.fetchone()[0]  # Fetch the generated session_id
            conn.commit()
            logger.info(f"New diagnostics session created successfully. Session ID: {session_id}")

            # Return session_id to mobile
            return jsonify({"session_id": session_id}), 201
    except Exception as e:
        logger.error(f"Database operation error during session creation: {e}")
        return jsonify({"error": "Failed to create session"}), 500
    finally:
        conn.close()
        logger.info("Database connection closed after session creation.")

# Function to extract session_id, input type, and image side from the object key
def parse_image_key(key):
    try:
        # Extract the part after the last "/" in the key
        filename = key.split("/")[-1]
        # Split the filename using "_"
        parts = filename.split("_")
        if len(parts) < 3:
            raise ValueError("Invalid URL format: Missing expected parts (session_id, input, side).")
        
        # Extract session_id, input, and side
        session_id = parts[0]
        input_type = parts[1]
        image_side = parts[2].split(".")[0]  # Remove file extension
        return session_id, input_type, image_side
    except Exception as e:
        logger.error(f"Error parsing key '{key}': {e}")
        return None

class LetterboxResize:
    def __init__(self, target_size, stride=32):
        self.target_size = target_size
        self.stride = stride
 
    def __call__(self, image):
        # Get original dimensions
        original_width, original_height = image.size
 
        # Calculate scaling factor
        scale = self.target_size / max(original_width, original_height)
 
        # Compute new dimensions (before stride adjustment)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
 
        # Adjust dimensions to be multiples of stride
        new_width = int(round(new_width / self.stride) * self.stride)
        new_height = int(round(new_height / self.stride) * self.stride)
 
        # Resize image
        image = F.resize(image, (new_height, new_width))
 
        # Return resized image
        return image

# Function to download an image from S3
def download_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        logger.error(f"Failed to download image from S3: {e}")
        return None


def upload_image_to_s3(bucket_name, key, image, s3_client):
    try:
        # Convert numpy array to a Pillow Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Get the image format (JPEG, PNG, etc.)
        img_format = image.format if image.format else "JPEG"  # Default to "JPEG" if no format found

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=f".{img_format.lower()}") as tmp_file:
            image.save(tmp_file.name, format=img_format)
            tmp_file.seek(0)
            
            # Upload the image to S3
            s3_client.upload_file(
                Filename=tmp_file.name,
                Bucket=bucket_name,
                Key=key,
                ExtraArgs={"ContentType": f"image/{img_format.lower()}"}
            )
            
            # Retrieve the versionId if versioning is enabled
            response = s3_client.head_object(Bucket=bucket_name, Key=key)
            version_id = response.get('VersionId', None)
            
            return {"bucket_name": bucket_name, "key": key, "version_id": version_id}
    
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        return None

def yolo_result_to_dict(results) -> Optional[List[Dict[str, Any]]]:
    """
    Convert YOLOv8 results to a dictionary format containing detection information.
    Args:
        results: List of YOLO prediction result objects
    Returns:
        A list of dictionaries with 'boxes', 'orig_shape', and 'names'.
    """
    if not results:
        return None

    all_results = []
    for r in results:  # Iterate through the list of result objects
        boxes = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                # Append individual detection box details
                boxes.append({
                    'cls': int(box.cls[0].item()),  # Class ID
                    'conf': float(box.conf[0].item()),  # Confidence
                    'xywh': [float(x) for x in box.xywh[0].tolist()]  # Bounding box coordinates (xywh format)
                })

        # Construct the dictionary for the current result
        result_dict = {
            'boxes': boxes,
            'orig_shape': r.orig_shape,  # Original image shape
            'names': r.names  # Class names
        }
        all_results.append(result_dict)

    return all_results


# Function to save inference results and S3 info into the database
def save_inference_results_to_db(session_id, image_side, uploaded_info, inference_results, db_connection):
    try:
        cursor = db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check for the current entry with session_id, image_side, active, and status in one query
        query = """
            SELECT session_id, image_side, active, status FROM ai_session_image
            WHERE session_id = %s AND image_side = %s AND active = 'True' AND status = 'Open' AND deleted_at IS NULL
        """
        cursor.execute(query, (session_id, image_side))
        existing_entry = cursor.fetchone()

        if existing_entry:
            # Proceed with the update since the session_id, image_side, active, and status conditions are met
            update_query = """
                UPDATE ai_session_image
                SET output_image_bucket_name = %s,
                    output_image_object_key = %s,
                    output_image_version_id = %s,
                    result_after_nms = %s,
                    status = %s,
                    updated_at = %s
                 WHERE session_id = %s 
                    AND image_side = %s
                    AND active = 'True' 
                    AND status = 'Open'
                    AND deleted_at IS NULL
            """
            
            utc_now = datetime.datetime.now(datetime.timezone.utc)

            cursor.execute(update_query, (
                uploaded_info['bucket_name'],
                uploaded_info['key'],
                uploaded_info.get('version_id', None),
                json.dumps(inference_results),  # Save results in JSON format
                "Completed",
                utc_now,
                session_id,
                image_side
            ))
            db_connection.commit()
            logger.info(f"Database updated successfully for session_id {session_id} and image_side {image_side}.")
        else:
            logger.warning(f"No entry found with session_id {session_id}, image_side {image_side}, active 'Active', and status 'Open'.")
    except Exception as e:
        cursor = db_connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check for the current entry with session_id, image_side, active, and status in one query
        query = """
            SELECT session_id, image_side, active, status FROM ai_session_image
            WHERE session_id = %s AND image_side = %s AND active = 'True' AND status = 'Open' AND deleted_at IS NULL
        """
        cursor.execute(query, (session_id, image_side))
        existing_entry = cursor.fetchone()
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            rows = cur.fetchall()
            # Update query - ai-session_image - status: failed
            update_query = """
                UPDATE ai_session_image
                SET status = %s,
                    updated_at = %s
                WHERE session_id = %s AND image_side = %s
            """

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(
                update_query,
                ( "Failed", utc_now, session_id, image_side)
            )
            db_connection.commit()
        logger.error(f"Error updating database: {e}")

# Updated function to poll messages from the SQS queue
async def poll_sqs():
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                AttributeNames=['All'],
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )

            if 'Messages' in response:
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle'] 
                message_body = message['Body']

                try:
                    message_data = json.loads(message_body)
                    records = message_data.get("Records", [])
                    for record in records:
                        key = record.get("s3", {}).get("object", {}).get("key")
                        bucket = record.get("s3", {}).get("bucket", {}).get("name")
                        version_id = record.get("s3", {}).get("object", {}).get("versionId")

                        if not key or not bucket or not version_id:
                            logger.error("Missing key, bucket, or versionId in message.")
                            continue

                        logger.info(f"Processing key: {key}")
                        result = parse_image_key(key)
                        if result:
                            session_id, input_type, image_side = result
                            logger.info(f"Extracted - Session ID: {session_id}, Input: {input_type}, Side: {image_side}")

                            conn = get_db_connection()
                            if conn:
                                try:
                                    with conn.cursor() as cur:
                                        # Step 1: Check if session_id exists in the ai_session table
                                        query = "SELECT 1 FROM ai_session WHERE session_id = %s"
                                        cur.execute(query, (session_id,))
                                        session_exists = cur.fetchone()

                                        if session_exists:
                                            # Step 2: Check if session_id and image_side already exist in ai_session_image table
                                            query = """
                                                SELECT 1 FROM ai_session_image 
                                                WHERE session_id = %s AND image_side = %s
                                            """
                                            cur.execute(query, (session_id, image_side))
                                            image_exists = cur.fetchone()

                                            if image_exists:
                                                # If record exists, update the `active` column to False and `status` to NULL
                                                update_query = """
                                                    UPDATE ai_session_image
                                                    SET active = False, status = NULL
                                                    WHERE session_id = %s AND image_side = %s
                                                """
                                                cur.execute(update_query, (session_id, image_side))
                                                logger.info(f"Existing record found. Marked as inactive for session_id {session_id} and image_side {image_side}.")
                                                
                                            # Step 3: Insert new record with status 'Open'
                                            insert_query = """
                                                INSERT INTO ai_session_image (session_id, image_side, input_image_bucket_name, input_image_object_key, input_image_version_id, status, created_at, updated_at)
                                                VALUES (%s, %s, %s, %s, %s, 'Open', NOW(), NOW())
                                            """
                                            cur.execute(insert_query, (session_id, image_side, bucket, key, version_id))
                                            conn.commit()
                                            logger.info(f"New record inserted with status 'Open' for session_id {session_id} and image_side {image_side}.")
                                        else:
                                            logger.warning(f"Session ID {session_id} does not exist. Skipping message.")
                                            # Stop the process if session_id doesn't exist
                                            return

                                except Exception as db_error:
                                    logger.error(f"Database error: {db_error}")
                                finally:
                                    conn.close()


                            image = download_image_from_s3(bucket, key)
                            if image:
                                # Resize the image for inference
                                # resize_transform = LetterboxResize(target_size=imgsz)
                                # resized_image = resize_transform(image)

                                # Select model based on image_side
                                if image_side in "FRONT":
                                    selected_model = model_front
                                elif image_side in ["TOP", "BOTTOM"]:
                                    selected_model = model_topbottom
                                elif image_side in ["LEFT", "RIGHT"]:
                                    selected_model = model_leftright
                                elif image_side in "BACK":
                                    selected_model = model_back
                                else:
                                    selected_model = model_front

                                # Class mapping
                                class_mapping = {
                                    0: "Screen_cracked",
                                    1: "Housing_cracked",
                                    2: "Screen_Major_scratch",
                                    3: "Screen_Minor_scratch",
                                    4: "Housing_Major_scratch",
                                    5: "Housing_Minor_scratch",
                                    6: "DentDing",
                                    7: "Discoloration"
                                }

                                 # Define per-class confidence thresholds
                                class_thresholds = {
                                    "Screen_cracked": 0.25,  # Lower threshold for Screen_cracked
                                    "Housing_cracked": 0.25,  # Lower threshold for Housing_cracked
                                    "Screen_Major_scratch": 0.25,
                                    "Screen_Minor_scratch": 0.25,
                                    "Housing_Major_scratch": 0.25,
                                    "Housing_Minor_scratch": 0.25,
                                    "DentDing": 0.25,
                                    "Discoloration": 0.20
                                }

                                # Perform inference
                                try:
                                    CRACK_CLASSES = [4, 7, 11, 13]  # Housing_cracked, Screen_cracked, dense_crack, hairline_crack
                                    OTHER_CLASSES_LR_TB = [0, 1, 10, 12, 14, 15, 19, 21, 22]
                                    LOW_CONF_THRESHOLD = 0.2
                                    HIGH_CONF_THRESHOLD_LR_TB = 0.3
                                    HIGH_CONF_THRESHOLD_BACK = 0.3
                                    if image_side == "FRONT":
                                        results = selected_model.predict(image, conf=0.35, iou=0.2)
                                    elif image_side == "BACK":
                                        # classes_back = [1, 10, 12, 19, 21, 22, 11, 13] # for back
                                        # results = selected_model.predict(image, classes=classes_back,conf=0.6,iou=0.5)
                                        classes_back=[1, 10, 12, 19, 21, 22, 11, 13, 4, 7]
                                        results = selected_model.predict(image, classes=classes_back,conf=LOW_CONF_THRESHOLD,iou=0.2)
                                        high_threshold = HIGH_CONF_THRESHOLD_BACK
                                        # print(results[0].boxes.data, "BACK result from main.py")
                                    else:
                                        classes_sides = [0,1,10, 12, 14, 15, 19, 21, 22] # for other sides
                                        # results = selected_model.predict(image, classes=classes_sides,conf=0.45,iou=0.35)
                                        results = selected_model.predict(image, classes=classes_sides,conf=LOW_CONF_THRESHOLD,iou=0.2)
                                        high_threshold = HIGH_CONF_THRESHOLD_LR_TB
                                    
                                    if image_side != "FRONT":
                                        filtered_boxes = []
                                        if results and len(results[0].boxes) > 0:
                                            for r in results[0].boxes:
                                                cls = int(r.cls.item())
                                                conf = r.conf.item()

                                                if cls in CRACK_CLASSES:
                                                    if conf >= LOW_CONF_THRESHOLD:
                                                        filtered_boxes.append(r)
                                                else:
                                                    if conf >= high_threshold:
                                                        filtered_boxes.append(r)

                                            if filtered_boxes:
                                                results[0].boxes.data = torch.stack([
                                                    torch.tensor(b.data) if isinstance(b.data, np.ndarray) else b.data
                                                    for b in filtered_boxes
                                                ])
                                            else:
                                                results[0].boxes.data = torch.empty((0, 6))

                                    if image_side == "FRONT":
                                        logger.info("Processing front image")
                                        filtered_results = []
                                        for result in results[0].boxes.data:
                                            x1, y1, x2, y2, conf, class_id = result.tolist()
                                            class_name = class_mapping.get(int(class_id))
                                            threshold = class_thresholds.get(class_name, 0.34)

                                            if conf >= threshold:
                                                filtered_results.append(result)

                                        results[0].boxes.data = np.array(filtered_results)

                                    inference_results = yolo_result_to_dict(results)

                                    # Add this block after: inference_results = yolo_result_to_dict(results)

                                    if image_side == "BACK":
                                        is_back_lit = br_class_back.predict(np.array(image))
                                        logger.info(f"is back lit {is_back_lit}")
                                        if inference_results and len(inference_results) > 0:
                                            # Add to the first (and typically only) result dict
                                            inference_results[0]['brightness_prediction_back'] = is_back_lit
                                        else:
                                            # If no YOLO detections, create a minimal structure with brightness info
                                            inference_results = [{
                                                'boxes': [],
                                                'orig_shape': image.size if hasattr(image, 'size') else None,
                                                'names': {},
                                                'brightness_prediction_back': is_back_lit
                                            }]

                                    if image_side == "FRONT":
                                        label, conf = predict_screen_brightness(np.array(image))
                                        is_screen_lit = True if label==1 else False
                                        logger.info(f"screenbrightness{is_screen_lit}")
                                        if inference_results and len(inference_results) > 0:
                                            # Add to the first (and typically only) result dict
                                            inference_results[0]['brightness_prediction_screen'] = is_screen_lit
                                        else:
                                            # If no YOLO detections, create a minimal structure with brightness info
                                            inference_results = [{
                                                'boxes': [],
                                                'orig_shape': image.size if hasattr(image, 'size') else None,
                                                'names': {},
                                                'brightness_prediction_screen': is_screen_lit
                                            }]

                                    # Plot the results to image
                                    # inference_results = []
                                    for r in results:
                                        im_bgr = r.plot()  # BGR-order numpy array
                                        im_rgb = Image.fromarray(im_bgr[..., ::-1])

                                    #     boxes = r.boxes.xyxy  # xyxy bounding box coordinates
                                    #     confidences = r.boxes.conf  # Confidence scores
                                    #     class_ids = r.boxes.cls  # Class IDs

                                    #     # Package the results in a list of dictionaries
                                    #     for i in range(len(boxes)):
                                    #         inference_results.append({
                                    #             "coordinates": boxes[i].tolist(),
                                    #             "confidence": confidences[i].item(),
                                    #             "class_id": class_ids[i].item()
                                    #         })

                                    # Upload the plotted image to S3
                                    output_image_key = f"output-image/{session_id}_OUTPUT_{image_side}.jpg"  # Update format if needed
                                    uploaded_info = upload_image_to_s3(bucket, output_image_key, im_rgb, s3_client)

                                    # Save inference results and S3 info to the database
                                    if uploaded_info:
                                        conn = get_db_connection()
                                        if conn:
                                            save_inference_results_to_db(session_id, image_side, uploaded_info, inference_results, conn)
                                            conn.close()
                                    # After successfully processing and saving results, delete the message from SQS
                                    sqs_client.delete_message(
                                        QueueUrl=SQS_QUEUE_URL,
                                        ReceiptHandle=receipt_handle
                                    )
                                    logger.info(f"Message deleted from SQS queue: {receipt_handle}")

                                except Exception as e:
                                    logger.error(f"Error during inference or uploading image: {e}")
                                    # Update the `status` column to 'Failed' in the database
                                    conn = get_db_connection()
                                    if conn:
                                        try:
                                            with conn.cursor() as cur:
                                                update_query = """
                                                    UPDATE ai_session_image
                                                    SET status = 'Failed', updated_at = NOW()
                                                    WHERE session_id = %s 
                                                        AND image_side = %s
                                                        AND active = 'True' 
                                                        AND status = 'Open'
                                                        AND deleted_at IS NULL
                                                """
                                                cur.execute(update_query, (session_id, image_side))
                                                conn.commit()
                                                logger.info(f"Updated status to 'Failed' for session_id {session_id} and image_side {image_side}.")
                                        except Exception as db_error:
                                            logger.error(f"Database error while updating status to 'Failed': {db_error}")
                                        finally:
                                            conn.close()
                                    # Do not delete the message in case of error, so it can be retried
                            else:
                                logger.error("Image download failed, skipping message.")

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except Exception as e:
            logger.error(f"Error receiving messages from SQS: {e}")

# Function to run the polling loop in a background thread
def start_polling_in_background():
    asyncio.run(poll_sqs())

# Start the polling loop in a background thread
thread = threading.Thread(target=start_polling_in_background)
thread.daemon = True  # This makes the thread terminate when the main program ends
thread.start()

@app.route("/api/session/start-evaluate", methods=["POST"])
def start_evaluation():
    # 1. Extract Headers and Body
    logger.info("Received request to start evaluation session.")
    api_key = request.headers.get("x-api-key")
    data = request.get_json()
    session_id = data.get("session_id")

    if not api_key or not session_id:
        logger.warning("Missing api_key header or session_id in the request.")
        return jsonify({"error": "api_key header and session_id are required"}), 400

    # 2. Verify API Key using Generic Function
    api_key_id = verify_api_key(api_key)
    if not api_key_id:
        logger.warning("Unauthorized access with an invalid API Key.")
        return jsonify({"error": "Invalid API Key"}), 401

    # 3. Connect to the Database and Check session_id Validity
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to the database for session evaluation.")
        return jsonify({"error": "Database connection failed"}), 500
    try:
        with conn.cursor() as cur:
            # Query to check for session_id in the ai_session_image table
            query = """
                SELECT image_side, output_image_object_key, result_after_nms, output_image_version_id, input_image_object_key, input_image_version_id
                FROM ai_session_image
                WHERE session_id = %s
                  AND image_side IN ('FRONT', 'BACK', 'TOP', 'BOTTOM', 'LEFT', 'RIGHT')
                  AND active = TRUE
                  AND status = 'Completed'
                  AND deleted_at IS NULL
            """

            cur.execute(query, (session_id,))
            rows = cur.fetchall()

            if len(rows) != 6:
                logger.warning(f"Not all image sides found for session_id {session_id}.")
                return jsonify({"error": "Missing image sides for the given session_id. Please upload all the images again."}), 404

            # Update the status in ai_session table to 'Processing'
            update_status_query = """
                UPDATE ai_session
                SET status = %s, updated_at = %s
                WHERE session_id = %s
            """
            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(update_status_query, ("Processing", utc_now, session_id))
            conn.commit()

             # Prepare detection_results for get_ai_grade
            detection_results = {}
            output_image_keys = {}
            input_image_keys = {}

            for row in rows:
                image_side = row[0].lower()  # Normalize to lowercase keys
                output_image_object_key = row[1]
                result_after_nms = row[2]
                output_image_version_id = row[3]
                input_image_object_key = row[4]
                input_image_version_id = row[5]

                detection_results[image_side] = result_after_nms
                # output_image_keys[image_side] = f"{base_url}{output_image_object_key}"  # Prepend the base URL
                output_image_keys[image_side] =  f"{base_url}{output_image_object_key}?versionId={output_image_version_id}"
                input_image_keys[image_side] = f"{base_url}{input_image_object_key}?versionId={input_image_version_id}"

            # Pass detection_results to get_ai_grade
            results = get_ai_grade.get_ai_grade(
                        detection_results=detection_results,
                        use_ml_housing = True,
                        device_model=None,
                        dimensions_csv=None
                    )

            # for brightness
            is_back_lit = True
            back_results = detection_results.get('back')
            logger.info(f"back results {back_results}")
            if back_results and len(back_results) > 0:
                is_back_lit = back_results[0].get('brightness_prediction_back')
                logger.info(f"brightness info {is_back_lit}")

            is_front_lit = True
            front_results = detection_results.get('front')
            if front_results and len(front_results) > 0:
                is_front_lit = front_results[0].get('brightness_prediction_screen')
                logger.info(f"brightness info {is_front_lit}")

            screen_grade = results.get("ai_screen")
            housing_grade = results.get("ai_housing")
            cs_count = results.get("cover_spot_count")
            defects_screen_dict = results["details"]["screen"]["defects"]


            # Prepare the API response
            response = {
                "output_img_url": input_image_keys,
                "AI_Grading": {
                    "ai_screen": screen_grade,
                    "ai_housing": housing_grade,
                    "cover_spot_count":cs_count,
                    "defects_details_screen":defects_screen_dict,
                    "screen_defect_threshold": screen_defect_threshold,
                    "back_defect_threshold": back_defect_threshold,
                }
                ,
                "is_back_lit": is_back_lit,     
                "is_front_lit": is_front_lit 
            }

            # Update the ai_session table with the new grading values and current UTC timestamp, and the grading status sucess
            update_query = """
                UPDATE ai_session
                SET ai_screen_grading = %s,
                    ai_housing_grading = %s,
                    updated_at = %s,
                    status = %s
                WHERE session_id = %s
            """

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(update_query, (screen_grade, housing_grade, utc_now, "Grading_Sucess", session_id))
            conn.commit()

            logger.info(f"Evaluation completed for session_id {session_id}.")
            return jsonify(response), 200

    # update the db status if garding failed
    except Exception as e:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            rows = cur.fetchall()
            # Update query
            update_query = """
                UPDATE ai_session
                SET updated_at = %s,
                    status = %s
                WHERE session_id = %s
            """

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(
                update_query,
                (utc_now, "Grading_Failed", session_id)
            )
            conn.commit()
        logger.error(f"Error during session evaluation: {e}")
        return jsonify({"error": f"Failed to evaluate session due to: {str(e)}"}), 500

    finally:
        conn.close()
        logger.info("Database connection closed after session evaluation.")


@app.route("/api/session/qc-evaluation", methods=["POST"])
def qc_evaluation():
    logger.info("Received request for QC Evaluation.")
    api_key = request.headers.get("x-api-key")
    data = request.get_json()
    session_id = data.get("session_id")
    qc_screen_grading = data.get("qcScreenGrading")
    qc_screen_remark = data.get("qcScreenRemarks")
    qc_housing_grading = data.get("qcHousingGrading")
    qc_housing_remark = data.get("qcHousingRemarks")

    if not api_key or not session_id:
        logger.warning("Missing api_key header or session_id in the request.")
        return jsonify({"error": "api_key header and session_id are required"}), 400

    # Log remarks for debugging
    logger.debug(f"qc_screen_remark: {qc_screen_remark}, qc_housing_remark: {qc_housing_remark}")

    api_key_id = verify_api_key(api_key)
    if not api_key_id:
        logger.warning("Unauthorized access with an invalid API Key.")
        return jsonify({"error": "Invalid API Key"}), 401

    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to the database for QC evaluation.")
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn.cursor() as cur:
            # Check session existence
            check_query = "SELECT session_id FROM ai_session WHERE session_id = %s"
            cur.execute(check_query, (session_id,))
            session_exists = cur.fetchone()

            if not session_exists:
                logger.warning(f"Session ID {session_id} does not exist in the database.")
                return jsonify({"error": "Session ID does not exist"}), 404

            # Update query
            update_query = """
                UPDATE ai_session
                SET qc_screen_grading = %s,
                    qc_screen_remark = %s,
                    qc_housing_grading = %s,
                    qc_housing_remark = %s,
                    updated_at = %s,
                    status = %s
                WHERE session_id = %s
            """

            # Handle null remarks
            qc_screen_remark = qc_screen_remark if qc_screen_remark else ""
            qc_housing_remark = qc_housing_remark if qc_housing_remark else ""

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(
                update_query,
                (qc_screen_grading, qc_screen_remark, qc_housing_grading, qc_housing_remark, utc_now, "QC-Evaluated", session_id)
            )
            conn.commit()

            logger.info(f"QC evaluation updated for session_id {session_id}.")
            return jsonify({"success": "True"}), 200

    except Exception as e:
        logger.error(f"Error during QC evaluation: {e}")
        return jsonify({"error": f"Failed to process QC evaluation due to: {str(e)}"}), 500

    finally:
        conn.close()
        logger.info("Database connection closed after QC evaluation.")


if __name__ == "__main__":
    app.run(host="52.220.71.136", debug=True, use_reloader=False, port=5001)
