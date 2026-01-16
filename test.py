@app.route("/api/session/start-evaluate", methods=["POST"])
def start_evaluation():
    # 1. Extract Headers and Body
    logger.info("Received request to start evaluation session.")
    api_key = request.headers.get("x-api-key")
    data = request.get_json()
    session_id = data.get("session_id")
    print(session_id)

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
                SELECT image_side, output_image_object_key, result_after_nms
                FROM ai_session_image
                WHERE session_id = %s
                  AND image_side IN ('FRONT', 'BACK', 'TOP', 'BOTTOM', 'LEFT', 'RIGHT')
                  AND active = TRUE
                  AND status = 'Open'
                  AND deleted_at IS NULL
            """
            cur.execute(query, (session_id,))
            rows = cur.fetchall()

            if len(rows) != 6:
                logger.warning(f"Not all image sides found for session_id {session_id}.")
                return jsonify({"error": "Missing or invalid image sides for the given session_id"}), 404

             # Prepare detection_results for get_ai_grade
            detection_results = {}
            output_image_keys = {}

            for row in rows:
                image_side = row[0].lower()  # Normalize to lowercase keys
                output_image_object_key = row[1]
                result_after_nms = row[2]

                detection_results[image_side] = result_after_nms
                output_image_keys[image_side] = f"{base_url}{output_image_object_key}"  # Prepend the base URL

            # Pass detection_results to get_ai_grade
            results = get_ai_grade.get_ai_grade(
                        detection_results=detection_results,
                        device_model=None,
                        dimensions_csv=None
                    )
            
            screen_grade = results.get("ai_screen")
            housing_grade = results.get("ai_housing")
            print("Screen Grade:", screen_grade)
            print("Housing Grade:", housing_grade)

            # Prepare the API response
            response = {
                "output_img_url": output_image_keys,
                "AI_Grading": {
                    "ai_screen": screen_grade,
                    "ai_housing": housing_grade
                }
            }

            # Update the ai_session table with the new grading values and current UTC timestamp
            update_query = """
                UPDATE ai_session
                SET ai_screen_grading = %s,
                    ai_housing_grading = %s,
                    updated_at = %s
                WHERE session_id = %s
            """

            utc_now = datetime.datetime.now(datetime.timezone.utc)
            cur.execute(update_query, (screen_grade, housing_grade, utc_now, session_id))
            conn.commit()

            logger.info(f"Evaluation completed for session_id {session_id} and grading updated.")
            return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during session evaluation: {e}")
        return jsonify({"error": f"Failed to evaluate session due to: {str(e)}"}), 500

    finally:
        conn.close()
        logger.info("Database connection closed after session evaluation.")
