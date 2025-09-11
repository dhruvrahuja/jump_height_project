from flask import Flask, request, jsonify
from flask_cors import CORS  
import os
import tempfile
from jump_height2 import detect_jumps

# ------------------- Flask App -------------------
app = Flask(__name__)
CORS(app)  

@app.route('/', methods=['GET'])
def index():
    """
    Health check endpoint
    """
    return jsonify({
        "message": "Jump Height Detection API",
        "status": "running",
        "version": "1.0",
        "endpoints": {
            "/": "GET - API status and information",
            "/jump": "POST - Upload video and height to detect jumps"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for monitoring
    """
    return jsonify({"status": "healthy"})

@app.route('/jump', methods=['POST'])
def jump():
    """
    Main endpoint for jump detection
    Expects:
    - video: video file (form data)
    - height_m: person's height in meters (form data)
    
    Returns:
    - JSON with jump detection results
    """
    if 'video' not in request.files or 'height_m' not in request.form:
        return jsonify({"error": "Missing video file or height parameter"}), 400

    video_file = request.files['video']
    
    # Validate height parameter
    try:
        height_m = float(request.form['height_m'])
        if height_m <= 0:
            return jsonify({"error": "Height must be a positive number"}), 400
    except ValueError:
        return jsonify({"error": "Invalid height_m value. Must be a number."}), 400

    # Validate video file
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        # Process the video and detect jumps
        result = detect_jumps(video_path, height_m)
        
        # Add metadata to response
        if "error" not in result:
            result["metadata"] = {
                "video_filename": video_file.filename,
                "person_height_m": height_m
            }
            
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(video_path):
            os.remove(video_path)

    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8001, debug=True)
