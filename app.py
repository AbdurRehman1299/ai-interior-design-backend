import os
import io
import base64
import json
import re
from typing import Tuple, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import colorgram
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import timm
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

load_dotenv()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

print("Loading MiDaS depth estimation model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    print("MiDaS model loaded successfully!")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    midas = None

print("Loading local text generation model (t5-small)...")
try:
    # Load from the local folder we created in cache_models.py
    MODEL_PATH = "./t5-small-local" 
    
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    
    print(f"Loading model from {MODEL_PATH}...")
    text_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    
    print("Creating text-generation pipeline...")
    text_pipe = pipeline(
        "text2text-generation",
        model=text_model,
        tokenizer=tokenizer,
        device=device
    )
    print("Local AI text pipeline (t5-small) loaded successfully.")
except Exception as e:
    print(f"Error loading local text model: {e}")
    text_pipe = None
    tokenizer = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tuple_to_hex(rgb_tuple) -> str:
    """Convert RGB tuple to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple.r, rgb_tuple.g, rgb_tuple.b)

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    print("Health check successful!")
    return jsonify({'status': 'healthy', 'message': 'Backend is running!'}), 200

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """
    Handles the image upload, runs depth estimation, and extracts colors.
    """
    print("\n--- Received a new request for /api/process-image ---")
    
    if midas is None:
        return jsonify({'error': 'MiDaS model is not loaded on server'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img)
        
        input_batch = transform(img_np).to(device)
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy() 
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        _, buffer = cv2.imencode('.png', depth_colored)
        depth_map_bytes = buffer.tobytes()
        
        encoded_depth_viz = base64.b64encode(depth_map_bytes).decode('utf-8')
        depth_map_url = f"data:image/png;base64,{encoded_depth_viz}"
        
        encoded_original = base64.b64encode(image_bytes).decode('utf-8')
        original_image_url = f"data:image/png;base64,{encoded_original}"
        
        extracted_colors = colorgram.extract(io.BytesIO(image_bytes), 5)
        hex_colors = [tuple_to_hex(color.rgb) for color in extracted_colors]

        print("Image processing and color extraction finished successfully.")

        return jsonify({
            'depthMapUrl': depth_map_url,
            'originalImageUrl': original_image_url,
            'colors': hex_colors
        })

    except Exception as e:
        print(f"An error occurred in the /api/process-image block: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-room-ai', methods=['POST'])
def generate_room_ai():
    """
    Handles the AI text prompt using the LOCAL t5-small model.
    """
    print("\n--- Received a new request for /api/generate-room-ai ---")
    
    if not text_pipe or not tokenizer:
        return jsonify({"error': 'AI text model is not loaded on server"}), 500

    try:
        data = request.get_json()
        user_prompt = data.get('prompt')
        available_items = data.get('availableItems')

        if not user_prompt or not available_items:
            return jsonify({"error": "Missing prompt or availableItems"}), 400
        
        # Create a simple string of available item IDs
        item_id_list = ", ".join([f"'{item['id']}'" for item in available_items])
        # Create a "few-shot" prompt to show the model the pattern.
        # We give it examples so it can complete the final one.
        prompt = f"""
user: "a modern sofa and a plant", options: ['modern-sofa', 'wooden-table', 'plant'], selection: ['modern-sofa', 'plant']
user: "I want a brown table", options: ['modern-sofa', 'wooden-table', 'floor-lamp'], selection: ['wooden-table']
user: "a black sofa", options: [{item_id_list}], selection: ['modern-sofa']
user: "{user_prompt}", options: [{item_id_list}], selection:
"""

        # Call the local model using the pipeline
        print("Generating AI response locally with t5-small...")
        outputs = text_pipe(
            prompt,
            max_new_tokens=150,  # Max length for the JSON array
            do_sample=False,
            num_beams=2,
        )
        ai_response_text = outputs[0]['generated_text']
        print(f"Local AI Response Text: {ai_response_text}")

        # Extract the JSON list (Robustly)
        try:
            match = re.search(r'\[.*?\]', ai_response_text)
            if not match:
                print(f"No JSON array found in response: '{ai_response_text}'")
                raise Exception("No JSON array found in AI response")

            json_str = match.group(0) # Get the matched string
            json_str_valid = json_str.replace("'", '"')
            furniture_ids = json.loads(json_str_valid)
            # Send the clean list back to the frontend
            return jsonify({"furniture_ids": furniture_ids})
        except Exception as e:
            print(f"Error parsing AI response: {e}")
            return jsonify({"error": f"AI returned an invalid format: {ai_response_text}"}), 500
    except Exception as e:
        print(f"Error in /api/generate-room-ai: {e}")
        return jsonify({"error": "Internal server error"}), 500

# --- Run the server ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=False, host='0.0.0.0', port=port)