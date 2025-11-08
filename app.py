import os
import io
import base64
from typing import Tuple, List
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import colorgram
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# --- SETUP ---
load_dotenv()
app = Flask(__name__)

# Security: Set max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app, resources={r"/api/*": {"origins": ["https://roomdev.vercel.app", "http://localhost:3000"]}})

# --- DEPTH MODEL CONFIGURATION ---
# Load MiDaS model for depth estimation
print("Loading MiDaS depth estimation model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MiDaS model (will download on first run, ~100MB)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

print("MiDaS model loaded successfully!")

# --- CONSTANTS ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# --- HELPER FUNCTIONS ---
def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def tuple_to_hex(rgb_tuple) -> str:
    """Convert RGB tuple to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(rgb_tuple.r, rgb_tuple.g, rgb_tuple.b)

# --- MAIN API ENDPOINT ---
@app.route('/api/process-image', methods=['POST'])
def process_image():
    # --- Print a startup message to confirm the endpoint is hit ---
    print("\n--- Received a new request ---")

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Validate file
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        image_bytes = file.read()
        print(f"Received image: {file.filename}, Size: {len(image_bytes)} bytes.")

        print("Running depth estimation...")
        
        # Load and convert image to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img)
        
        # Apply MiDaS transforms
        input_batch = transform(img_np).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap for visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Convert to PNG bytes
        _, buffer = cv2.imencode('.png', depth_colored)
        depth_map_bytes = buffer.tobytes()
        
        print("Depth estimation completed successfully!")
        
        encoded_depth_viz = base64.b64encode(depth_map_bytes).decode('utf-8')
        depth_map_url = f"data:image/png;base64,{encoded_depth_viz}"
        
        # Encode original image for texture
        encoded_original = base64.b64encode(image_bytes).decode('utf-8')
        original_image_url = f"data:image/png;base64,{encoded_original}"
        
        print("Extracting colors from original image...")
        extracted_colors = colorgram.extract(io.BytesIO(image_bytes), 5)
        hex_colors = [tuple_to_hex(color.rgb) for color in extracted_colors]

        print("AI model and color extraction finished successfully.")

        return jsonify({
            'depthMapUrl': depth_map_url,
            'colors': hex_colors
        })

    except Exception as e:
        print(f"An error occurred in the try block: {e}")
        return jsonify({'error': str(e)}), 500

# --- RUN THE SERVER ---
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, port=port)
