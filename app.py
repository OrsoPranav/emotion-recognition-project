# app.py - Main Flask application
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import base64
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your emotion classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the CNN model architecture
class EmotionNetwork(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNetwork, self).__init__()
        
        # CNN layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 54, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(54, 31, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(31, 57, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(57 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load the model
def load_model():
    try:
        model = EmotionNetwork()
        
        # Define the model path - you'll need to update this path
        model_path = os.path.join(os.path.dirname(__file__), "models", "Pranav-221AI030-Adagrad-model.pth")
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize model
model = load_model()

# Define image preprocessing 
def preprocess_image(image):
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 48x48 (for the model input)
        image = image.resize((48, 48))
        
        # Define transformations similar to what was used during training
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

# Route for serving static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for emotion prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = None
        
        # Check if image is sent as base64
        if 'image_data' in request.form:
            image_b64 = request.form.get('image_data')
            # Remove header if present
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            
            # Decode base64 to image
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
        
        # Check if image is sent as file
        elif 'image' in request.files:
            image_file = request.files['image']
            image = Image.open(image_file)
        
        # Return error if no image is provided
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Preprocess the image for prediction
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            probabilities = probabilities.cpu().numpy() * 100  # Convert to percentages
        
        predicted_emotion = EMOTIONS[np.argmax(probabilities)]
        
        # Save the original image to the 'saved_images' directory
        try:
            save_dir = os.path.join(os.path.dirname(__file__), "saved_images")
            os.makedirs(save_dir, exist_ok=True)
            # Determine the next index based on existing files
            existing_files = [f for f in os.listdir(save_dir) if f.endswith('.jpg')]
            next_index = len(existing_files) + 1
            save_filename = f"{next_index}_{predicted_emotion}.jpg"
            save_path = os.path.join(save_dir, save_filename)
            image.save(save_path, format='JPEG')
            logger.info(f"Image saved as {save_filename}")
        except Exception as save_err:
            logger.error(f"Error saving image: {str(save_err)}")
        
        # Create result dictionary
        results = {
            'emotions': EMOTIONS,
            'probabilities': probabilities.tolist(),
            'predicted_emotion': predicted_emotion
        }
        
        logger.info(f"Prediction successful: {predicted_emotion}")
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "saved_images"), exist_ok=True)
    
    # Get port from environment variable (for Render compatibility)
    port = int(os.environ.get("PORT", 5000))
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port)
