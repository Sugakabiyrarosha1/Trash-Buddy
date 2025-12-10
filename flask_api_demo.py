"""
Trash-Buddy Flask API Demo
Standalone Flask application for waste classification via file upload
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
label_classes = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def create_model(model_name='resnet50', num_classes=18, pretrained=False):
    """Create model architecture"""
    try:
        if model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        elif model_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v2(weights=weights)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except AttributeError:
        # Fallback for older torchvision versions
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def load_model():
    """Load the trained model"""
    global model, label_classes
    
    models_dir = Path('models')
    processed_data_dir = Path('processed_data')
    
    # Load model checkpoint
    model_checkpoint_path = list(models_dir.glob('best_model_*.pth'))[0]
    checkpoint = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration
    model_config = checkpoint.get('config', {})
    MODEL_NAME = model_config.get('MODEL_NAME', 'resnet50')
    NUM_CLASSES = len(np.load(processed_data_dir / 'label_classes.npy', allow_pickle=True))
    
    # Create and load model
    model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load label classes
    label_classes = np.load(processed_data_dir / 'label_classes.npy', allow_pickle=True)
    
    print(f"Model loaded: {MODEL_NAME}")
    print(f"   Device: {device}")
    print(f"   Classes: {len(label_classes)}")


def predict_image(image):
    """Predict class for an image"""
    # Preprocess image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(5, len(label_classes)))
    
    # Format results
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        results.append({
            'class': label_classes[idx.item()],
            'confidence': prob.item()
        })
    
    return results


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🗑️ Trash-Buddy Waste Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .upload-area {
            border: 2px dashed #3498db;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            background-color: #ecf0f1;
        }
        input[type="file"] {
            margin: 10px 0;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 0;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }
        .result-item {
            padding: 10px;
            margin: 5px 0;
            background-color: white;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }
        .confidence-bar {
            height: 20px;
            background-color: #3498db;
            border-radius: 3px;
            margin-top: 5px;
        }
        .error {
            color: #e74c3c;
            padding: 10px;
            background-color: #fadbd8;
            border-radius: 5px;
            margin: 10px 0;
        }
        img {
            max-width: 100%;
            max-height: 400px;
            margin: 20px 0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🗑️ Trash-Buddy Waste Classification</h1>
        <p style="text-align: center; color: #7f8c8d;">Upload an image to classify waste type</p>
        
        <form method="post" enctype="multipart/form-data" id="uploadForm">
            <div class="upload-area">
                <input type="file" name="file" id="fileInput" accept="image/*" required>
                <br>
                <button type="submit">Classify Image</button>
            </div>
        </form>
        
        <div id="preview"></div>
        <div id="results"></div>
    </div>
    
    <script>
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.innerHTML = '<img src="' + e.target.result + '" alt="Preview">';
                };
                reader.readAsDataURL(file);
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        results = predict_image(image)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Web interface for file upload"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE.replace(
                '<div id="results"></div>',
                '<div class="error">No file provided</div>'
            ))
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE.replace(
                '<div id="results"></div>',
                '<div class="error">No file selected</div>'
            ))
        
        if not allowed_file(file.filename):
            return render_template_string(HTML_TEMPLATE.replace(
                '<div id="results"></div>',
                '<div class="error">Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP</div>'
            ))
        
        try:
            # Read image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save image for display
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            image.save(image_path)
            
            # Make prediction
            results = predict_image(image)
            
            # Format results HTML
            results_html = '<div class="results"><h2>Classification Results:</h2>'
            for i, result in enumerate(results, 1):
                confidence_pct = result['confidence'] * 100
                results_html += f'''
                <div class="result-item">
                    <div>
                        <strong>{i}. {result['class']}</strong>
                        <div class="confidence-bar" style="width: {confidence_pct}%"></div>
                    </div>
                    <div><strong>{confidence_pct:.2f}%</strong></div>
                </div>
                '''
            results_html += '</div>'
            
            # Add image preview
            image_html = f'<img src="/uploads/{secure_filename(file.filename)}" alt="Uploaded image">'
            
            return render_template_string(HTML_TEMPLATE.replace(
                '<div id="preview"></div>',
                f'<div id="preview">{image_html}</div>'
            ).replace(
                '<div id="results"></div>',
                f'<div id="results">{results_html}</div>'
            ))
        
        except Exception as e:
            return render_template_string(HTML_TEMPLATE.replace(
                '<div id="results"></div>',
                f'<div class="error">Error: {str(e)}</div>'
            ))
    
    return render_template_string(HTML_TEMPLATE)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return app.send_static_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    print("=" * 80)
    print("Trash-Buddy Flask API Demo")
    print("=" * 80)
    print("\nLoading model...")
    load_model()
    print("\nModel loaded successfully!")
    print("\nStarting Flask server...")
    print("   Web Interface: http://localhost:5000")
    print("   API Endpoint: http://localhost:5000/predict")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)












