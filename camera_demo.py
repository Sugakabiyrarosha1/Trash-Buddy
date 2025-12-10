"""
Trash-Buddy Real-Time Camera Demo
Real-time waste classification using webcam
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import time

# Configuration
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FPS_UPDATE_INTERVAL = 1.0  # Update FPS every second
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    return model, label_classes


def predict_image(model, image, label_classes):
    """Predict class for an image"""
    # Preprocess image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(3, len(label_classes)))
    
    # Format results
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        if prob.item() >= CONFIDENCE_THRESHOLD:
            results.append({
                'class': label_classes[idx.item()],
                'confidence': prob.item()
            })
    
    return results


def draw_predictions(frame, predictions, fps):
    """Draw predictions on frame"""
    # Draw FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw predictions
    y_offset = 70
    for i, pred in enumerate(predictions):
        class_name = pred['class']
        confidence = pred['confidence']
        confidence_pct = confidence * 100
        
        # Draw background rectangle
        text = f"{i+1}. {class_name}: {confidence_pct:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Color based on confidence
        if confidence >= 0.7:
            color = (0, 255, 0)  # Green
        elif confidence >= 0.5:
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 0, 255)  # Red
        
        # Draw rectangle
        cv2.rectangle(frame, (10, y_offset - text_height - 5), 
                     (20 + text_width, y_offset + 5), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw confidence bar
        bar_width = int(confidence * 200)
        cv2.rectangle(frame, (15, y_offset + 10), 
                     (15 + bar_width, y_offset + 20), color, -1)
        cv2.rectangle(frame, (15, y_offset + 10), 
                     (215, y_offset + 20), (255, 255, 255), 1)
        
        y_offset += 40
    
    return frame


def main():
    """Main function for camera demo"""
    print("=" * 80)
    print("Trash-Buddy Real-Time Camera Demo")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model, label_classes = load_model()
    print("Model loaded successfully!")
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera initialized")
    print("\nControls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save current frame")
    print("   - Press 'r' to reset predictions")
    print("\nStarting detection...")
    print("=" * 80)
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Prediction smoothing
    prediction_history = []
    history_size = 5
    
    frame_count = 0
    last_prediction_time = time.time()
    prediction_interval = 0.1  # Predict every 100ms
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Update FPS
        fps_counter += 1
        if current_time - fps_start_time >= FPS_UPDATE_INTERVAL:
            current_fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time
        
        # Make prediction at intervals
        if current_time - last_prediction_time >= prediction_interval:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Predict
            predictions = predict_image(model, pil_image, label_classes)
            
            # Smooth predictions using history
            if predictions:
                prediction_history.append(predictions[0])
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
            
            last_prediction_time = current_time
        
        # Use most common prediction from history
        if prediction_history:
            # Get most recent confident prediction
            current_predictions = [prediction_history[-1]]
            if len(prediction_history) >= 3:
                # Add top 2 more if available
                for pred in prediction_history[-3:-1]:
                    if pred not in current_predictions:
                        current_predictions.append(pred)
                        if len(current_predictions) >= 3:
                            break
        else:
            current_predictions = []
        
        # Draw predictions on frame
        frame = draw_predictions(frame, current_predictions, current_fps)
        
        # Display frame
        cv2.imshow('Trash-Buddy Real-Time Detection', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('r'):
            # Reset predictions
            prediction_history = []
            print("Predictions reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera demo closed")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()












