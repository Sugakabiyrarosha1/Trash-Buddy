"""
Trash-Buddy Inference Script
Standalone script for making predictions on waste images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import json
import argparse

class WasteClassifier:
    """Waste classification model wrapper"""

    def __init__(self, model_path, label_classes_path, device='auto'):
        """
        Initialize the classifier

        Args:
            model_path: Path to model checkpoint (.pth file)
            label_classes_path: Path to label classes (.npy file)
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load label classes
        self.label_classes = np.load(label_classes_path, allow_pickle=True)

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('config', {})

        self.model_name = model_config.get('MODEL_NAME', 'resnet50')
        self.image_size = model_config.get('IMAGE_SIZE', 224)
        self.imagenet_mean = model_config.get('IMAGENET_MEAN', [0.485, 0.456, 0.406])
        self.imagenet_std = model_config.get('IMAGENET_STD', [0.229, 0.224, 0.225])

        # Create model
        self.model = self._create_model(self.model_name, len(self.label_classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.imagenet_mean, std=self.imagenet_std)
        ])

    def _create_model(self, model_name, num_classes):
        """Create model architecture"""
        try:
            if model_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT
                model = models.efficientnet_b0(weights=weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_name == 'mobilenet_v2':
                weights = models.MobileNet_V2_Weights.DEFAULT
                model = models.mobilenet_v2(weights=weights)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except AttributeError:
            # Fallback for older versions
            if model_name == 'resnet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif model_name == 'mobilenet_v2':
                model = models.mobilenet_v2(pretrained=True)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    def predict(self, image_path, top_k=3):
        """
        Predict class for an image

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.label_classes)), dim=1)

        # Format results
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        top_classes = [self.label_classes[idx] for idx in top_indices]

        return {
            'predicted_class': top_classes[0],
            'confidence': float(top_probs[0]),
            'top_k': [
                {'class': cls, 'confidence': float(prob)}
                for cls, prob in zip(top_classes, top_probs)
            ]
        }


def main():
    parser = argparse.ArgumentParser(description='Trash-Buddy Inference')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/best_model_resnet50.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--labels', type=str, default='processed_data/label_classes.npy',
                       help='Path to label classes file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    parser.add_argument('--output', type=str, help='Output JSON file path')

    args = parser.parse_args()

    # Initialize classifier
    print(f"Loading model from {args.model}...")
    classifier = WasteClassifier(args.model, args.labels)
    print(f"Model loaded on {classifier.device}")

    # Make prediction
    print(f"\nPredicting on {args.image}...")
    result = classifier.predict(args.image, top_k=args.top_k)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nTop {args.top_k} Predictions:")
    for i, pred in enumerate(result['top_k'], 1):
        print(f"  {i}. {pred['class']:<30s} {pred['confidence']*100:.2f}%")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
