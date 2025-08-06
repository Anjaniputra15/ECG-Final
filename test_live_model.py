#!/usr/bin/env python3
"""
Live ECG R-Peak Detection Testing
Load the trained bootstrap model and test on real ECG images.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import argparse
import os

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from enhanced_preprocessing import ECGImagePreprocessor

# Gemini AI integration (optional)
try:
    from AIanalysis.ecg_result_analyzer import ECGResultAnalyzer
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini AI integration not available")


class SimpleECGDetector(nn.Module):
    """Simple CNN model for R-peak detection - matches bootstrap training."""
    
    def __init__(self, num_classes=6, num_queries=60):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # DETR-style heads
        feature_dim = 512 * 8 * 8
        self.query_embed = nn.Embedding(num_queries, 256)
        self.input_proj = nn.Linear(feature_dim, 256)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(256, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 3)
        
        # Output heads
        self.class_head = nn.Linear(256, num_classes)
        self.keypoint_head = nn.Linear(256, 2)
        self.confidence_head = nn.Linear(256, 1)
        
    def forward(self, images):
        B = images.size(0)
        
        # Extract features
        features = self.backbone(images)  # [B, 512, 8, 8]
        features = features.view(B, -1)  # [B, 512*8*8]
        features = self.input_proj(features)  # [B, 256]
        
        # Prepare queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, 256]
        
        # Decoder
        memory = features.unsqueeze(1)  # [B, 1, 256]
        decoder_output = self.decoder(queries, memory)  # [B, num_queries, 256]
        
        # Output heads
        class_logits = self.class_head(decoder_output)  # [B, num_queries, num_classes]
        keypoint_coords = torch.sigmoid(self.keypoint_head(decoder_output))  # [B, num_queries, 2]
        confidence_scores = torch.sigmoid(self.confidence_head(decoder_output))  # [B, num_queries, 1]
        
        return {
            'class_logits': class_logits,
            'keypoint_coords': keypoint_coords,
            'confidence_scores': confidence_scores
        }


def load_trained_model(model_path, device='mps'):
    """Load the trained bootstrap model."""
    model = SimpleECGDetector(num_classes=6, num_queries=60)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def preprocess_ecg_image(image_path, preprocessor):
    """Preprocess ECG image for model input."""
    # Convert synthetic image directly to tensor
    image = Image.open(image_path).convert('RGB').resize((512, 512))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor, image


def detect_rpeaks(model, image_tensor, device='mps', confidence_threshold=0.1):
    """Run R-peak detection on preprocessed image."""
    with torch.no_grad():
        # Add batch dimension and move to device
        images = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Extract outputs
        confidence_scores = outputs['confidence_scores'].squeeze(-1).squeeze(0)  # [num_queries]
        keypoint_coords = outputs['keypoint_coords'].squeeze(0)  # [num_queries, 2]
        class_logits = outputs['class_logits'].squeeze(0)  # [num_queries, num_classes]
        
        # Filter by confidence
        high_conf_mask = confidence_scores > confidence_threshold
        
        detections = {
            'confidences': confidence_scores[high_conf_mask].cpu().numpy(),
            'keypoints': keypoint_coords[high_conf_mask].cpu().numpy(),
            'classes': torch.argmax(class_logits[high_conf_mask], dim=1).cpu().numpy(),
            'all_confidences': confidence_scores.cpu().numpy(),
            'all_keypoints': keypoint_coords.cpu().numpy()
        }
        
        return detections


def visualize_results(original_image, detections, confidence_threshold=0.1):
    """Visualize R-peak detections on the original image."""
    # Create a copy for drawing
    vis_image = original_image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Draw detected R-peaks
    image_width, image_height = vis_image.size
    
    for i, (conf, keypoint) in enumerate(zip(detections['confidences'], detections['keypoints'])):
        if conf > confidence_threshold:
            # Convert normalized coordinates to image coordinates
            x = int(keypoint[0] * image_width)
            y = int(keypoint[1] * image_height)
            
            # Draw detection circle and confidence
            radius = 8
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        outline='red', fill='red', width=3)
            draw.text((x+10, y-10), f'{conf:.3f}', fill='red')
    
    return vis_image


def test_single_image(model_path, image_path, confidence_threshold=0.1, device='mps', 
                     use_gemini=False, gemini_api_key=None, output_dir='analysis_results'):
    """Test the model on a single ECG image with optional Gemini AI analysis."""
    print(f"🔍 Testing ECG R-Peak Detection")
    print(f"📁 Model: {Path(model_path).name}")
    print(f"🖼️  Image: {Path(image_path).name}")
    print(f"🎯 Confidence Threshold: {confidence_threshold}")
    if use_gemini and GEMINI_AVAILABLE:
        print(f"🤖 Gemini AI: Enabled")
    print("-" * 50)
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    
    # Initialize preprocessor
    preprocessor = ECGImagePreprocessor()
    
    # Preprocess image
    print("Preprocessing image...")
    image_tensor, original_image = preprocess_ecg_image(image_path, preprocessor)
    
    # Run detection
    print("Running R-peak detection...")
    detections = detect_rpeaks(model, image_tensor, device, confidence_threshold)
    
    # Results
    num_detections = len(detections['confidences'])
    max_confidence = float(np.max(detections['all_confidences']))
    mean_confidence = float(np.mean(detections['all_confidences']))
    
    print("\n📊 DETECTION RESULTS:")
    print(f"   Detections found: {num_detections}")
    print(f"   Max confidence: {max_confidence:.4f}")
    print(f"   Mean confidence: {mean_confidence:.4f}")
    
    if num_detections > 0:
        print(f"\n🎯 R-PEAK LOCATIONS:")
        for i, (conf, keypoint) in enumerate(zip(detections['confidences'], detections['keypoints'])):
            x_norm, y_norm = keypoint
            print(f"   Peak {i+1}: ({x_norm:.3f}, {y_norm:.3f}) - confidence: {conf:.4f}")
    else:
        print(f"\n⚠️  No R-peaks detected above threshold {confidence_threshold}")
        print(f"   Try lowering threshold or check image quality")
    
    # Visualize results
    vis_image = visualize_results(original_image, detections, confidence_threshold)
    
    # Save visualization
    output_path = Path(image_path).parent / f"detected_{Path(image_path).name}"
    vis_image.save(output_path)
    print(f"\n💾 Visualization saved: {output_path}")
    
    # Gemini AI analysis (if enabled)
    if use_gemini and GEMINI_AVAILABLE:
        try:
            print(f"\n🤖 Running Gemini AI analysis...")
            
            # Initialize ECG analyzer with Gemini
            analyzer = ECGResultAnalyzer(gemini_api_key)
            
            # Run comprehensive analysis
            comprehensive_analysis = analyzer.analyze_with_gemini(
                image_path, detections
            )
            
            if comprehensive_analysis.get('status') == 'success':
                # Display summary
                summary_report = analyzer.generate_summary_report(comprehensive_analysis)
                print(f"\n{summary_report}")
                
                # Save detailed analysis
                analyzer.save_analysis(comprehensive_analysis, output_dir)
                
                return detections, vis_image, comprehensive_analysis
            else:
                print(f"❌ Gemini analysis failed: {comprehensive_analysis.get('error')}")
                
        except Exception as e:
            print(f"❌ Gemini integration error: {e}")
    
    return detections, vis_image


def test_directory(model_path, image_dir, confidence_threshold=0.1, device='mps'):
    """Test the model on all ECG images in a directory."""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🔍 Testing on {len(image_files)} images in {image_dir}")
    
    # Load model once
    model = load_trained_model(model_path, device)
    preprocessor = ECGImagePreprocessor()
    
    results = []
    
    for image_path in image_files:
        print(f"\nTesting: {image_path.name}")
        
        try:
            # Process image
            image_tensor, original_image = preprocess_ecg_image(image_path, preprocessor)
            detections = detect_rpeaks(model, image_tensor, device, confidence_threshold)
            
            num_detections = len(detections['confidences'])
            max_conf = float(np.max(detections['all_confidences']))
            
            result = {
                'image': image_path.name,
                'detections': num_detections,
                'max_confidence': max_conf,
                'keypoints': detections['keypoints'].tolist() if len(detections['keypoints']) > 0 else []
            }
            
            results.append(result)
            print(f"  -> {num_detections} detections, max conf: {max_conf:.4f}")
            
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Save results
    results_path = image_dir / "batch_detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Batch results saved: {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained ECG R-peak detection model with optional Gemini AI analysis')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--image', help='Single ECG image to test')
    parser.add_argument('--dir', help='Directory of ECG images to test')
    parser.add_argument('--threshold', type=float, default=0.1, help='Confidence threshold (default: 0.1)')
    parser.add_argument('--device', default='mps', help='Device to use (mps, cuda, cpu)')
    parser.add_argument('--gemini', action='store_true', help='Enable Gemini AI analysis (requires GEMINI_API_KEY)')
    parser.add_argument('--gemini-key', help='Gemini API key (or set GEMINI_API_KEY env variable)')
    parser.add_argument('--output-dir', default='analysis_results', help='Output directory for analysis reports')
    
    args = parser.parse_args()
    
    # Verify model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Check Gemini requirements
    gemini_api_key = args.gemini_key or os.getenv('GEMINI_API_KEY')
    if args.gemini:
        if not GEMINI_AVAILABLE:
            print("❌ Gemini AI integration not available. Install requirements.")
            return
        if not gemini_api_key:
            print("❌ Gemini API key required. Set GEMINI_API_KEY or use --gemini-key")
            return
        print("✅ Gemini AI integration ready")
    
    # Create output directory
    if args.gemini:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Test mode
    if args.image:
        if not Path(args.image).exists():
            print(f"❌ Image file not found: {args.image}")
            return
        test_single_image(args.model, args.image, args.threshold, args.device, 
                         args.gemini, gemini_api_key, args.output_dir)
        
    elif args.dir:
        if not Path(args.dir).exists():
            print(f"❌ Directory not found: {args.dir}")
            return
        test_directory(args.model, args.dir, args.threshold, args.device)
        
    else:
        print("❌ Please specify either --image or --dir")
        return


if __name__ == "__main__":
    main()