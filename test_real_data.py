#!/usr/bin/env python3
"""
Real ECG Data Testing Script
Test your models on actual ECG images and datasets with detailed analysis.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our models
from models.multimodal_ecg_model import create_multimodal_ecg_model
from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
from uncertainty.monte_carlo_uncertainty import MonteCarloDropout
from visualization.attention_maps import AttentionVisualizer
from augmentation.enhanced_transforms import create_ecg_augmentation_pipeline

class RealECGTester:
    """Test ECG models on real ECG data."""
    
    def __init__(self, device='mps'):
        self.device = device
        
        # Create experiments folder with date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"real_data_test_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🫀 Real ECG Data Testing Suite")
        print(f"Device: {device}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
    
    def load_ecg_image(self, image_path: str, target_size=(224, 224)):
        """Load and preprocess ECG image."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(target_size)
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            return image_tensor, image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None
    
    def test_multimodal_on_real_data(self, image_paths: list, num_samples=5):
        """Test multi-modal model on real ECG images."""
        print(f"\n🔬 Testing Multi-Modal Model on {len(image_paths)} real ECG images...")
        
        # Create model
        model = create_multimodal_ecg_model(num_classes=6)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"\nProcessing ECG {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load ECG image
            image_tensor, original_image = self.load_ecg_image(image_path, (224, 224))
            if image_tensor is None:
                continue
            
            # Create dummy signal and metadata (since we only have images)
            batch_size = 1
            images = image_tensor.unsqueeze(0).to(self.device)
            signals = torch.randn(batch_size, 12, 1000).to(self.device)  # Dummy 12-lead signal
            numerical_meta = torch.randn(batch_size, 5).to(self.device)  # Dummy metadata
            categorical_meta = {
                'sex': torch.randint(0, 2, (batch_size,)).to(self.device),
                'device_type': torch.randint(0, 5, (batch_size,)).to(self.device),
                'recording_condition': torch.randint(0, 3, (batch_size,)).to(self.device)
            }
            
            # Forward pass
            with torch.no_grad():
                outputs = model(images, signals, numerical_meta, categorical_meta)
            
            # Get predictions
            diagnosis_probs = torch.softmax(outputs['diagnosis_logits'], dim=1)
            severity_probs = torch.softmax(outputs['severity_logits'], dim=1)
            confidence = outputs['confidence_scores']
            
            # Uncertainty estimation
            uncertainty_outputs = model.predict_with_uncertainty(
                images, signals, numerical_meta, categorical_meta, n_samples=10
            )
            
            # Store results
            result = {
                'image_name': Path(image_path).name,
                'diagnosis_prediction': diagnosis_probs.cpu().numpy().tolist(),
                'severity_prediction': severity_probs.cpu().numpy().tolist(),
                'confidence_score': confidence.cpu().numpy().tolist(),
                'epistemic_uncertainty': uncertainty_outputs['epistemic_uncertainty'].cpu().numpy().tolist(),
                'aleatoric_uncertainty': uncertainty_outputs['aleatoric_uncertainty'].cpu().numpy().tolist(),
                'predicted_diagnosis': int(diagnosis_probs.argmax(dim=1).item()),
                'predicted_severity': int(severity_probs.argmax(dim=1).item())
            }
            results.append(result)
            
            # Print predictions
            print(f"  📊 Diagnosis Prediction: Class {result['predicted_diagnosis']} (confidence: {confidence.item():.3f})")
            print(f"  📈 Severity Prediction: Class {result['predicted_severity']}")
            print(f"  🎯 Epistemic Uncertainty: {uncertainty_outputs['epistemic_uncertainty'].item():.4f}")
            print(f"  📊 Aleatoric Uncertainty: {uncertainty_outputs['aleatoric_uncertainty'].mean().item():.4f}")
            
            # Save visualization
            self.save_prediction_visualization(original_image, result, i+1)
        
        # Save all results
        with open(self.output_dir / 'multimodal_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Multi-modal testing complete! Results saved to {self.output_dir}/multimodal_results.json")
        return results
    
    def test_vit_on_real_data(self, image_paths: list, num_samples=5):
        """Test Vision Transformer on real ECG images."""
        print(f"\n🤖 Testing Vision Transformer on {len(image_paths)} real ECG images...")
        
        # Create model
        model = create_ecg_vit_detr_model(pretrained=False)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"\nProcessing ECG {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load ECG image (512x512 for ViT)
            image_tensor, original_image = self.load_ecg_image(image_path, (512, 512))
            if image_tensor is None:
                continue
            
            # Forward pass
            images = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(images)
            
            # Process outputs
            class_logits = outputs['class_logits']
            bbox_coords = outputs['bbox_coords'] 
            keypoint_coords = outputs['keypoint_coords']
            confidence_scores = outputs['confidence_scores']
            
            # Get high confidence detections
            confidence_threshold = 0.5
            high_conf_mask = confidence_scores.squeeze(-1) > confidence_threshold
            
            num_detections = high_conf_mask.sum().item()
            
            # Store results
            result = {
                'image_name': Path(image_path).name,
                'num_detections': num_detections,
                'total_queries': int(class_logits.shape[1]),
                'confidence_threshold': confidence_threshold,
                'max_confidence': float(confidence_scores.max().item()),
                'mean_confidence': float(confidence_scores.mean().item())
            }
            
            if num_detections > 0:
                # Get detected keypoints
                detected_keypoints = keypoint_coords[0][high_conf_mask[0]].cpu().numpy()
                detected_classes = class_logits[0][high_conf_mask[0]].argmax(dim=1).cpu().numpy()
                detected_confidences = confidence_scores[0][high_conf_mask[0]].cpu().numpy()
                
                result.update({
                    'detected_keypoints': detected_keypoints.tolist(),
                    'detected_classes': detected_classes.tolist(),
                    'detected_confidences': detected_confidences.tolist()
                })
            
            results.append(result)
            
            # Print results
            print(f"  🎯 Detections: {num_detections}/{result['total_queries']} above {confidence_threshold} confidence")
            print(f"  📊 Max Confidence: {result['max_confidence']:.3f}")
            print(f"  📈 Mean Confidence: {result['mean_confidence']:.3f}")
            
            # Save visualization
            self.save_vit_visualization(original_image, result, outputs, i+1)
        
        # Save all results
        with open(self.output_dir / 'vit_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ ViT testing complete! Results saved to {self.output_dir}/vit_results.json")
        return results
    
    def test_augmentation_on_real_data(self, image_paths: list, num_samples=3):
        """Test data augmentation on real ECG images."""
        print(f"\n🔄 Testing Data Augmentation on {len(image_paths)} real ECG images...")
        
        # Create augmentation pipeline
        transform = create_ecg_augmentation_pipeline(
            image_size=(224, 224),
            augmentation_type='medium',
            use_albumentations=False
        )
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"\nProcessing ECG {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load original image
            _, original_image = self.load_ecg_image(image_path, (224, 224))
            if original_image is None:
                continue
            
            # Apply augmentation multiple times
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'ECG Augmentation Results: {Path(image_path).name}', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')
            
            # Apply augmentations
            for idx in range(5):
                row = (idx + 1) // 3
                col = (idx + 1) % 3
                
                # Apply augmentation
                augmented_tensor = transform(original_image)
                if isinstance(augmented_tensor, torch.Tensor):
                    augmented_image = (augmented_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    axes[row, col].imshow(augmented_image)
                    axes[row, col].set_title(f'Augmented {idx+1}')
                    axes[row, col].axis('off')
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(self.output_dir / f'augmentation_real_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Augmentation visualization saved")
        
        print(f"\n✅ Augmentation testing complete!")
    
    def save_prediction_visualization(self, original_image, result, image_num):
        """Save multi-modal prediction visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title('Original ECG')
        ax1.axis('off')
        
        # Prediction probabilities
        diagnosis_probs = result['diagnosis_prediction'][0]
        class_names = ['Normal', 'AF', 'AFL', 'Brady', 'Tachy', 'AVB']
        
        bars = ax2.bar(range(len(diagnosis_probs)), diagnosis_probs)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45)
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Diagnosis Predictions\nPredicted: {class_names[result["predicted_diagnosis"]]}')
        ax2.grid(True, alpha=0.3)
        
        # Highlight predicted class
        bars[result['predicted_diagnosis']].set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'multimodal_prediction_{image_num}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_vit_visualization(self, original_image, result, outputs, image_num):
        """Save ViT detection visualization."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        ax.imshow(original_image)
        ax.set_title(f'ViT-DETR Detections: {result["num_detections"]} found')
        
        # Plot detected keypoints if any
        if 'detected_keypoints' in result:
            keypoints = np.array(result['detected_keypoints'])
            classes = result['detected_classes']
            confidences = result['detected_confidences']
            
            # Convert normalized coordinates to image coordinates
            h, w = original_image.size[::-1]
            keypoints_img = keypoints * [w, h]
            
            # Plot keypoints
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
            class_names = ['P', 'Q', 'R', 'S', 'T', 'Background']
            
            for i, (kp, cls, conf) in enumerate(zip(keypoints_img, classes, confidences)):
                color = colors[cls % len(colors)]
                ax.plot(kp[0], kp[1], 'o', color=color, markersize=8)
                ax.text(kp[0], kp[1]-10, f'{class_names[cls]}\n{conf[0]:.2f}', 
                       ha='center', va='bottom', fontsize=8, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'vit_detections_{image_num}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, multimodal_results, vit_results):
        """Generate comprehensive summary report."""
        print(f"\n📊 Generating Summary Report...")
        
        report = {
            'summary': {
                'total_images_processed': len(multimodal_results),
                'models_tested': ['MultiModal ECG', 'Vision Transformer DETR'],
                'test_date': str(Path(__file__).stat().st_mtime),
            },
            'multimodal_analysis': {
                'average_confidence': np.mean([r['confidence_score'][0] for r in multimodal_results]),
                'average_epistemic_uncertainty': np.mean([r['epistemic_uncertainty'][0] for r in multimodal_results]),
                'diagnosis_distribution': {},
                'severity_distribution': {}
            },
            'vit_analysis': {
                'average_detections_per_image': np.mean([r['num_detections'] for r in vit_results]),
                'average_max_confidence': np.mean([r['max_confidence'] for r in vit_results]),
                'images_with_detections': sum(1 for r in vit_results if r['num_detections'] > 0)
            }
        }
        
        # Count diagnosis predictions
        for result in multimodal_results:
            pred = result['predicted_diagnosis']
            if pred not in report['multimodal_analysis']['diagnosis_distribution']:
                report['multimodal_analysis']['diagnosis_distribution'][pred] = 0
            report['multimodal_analysis']['diagnosis_distribution'][pred] += 1
        
        # Count severity predictions
        for result in multimodal_results:
            pred = result['predicted_severity']
            if pred not in report['multimodal_analysis']['severity_distribution']:
                report['multimodal_analysis']['severity_distribution'][pred] = 0
            report['multimodal_analysis']['severity_distribution'][pred] += 1
        
        # Save report
        with open(self.output_dir / 'summary_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n📋 REAL DATA TESTING SUMMARY")
        print("=" * 60)
        print(f"📸 Images Processed: {report['summary']['total_images_processed']}")
        print(f"🔬 Multi-Modal Average Confidence: {report['multimodal_analysis']['average_confidence']:.3f}")
        print(f"🎯 Multi-Modal Average Uncertainty: {report['multimodal_analysis']['average_epistemic_uncertainty']:.4f}")
        print(f"🤖 ViT Average Detections/Image: {report['vit_analysis']['average_detections_per_image']:.1f}")
        print(f"🎯 ViT Average Max Confidence: {report['vit_analysis']['average_max_confidence']:.3f}")
        print(f"📊 Images with ViT Detections: {report['vit_analysis']['images_with_detections']}/{len(vit_results)}")
        print(f"\n✅ Full report saved to {self.output_dir}/summary_report.json")

def main():
    """Main function to test models on real ECG data."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Initialize tester
    tester = RealECGTester(device=device)
    
    # Find available ECG images
    data_dir = Path("data")
    image_paths = []
    
    # Check different data directories
    search_dirs = [
        data_dir / "raw" / "scanned_ecgs",
        data_dir / "processed" / "ptbxl" / "images",
        data_dir / "annotations" / "manual",
        Path("imgs")  # Additional images directory
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for img_file in search_dir.glob("*.png"):
                image_paths.append(str(img_file))
                if len(image_paths) >= 10:  # Limit for demo
                    break
        if len(image_paths) >= 10:
            break
    
    if not image_paths:
        print("❌ No ECG images found! Please ensure you have ECG images in:")
        for search_dir in search_dirs:
            print(f"  - {search_dir}")
        return
    
    print(f"📁 Found {len(image_paths)} ECG images for testing")
    print(f"📸 Sample images: {[Path(p).name for p in image_paths[:3]]}")
    
    # Test models on real data
    print(f"\n🚀 Starting Real Data Testing...")
    
    # Test multi-modal model
    multimodal_results = tester.test_multimodal_on_real_data(image_paths, num_samples=5)
    
    # Test Vision Transformer
    vit_results = tester.test_vit_on_real_data(image_paths, num_samples=5)
    
    # Test augmentation
    tester.test_augmentation_on_real_data(image_paths, num_samples=3)
    
    # Generate summary report
    tester.generate_summary_report(multimodal_results, vit_results)
    
    print(f"\n🎉 Real data testing complete!")
    print(f"📁 All results saved to: {tester.output_dir}")
    print(f"\n📋 Check these files:")
    print(f"  • multimodal_results.json - Detailed predictions")
    print(f"  • vit_results.json - PQRST detection results")
    print(f"  • summary_report.json - Overall analysis")
    print(f"  • *.png - Visualization results")

if __name__ == "__main__":
    main()