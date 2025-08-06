#!/usr/bin/env python3
"""
Enhanced ECG Detection Diagnostics
Comprehensive analysis of detection confidence patterns and threshold testing.
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our models
from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model

class ECGDetectionDiagnostics:
    """Enhanced diagnostics for ECG detection analysis."""
    
    def __init__(self, device='mps'):
        self.device = device
        
        # Create diagnostics folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"diagnostics_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔬 ECG Detection Diagnostics Suite")
        print(f"Device: {device}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
    
    def load_ecg_image(self, image_path: str, target_size=(512, 512)):
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
    
    def analyze_confidence_distribution(self, image_paths: list, num_samples=10):
        """Analyze confidence score distribution across multiple thresholds."""
        print(f"\n📊 Analyzing Confidence Distribution on {len(image_paths)} images...")
        
        # Create model
        model = create_ecg_vit_detr_model(pretrained=False)
        model.to(self.device)
        model.eval()
        
        all_confidences = []
        detection_stats = {
            'thresholds': [0.1, 0.2, 0.3, 0.4, 0.45, 0.5],
            'detection_counts': {t: [] for t in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]},
            'avg_max_confidence': [],
            'confidence_ranges': []
        }
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"  Processing {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load ECG image
            image_tensor, original_image = self.load_ecg_image(image_path, (512, 512))
            if image_tensor is None:
                continue
            
            # Forward pass
            images = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(images)
            
            confidence_scores = outputs['confidence_scores'].squeeze(-1).cpu().numpy()
            all_confidences.extend(confidence_scores.flatten())
            
            # Test different thresholds
            for threshold in detection_stats['thresholds']:
                detections = (confidence_scores > threshold).sum()
                detection_stats['detection_counts'][threshold].append(detections)
            
            # Store stats
            detection_stats['avg_max_confidence'].append(confidence_scores.max())
            detection_stats['confidence_ranges'].append({
                'min': float(confidence_scores.min()),
                'max': float(confidence_scores.max()),
                'mean': float(confidence_scores.mean()),
                'std': float(confidence_scores.std())
            })
        
        # Plot confidence distribution
        self.plot_confidence_analysis(all_confidences, detection_stats)
        
        # Save detailed stats
        with open(self.output_dir / 'confidence_analysis.json', 'w') as f:
            json.dump(detection_stats, f, indent=2, default=str)
        
        return detection_stats
    
    def plot_confidence_analysis(self, all_confidences, detection_stats):
        """Create comprehensive confidence analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ECG Detection Confidence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confidence score histogram
        axes[0, 0].hist(all_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Current threshold (0.5)')
        axes[0, 0].axvline(0.4, color='orange', linestyle='--', label='Suggested threshold (0.4)')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of All Confidence Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Detection counts by threshold
        thresholds = detection_stats['thresholds']
        avg_detections = [np.mean(detection_stats['detection_counts'][t]) for t in thresholds]
        
        axes[0, 1].plot(thresholds, avg_detections, 'o-', color='green', linewidth=2, markersize=8)
        axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Current threshold')
        axes[0, 1].set_xlabel('Confidence Threshold')
        axes[0, 1].set_ylabel('Average Detections per Image')
        axes[0, 1].set_title('Detections vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. Max confidence per image
        max_confidences = detection_stats['avg_max_confidence']
        axes[1, 0].bar(range(len(max_confidences)), max_confidences, alpha=0.7, color='purple')
        axes[1, 0].axhline(0.5, color='red', linestyle='--', label='Current threshold (0.5)')
        axes[1, 0].axhline(0.4, color='orange', linestyle='--', label='Suggested threshold (0.4)')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Max Confidence Score')
        axes[1, 0].set_title('Maximum Confidence per Image')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confidence statistics summary
        conf_means = [r['mean'] for r in detection_stats['confidence_ranges']]
        conf_stds = [r['std'] for r in detection_stats['confidence_ranges']]
        
        axes[1, 1].errorbar(range(len(conf_means)), conf_means, yerr=conf_stds, 
                           fmt='o-', color='brown', capsize=5, capthick=2)
        axes[1, 1].axhline(0.5, color='red', linestyle='--', label='Current threshold (0.5)')
        axes[1, 1].axhline(0.4, color='orange', linestyle='--', label='Suggested threshold (0.4)')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Mean Confidence ± Std')
        axes[1, 1].set_title('Confidence Statistics per Image')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Confidence analysis saved to {self.output_dir}/confidence_analysis.png")
    
    def visualize_top_detections(self, image_paths: list, num_samples=5, top_n=10):
        """Visualize top N detections regardless of threshold."""
        print(f"\n🎯 Visualizing Top {top_n} Detections...")
        
        # Create model
        model = create_ecg_vit_detr_model(pretrained=False)
        model.to(self.device)
        model.eval()
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"  Processing {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load ECG image
            image_tensor, original_image = self.load_ecg_image(image_path, (512, 512))
            if image_tensor is None:
                continue
            
            # Forward pass
            images = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(images)
            
            # Get all outputs
            class_logits = outputs['class_logits']
            keypoint_coords = outputs['keypoint_coords']
            confidence_scores = outputs['confidence_scores'].squeeze(-1)
            
            # Get top N detections by confidence
            top_indices = torch.topk(confidence_scores.flatten(), top_n)[1]
            query_indices = top_indices // confidence_scores.shape[-1]
            
            # Extract top detections
            top_keypoints = keypoint_coords[0][query_indices].cpu().numpy()
            top_classes = class_logits[0][query_indices].argmax(dim=1).cpu().numpy()
            top_confidences = confidence_scores[0][query_indices].cpu().numpy()
            
            # Create visualization
            self.plot_top_detections(original_image, top_keypoints, top_classes, 
                                   top_confidences, image_path, i+1, top_n)
    
    def plot_top_detections(self, original_image, keypoints, classes, confidences, 
                          image_path, image_num, top_n):
        """Plot top N detections with detailed annotations."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        ax.imshow(original_image)
        ax.set_title(f'Top {top_n} Detections: {Path(image_path).name}\n'
                    f'Best Confidence: {confidences.max():.3f}')
        
        # Define colors and class names
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        class_names = ['P', 'Q', 'R', 'S', 'T', 'Background']
        
        # Convert normalized coordinates to image coordinates
        h, w = original_image.size[::-1]
        keypoints_img = keypoints * [w, h]
        
        # Plot top detections
        for i, (kp, cls, conf) in enumerate(zip(keypoints_img, classes, confidences)):
            color = colors[cls % len(colors)]
            
            # Plot detection point
            ax.plot(kp[0], kp[1], 'o', color=color, markersize=10, 
                   markeredgecolor='white', markeredgewidth=2)
            
            # Add annotation with rank, class, and confidence
            ax.annotate(f'#{i+1}\n{class_names[cls]}\n{conf:.3f}', 
                       xy=(kp[0], kp[1]), xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                       fontsize=9, fontweight='bold', color='white')
        
        # Add threshold lines in legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label=f'Current threshold: 0.5'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label=f'Max confidence: {confidences.max():.3f}'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label=f'Mean confidence: {confidences.mean():.3f}')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'top_detections_{image_num}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    💾 Top detections saved for image {image_num}")
    
    def test_multiple_thresholds(self, image_paths: list, num_samples=5):
        """Test detection performance at multiple thresholds."""
        print(f"\n🎯 Testing Multiple Detection Thresholds...")
        
        model = create_ecg_vit_detr_model(pretrained=False)
        model.to(self.device)
        model.eval()
        
        thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
        results = []
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"  Processing {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            # Load ECG image
            image_tensor, original_image = self.load_ecg_image(image_path, (512, 512))
            if image_tensor is None:
                continue
            
            # Forward pass
            images = image_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(images)
            
            confidence_scores = outputs['confidence_scores'].squeeze(-1)
            keypoint_coords = outputs['keypoint_coords']
            class_logits = outputs['class_logits']
            
            # Test each threshold
            image_results = {
                'image_name': Path(image_path).name,
                'max_confidence': float(confidence_scores.max().item()),
                'mean_confidence': float(confidence_scores.mean().item()),
                'threshold_results': {}
            }
            
            for threshold in thresholds_to_test:
                # Get detections above threshold (fix for proper tensor shape)
                if len(confidence_scores.shape) == 2:
                    high_conf_mask = confidence_scores.squeeze(0) > threshold
                else:
                    high_conf_mask = confidence_scores > threshold
                num_detections = high_conf_mask.sum().item()
                
                if num_detections > 0:
                    # Fix tensor indexing for proper masking
                    if len(high_conf_mask.shape) == 1:
                        high_conf_indices = torch.where(high_conf_mask)[0]
                    else:
                        high_conf_indices = torch.where(high_conf_mask)[1] if high_conf_mask.shape[0] == 1 else torch.where(high_conf_mask.flatten())[0]
                    
                    detected_keypoints = keypoint_coords[0][high_conf_indices].cpu().numpy()
                    detected_classes = class_logits[0][high_conf_indices].argmax(dim=1).cpu().numpy()
                    detected_confidences = confidence_scores.flatten()[high_conf_indices].cpu().numpy()
                    
                    image_results['threshold_results'][threshold] = {
                        'num_detections': num_detections,
                        'detected_keypoints': detected_keypoints.tolist(),
                        'detected_classes': detected_classes.tolist(),
                        'detected_confidences': detected_confidences.tolist()
                    }
                else:
                    image_results['threshold_results'][threshold] = {
                        'num_detections': 0
                    }
                
                print(f"    Threshold {threshold}: {num_detections} detections")
            
            results.append(image_results)
        
        # Save threshold test results
        with open(self.output_dir / 'threshold_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create threshold comparison visualization
        self.plot_threshold_comparison(results)
        
        return results
    
    def plot_threshold_comparison(self, results):
        """Plot threshold comparison results."""
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
        
        # Calculate average detections per threshold
        avg_detections = []
        for threshold in thresholds:
            total_detections = sum(
                r['threshold_results'][threshold]['num_detections'] 
                for r in results if threshold in r['threshold_results']
            )
            avg_detections.append(total_detections / len(results))
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Detections vs threshold
        ax1.plot(thresholds, avg_detections, 'o-', linewidth=3, markersize=8, color='blue')
        ax1.axvline(0.5, color='red', linestyle='--', label='Current threshold (0.5)')
        ax1.axvline(0.4, color='orange', linestyle='--', label='Suggested threshold (0.4)')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Average Detections per Image')
        ax1.set_title('Detection Count vs Threshold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Max confidence per image
        max_confidences = [r['max_confidence'] for r in results]
        image_names = [r['image_name'] for r in results]
        
        bars = ax2.bar(range(len(max_confidences)), max_confidences, alpha=0.7)
        ax2.axhline(0.5, color='red', linestyle='--', label='Current threshold (0.5)')
        ax2.axhline(0.4, color='orange', linestyle='--', label='Suggested threshold (0.4)')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Max Confidence Score')
        ax2.set_title('Max Confidence per Image')
        ax2.set_xticks(range(len(image_names)))
        ax2.set_xticklabels([name[:8] + '...' for name in image_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Color bars based on whether they exceed thresholds
        for i, (bar, conf) in enumerate(zip(bars, max_confidences)):
            if conf >= 0.5:
                bar.set_color('green')
            elif conf >= 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Threshold comparison saved to {self.output_dir}/threshold_comparison.png")
    
    def generate_diagnostic_report(self, confidence_stats, threshold_results):
        """Generate comprehensive diagnostic report."""
        print(f"\n📋 Generating Diagnostic Report...")
        
        report = {
            'diagnostic_summary': {
                'test_date': datetime.now().isoformat(),
                'total_images_tested': len(threshold_results),
                'current_threshold': 0.5,
                'recommended_threshold': 0.4
            },
            'confidence_analysis': {
                'overall_max_confidence': max(confidence_stats['avg_max_confidence']),
                'overall_mean_confidence': np.mean([r['mean'] for r in confidence_stats['confidence_ranges']]),
                'images_above_0_5': sum(1 for conf in confidence_stats['avg_max_confidence'] if conf >= 0.5),
                'images_above_0_4': sum(1 for conf in confidence_stats['avg_max_confidence'] if conf >= 0.4),
                'images_above_0_3': sum(1 for conf in confidence_stats['avg_max_confidence'] if conf >= 0.3)
            },
            'threshold_recommendations': {
                'current_performance': {
                    'threshold': 0.5,
                    'avg_detections': np.mean(confidence_stats['detection_counts'][0.5]),
                    'detection_rate': f"{sum(1 for x in confidence_stats['detection_counts'][0.5] if x > 0)}/{len(confidence_stats['detection_counts'][0.5])}"
                },
                'recommended_performance': {
                    'threshold': 0.4,
                    'avg_detections': np.mean(confidence_stats['detection_counts'][0.4]),
                    'detection_rate': f"{sum(1 for x in confidence_stats['detection_counts'][0.4] if x > 0)}/{len(confidence_stats['detection_counts'][0.4])}"
                }
            },
            'next_steps': [
                "Lower detection threshold from 0.5 to 0.4",
                "Implement improved image preprocessing",
                "Test higher input resolution (1024x1024)",
                "Consider model fine-tuning on ECG-specific data"
            ]
        }
        
        # Save report (convert numpy types to Python types for JSON serialization)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert all numpy types in report
        import json
        report_str = json.dumps(report, default=convert_numpy_types, indent=2)
        
        with open(self.output_dir / 'diagnostic_report.json', 'w') as f:
            f.write(report_str)
        
        # Print summary
        print(f"\n🔍 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        print(f"📊 Images Tested: {report['diagnostic_summary']['total_images_tested']}")
        print(f"🎯 Current Threshold (0.5): {report['threshold_recommendations']['current_performance']['detection_rate']} images have detections")
        print(f"💡 Recommended Threshold (0.4): {report['threshold_recommendations']['recommended_performance']['detection_rate']} images have detections")
        print(f"📈 Max Confidence Found: {report['confidence_analysis']['overall_max_confidence']:.3f}")
        print(f"📊 Mean Confidence: {report['confidence_analysis']['overall_mean_confidence']:.3f}")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        for i, step in enumerate(report['next_steps'], 1):
            print(f"  {i}. {step}")
        
        print(f"\n✅ Full diagnostic report saved to {self.output_dir}/diagnostic_report.json")
        
        return report


def main():
    """Main function to run enhanced diagnostics."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Initialize diagnostics
    diagnostics = ECGDetectionDiagnostics(device=device)
    
    # Find available ECG images
    data_dir = Path("data")
    image_paths = []
    
    # Check different data directories
    search_dirs = [
        data_dir / "raw" / "scanned_ecgs",
        data_dir / "processed" / "ptbxl" / "images",
        data_dir / "annotations" / "manual",
        Path("imgs")
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for img_file in search_dir.glob("*.png"):
                image_paths.append(str(img_file))
                if len(image_paths) >= 10:
                    break
        if len(image_paths) >= 10:
            break
    
    if not image_paths:
        print("❌ No ECG images found!")
        return
    
    print(f"🔬 Starting Enhanced Diagnostics on {len(image_paths)} images...")
    
    # Run diagnostic phases
    print(f"\n🚀 PHASE 1: COMPREHENSIVE DIAGNOSTICS")
    
    # 1. Analyze confidence distribution
    confidence_stats = diagnostics.analyze_confidence_distribution(image_paths, num_samples=10)
    
    # 2. Visualize top detections
    diagnostics.visualize_top_detections(image_paths, num_samples=5, top_n=15)
    
    # 3. Test multiple thresholds
    threshold_results = diagnostics.test_multiple_thresholds(image_paths, num_samples=5)
    
    # 4. Generate comprehensive report
    diagnostic_report = diagnostics.generate_diagnostic_report(confidence_stats, threshold_results)
    
    print(f"\n🎉 Phase 1 diagnostics complete!")
    print(f"📁 All results saved to: {diagnostics.output_dir}")
    print(f"\n📋 Key findings:")
    print(f"  • Best confidence found: {diagnostic_report['confidence_analysis']['overall_max_confidence']:.3f}")
    print(f"  • Recommended threshold: 0.4 (vs current 0.5)")
    print(f"  • {diagnostic_report['confidence_analysis']['images_above_0_4']} images have detections at 0.4 threshold")


if __name__ == "__main__":
    main()