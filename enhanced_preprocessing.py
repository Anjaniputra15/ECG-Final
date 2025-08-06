#!/usr/bin/env python3
"""
Enhanced ECG Image Preprocessing Pipeline
Advanced preprocessing to boost model confidence and detection accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as transforms
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ECGImagePreprocessor:
    """Advanced ECG image preprocessing for better model performance."""
    
    def __init__(self):
        # ImageNet normalization constants
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        
        # ECG-specific processing parameters
        self.grid_removal_params = {
            'horizontal_kernel_size': (40, 1),
            'vertical_kernel_size': (1, 40),
            'grid_threshold': 30
        }
        
        print("🔧 Enhanced ECG Preprocessing Pipeline Initialized")
        print("✅ ImageNet normalization ready")
        print("✅ Grid removal algorithms loaded")
        print("✅ Background inversion ready")
    
    def load_and_preprocess_ecg(self, image_path: str, target_size=(512, 512), 
                               preprocessing_level='enhanced'):
        """
        Load and apply comprehensive ECG preprocessing.
        
        Args:
            image_path: Path to ECG image
            target_size: Output image size
            preprocessing_level: 'basic', 'enhanced', 'aggressive'
        """
        try:
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            
            if preprocessing_level == 'basic':
                processed_image = self._basic_preprocessing(original_image, target_size)
            elif preprocessing_level == 'enhanced':
                processed_image = self._enhanced_preprocessing(original_image, target_size)
            elif preprocessing_level == 'aggressive':
                processed_image = self._aggressive_preprocessing(original_image, target_size)
            else:
                raise ValueError(f"Unknown preprocessing level: {preprocessing_level}")
            
            return processed_image, original_image
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None
    
    def _basic_preprocessing(self, image: Image.Image, target_size):
        """Basic preprocessing: resize + normalize."""
        # Resize
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # ImageNet normalization
        image_tensor = self._apply_imagenet_normalization(image_tensor)
        
        return image_tensor
    
    def _enhanced_preprocessing(self, image: Image.Image, target_size):
        """Enhanced preprocessing: grid removal + contrast + normalization."""
        # Step 1: Convert to numpy for OpenCV operations
        img_np = np.array(image)
        
        # Step 2: Remove grid lines
        img_no_grid = self._remove_grid_lines(img_np)
        
        # Step 3: Enhance contrast
        img_enhanced = self._enhance_contrast(img_no_grid)
        
        # Step 4: Background inversion (optional)
        img_inverted = self._invert_background(img_enhanced)
        
        # Step 5: Convert back to PIL and resize
        processed_pil = Image.fromarray(img_inverted.astype(np.uint8))
        processed_pil = processed_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        # Step 6: Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(processed_pil)).permute(2, 0, 1).float() / 255.0
        image_tensor = self._apply_imagenet_normalization(image_tensor)
        
        return image_tensor
    
    def _aggressive_preprocessing(self, image: Image.Image, target_size):
        """Aggressive preprocessing: full pipeline + noise reduction."""
        # Start with enhanced preprocessing
        img_np = np.array(image)
        
        # Advanced grid removal
        img_no_grid = self._advanced_grid_removal(img_np)
        
        # Noise reduction
        img_denoised = self._reduce_noise(img_no_grid)
        
        # Contrast enhancement
        img_enhanced = self._aggressive_contrast_enhancement(img_denoised)
        
        # Background inversion
        img_inverted = self._invert_background(img_enhanced)
        
        # Morphological operations for trace cleanup
        img_clean = self._morphological_cleanup(img_inverted)
        
        # Convert and resize
        processed_pil = Image.fromarray(img_clean.astype(np.uint8))
        processed_pil = processed_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        # Tensor conversion and normalization
        image_tensor = torch.from_numpy(np.array(processed_pil)).permute(2, 0, 1).float() / 255.0
        image_tensor = self._apply_imagenet_normalization(image_tensor)
        
        return image_tensor
    
    def _remove_grid_lines(self, image_np):
        """Remove horizontal and vertical grid lines from ECG."""
        # Convert to grayscale for grid detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Create kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                     self.grid_removal_params['horizontal_kernel_size'])
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                   self.grid_removal_params['vertical_kernel_size'])
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine line masks
        grid_mask = cv2.addWeighted(horizontal_lines, 1.0, vertical_lines, 1.0, 0.0)
        
        # Threshold to create binary mask
        _, grid_binary = cv2.threshold(grid_mask, self.grid_removal_params['grid_threshold'], 255, cv2.THRESH_BINARY)
        
        # Inpaint to remove grid lines
        result = image_np.copy()
        if len(image_np.shape) == 3:
            for channel in range(3):
                result[:, :, channel] = cv2.inpaint(image_np[:, :, channel], grid_binary, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def _advanced_grid_removal(self, image_np):
        """Advanced grid removal with multiple kernel sizes."""
        result = image_np.copy()
        
        # Multiple passes with different kernel sizes
        kernel_sizes = [(20, 1), (1, 20), (40, 1), (1, 40), (60, 1), (1, 60)]
        
        for h_size, v_size in kernel_sizes:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            
            # Horizontal lines
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, v_size))
            h_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
            
            # Vertical lines  
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (v_size, h_size))
            v_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, v_kernel)
            
            # Combine and inpaint
            grid_mask = cv2.addWeighted(h_lines, 1.0, v_lines, 1.0, 0.0)
            _, grid_binary = cv2.threshold(grid_mask, 25, 255, cv2.THRESH_BINARY)
            
            if len(result.shape) == 3:
                for channel in range(3):
                    result[:, :, channel] = cv2.inpaint(result[:, :, channel], grid_binary, 2, cv2.INPAINT_TELEA)
        
        return result
    
    def _enhance_contrast(self, image_np):
        """Enhance contrast using CLAHE."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _aggressive_contrast_enhancement(self, image_np):
        """More aggressive contrast enhancement."""
        # CLAHE on multiple color spaces
        result = image_np.copy()
        
        # LAB space enhancement
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Additional histogram equalization on grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        # Blend with original
        result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        blended = cv2.addWeighted(result_gray, 0.7, equalized, 0.3, 0)
        
        # Convert back to RGB
        result = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _invert_background(self, image_np, invert=True):
        """Invert background to make ECG traces more prominent."""
        if not invert:
            return image_np
        
        # Convert to grayscale for background detection
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Threshold to separate background from traces
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert the image
        inverted = 255 - image_np
        
        return inverted
    
    def _reduce_noise(self, image_np):
        """Reduce noise using bilateral filtering."""
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(image_np, 9, 75, 75)
        return filtered
    
    def _morphological_cleanup(self, image_np):
        """Clean up traces using morphological operations."""
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological opening to clean up traces
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Convert back to RGB
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        return result
    
    def _apply_imagenet_normalization(self, image_tensor):
        """Apply ImageNet normalization."""
        # Ensure tensor is float and in [0, 1] range
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        
        # Apply normalization
        for i in range(3):  # RGB channels
            image_tensor[i] = (image_tensor[i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        
        return image_tensor
    
    def create_preprocessing_comparison(self, image_path: str, output_dir: Path):
        """Create side-by-side comparison of preprocessing levels."""
        original_pil = Image.open(image_path).convert('RGB')
        
        # Apply different preprocessing levels
        basic_tensor, _ = self.load_and_preprocess_ecg(image_path, preprocessing_level='basic')
        enhanced_tensor, _ = self.load_and_preprocess_ecg(image_path, preprocessing_level='enhanced')
        aggressive_tensor, _ = self.load_and_preprocess_ecg(image_path, preprocessing_level='aggressive')
        
        # Convert tensors back to displayable images (denormalize)
        def denormalize_tensor(tensor):
            denorm = tensor.clone()
            for i in range(3):
                denorm[i] = denorm[i] * self.imagenet_std[i] + self.imagenet_mean[i]
            denorm = torch.clamp(denorm, 0, 1)
            return denorm.permute(1, 2, 0).numpy()
        
        basic_img = denormalize_tensor(basic_tensor)
        enhanced_img = denormalize_tensor(enhanced_tensor)
        aggressive_img = denormalize_tensor(aggressive_tensor)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'ECG Preprocessing Comparison: {Path(image_path).name}', 
                    fontsize=16, fontweight='bold')
        
        # Original
        axes[0, 0].imshow(original_pil)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Basic preprocessing
        axes[0, 1].imshow(basic_img)
        axes[0, 1].set_title('Basic Preprocessing\n(Resize + ImageNet Norm)')
        axes[0, 1].axis('off')
        
        # Enhanced preprocessing
        axes[1, 0].imshow(enhanced_img)
        axes[1, 0].set_title('Enhanced Preprocessing\n(Grid Removal + Contrast)')
        axes[1, 0].axis('off')
        
        # Aggressive preprocessing
        axes[1, 1].imshow(aggressive_img)
        axes[1, 1].set_title('Aggressive Preprocessing\n(Full Pipeline + Cleanup)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        comparison_path = output_dir / f'preprocessing_comparison_{Path(image_path).stem}.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Preprocessing comparison saved: {comparison_path.name}")
        
        return {
            'basic': basic_tensor,
            'enhanced': enhanced_tensor, 
            'aggressive': aggressive_tensor,
            'comparison_saved': str(comparison_path)
        }


class ECGPreprocessingTester:
    """Test different preprocessing approaches on ECG detection."""
    
    def __init__(self, device='mps'):
        self.device = device
        self.preprocessor = ECGImagePreprocessor()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"preprocessing_test_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧪 ECG Preprocessing Tester Initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def test_preprocessing_impact(self, image_paths: list, num_samples=5):
        """Test the impact of different preprocessing on detection performance."""
        print(f"\n🔬 Testing Preprocessing Impact on {len(image_paths)} images...")
        
        # Import model
        import sys
        current_dir = Path(__file__).parent
        sys.path.append(str(current_dir))
        from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
        
        # Create model
        model = create_ecg_vit_detr_model(pretrained=False)
        model.to(self.device)
        model.eval()
        
        results = []
        preprocessing_levels = ['basic', 'enhanced', 'aggressive']
        
        for i, image_path in enumerate(image_paths[:num_samples]):
            print(f"\n  Processing image {i+1}/{min(num_samples, len(image_paths))}: {Path(image_path).name}")
            
            image_results = {
                'image_name': Path(image_path).name,
                'preprocessing_results': {}
            }
            
            # Test each preprocessing level
            for level in preprocessing_levels:
                print(f"    Testing {level} preprocessing...")
                
                # Load and preprocess image
                processed_tensor, original_pil = self.preprocessor.load_and_preprocess_ecg(
                    image_path, target_size=(512, 512), preprocessing_level=level
                )
                
                if processed_tensor is None:
                    continue
                
                # Run through model
                images = processed_tensor.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = model(images)
                
                # Analyze results
                confidence_scores = outputs['confidence_scores'].squeeze(-1)
                
                # Test multiple thresholds
                thresholds = [0.3, 0.4, 0.45, 0.5]
                threshold_results = {}
                
                for threshold in thresholds:
                    if len(confidence_scores.shape) == 2:
                        high_conf_mask = confidence_scores.squeeze(0) > threshold
                    else:
                        high_conf_mask = confidence_scores > threshold
                    num_detections = high_conf_mask.sum().item()
                    threshold_results[threshold] = num_detections
                
                # Store results
                image_results['preprocessing_results'][level] = {
                    'max_confidence': float(confidence_scores.max().item()),
                    'mean_confidence': float(confidence_scores.mean().item()),
                    'std_confidence': float(confidence_scores.std().item()),
                    'threshold_detections': threshold_results
                }
                
                print(f"      Max confidence: {confidence_scores.max().item():.3f}")
                print(f"      Detections at 0.4: {threshold_results[0.4]}")
                print(f"      Detections at 0.5: {threshold_results[0.5]}")
            
            # Create preprocessing comparison visualization
            comparison_results = self.preprocessor.create_preprocessing_comparison(
                image_path, self.output_dir
            )
            image_results['comparison_file'] = comparison_results['comparison_saved']
            
            results.append(image_results)
        
        # Save detailed results
        import json
        with open(self.output_dir / 'preprocessing_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create performance comparison plots
        self.plot_preprocessing_performance(results)
        
        return results
    
    def plot_preprocessing_performance(self, results):
        """Plot preprocessing performance comparison."""
        preprocessing_levels = ['basic', 'enhanced', 'aggressive']
        metrics = ['max_confidence', 'mean_confidence']
        thresholds = [0.3, 0.4, 0.45, 0.5]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Preprocessing Impact on ECG Detection Performance', fontsize=16, fontweight='bold')
        
        # 1. Max confidence comparison
        max_confidences = {level: [] for level in preprocessing_levels}
        for result in results:
            for level in preprocessing_levels:
                if level in result['preprocessing_results']:
                    max_confidences[level].append(result['preprocessing_results'][level]['max_confidence'])
        
        x_pos = np.arange(len(results))
        width = 0.25
        
        for i, level in enumerate(preprocessing_levels):
            axes[0, 0].bar(x_pos + i*width, max_confidences[level], width, 
                          label=level.capitalize(), alpha=0.8)
        
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='Target (0.5)')
        axes[0, 0].axhline(y=0.4, color='orange', linestyle='--', label='Minimum (0.4)')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Max Confidence Score')
        axes[0, 0].set_title('Max Confidence by Preprocessing Level')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mean confidence comparison
        mean_confidences = {level: [] for level in preprocessing_levels}
        for result in results:
            for level in preprocessing_levels:
                if level in result['preprocessing_results']:
                    mean_confidences[level].append(result['preprocessing_results'][level]['mean_confidence'])
        
        for i, level in enumerate(preprocessing_levels):
            axes[0, 1].bar(x_pos + i*width, mean_confidences[level], width, 
                          label=level.capitalize(), alpha=0.8)
        
        axes[0, 1].set_xlabel('Image Index')
        axes[0, 1].set_ylabel('Mean Confidence Score')
        axes[0, 1].set_title('Mean Confidence by Preprocessing Level')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Detections at threshold 0.4
        detections_04 = {level: [] for level in preprocessing_levels}
        for result in results:
            for level in preprocessing_levels:
                if level in result['preprocessing_results']:
                    detections_04[level].append(result['preprocessing_results'][level]['threshold_detections'][0.4])
        
        for i, level in enumerate(preprocessing_levels):
            axes[1, 0].bar(x_pos + i*width, detections_04[level], width, 
                          label=level.capitalize(), alpha=0.8)
        
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Number of Detections')
        axes[1, 0].set_title('Detections at Threshold 0.4')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Detections at threshold 0.5
        detections_05 = {level: [] for level in preprocessing_levels}
        for result in results:
            for level in preprocessing_levels:
                if level in result['preprocessing_results']:
                    detections_05[level].append(result['preprocessing_results'][level]['threshold_detections'][0.5])
        
        for i, level in enumerate(preprocessing_levels):
            axes[1, 1].bar(x_pos + i*width, detections_05[level], width, 
                          label=level.capitalize(), alpha=0.8)
        
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Number of Detections')
        axes[1, 1].set_title('Detections at Threshold 0.5')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessing_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance comparison saved: preprocessing_performance_comparison.png")
    
    def generate_preprocessing_report(self, results):
        """Generate comprehensive preprocessing performance report."""
        print(f"\n📋 Generating Preprocessing Report...")
        
        preprocessing_levels = ['basic', 'enhanced', 'aggressive']
        
        # Calculate average improvements
        improvements = {}
        
        for level in preprocessing_levels:
            level_max_confs = []
            level_detections_04 = []
            level_detections_05 = []
            
            for result in results:
                if level in result['preprocessing_results']:
                    level_max_confs.append(result['preprocessing_results'][level]['max_confidence'])
                    level_detections_04.append(result['preprocessing_results'][level]['threshold_detections'][0.4])
                    level_detections_05.append(result['preprocessing_results'][level]['threshold_detections'][0.5])
            
            improvements[level] = {
                'avg_max_confidence': np.mean(level_max_confs),
                'avg_detections_04': np.mean(level_detections_04),
                'avg_detections_05': np.mean(level_detections_05),
                'images_above_04': sum(1 for conf in level_max_confs if conf >= 0.4),
                'images_above_05': sum(1 for conf in level_max_confs if conf >= 0.5)
            }
        
        # Create report
        report = {
            'test_summary': {
                'test_date': datetime.now().isoformat(),
                'total_images_tested': len(results),
                'preprocessing_levels_tested': preprocessing_levels
            },
            'performance_comparison': improvements,
            'recommendations': {
                'best_for_confidence': max(improvements.keys(), 
                                         key=lambda x: improvements[x]['avg_max_confidence']),
                'best_for_detections': max(improvements.keys(), 
                                         key=lambda x: improvements[x]['avg_detections_05']),
                'recommended_level': 'enhanced'  # Balance between performance and speed
            }
        }
        
        # Save report
        import json
        with open(self.output_dir / 'preprocessing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n🔧 PREPROCESSING PERFORMANCE REPORT")
        print("=" * 60)
        print(f"📊 Images Tested: {len(results)}")
        
        for level in preprocessing_levels:
            stats = improvements[level]
            print(f"\n🔸 {level.upper()} PREPROCESSING:")
            print(f"   Average Max Confidence: {stats['avg_max_confidence']:.3f}")
            print(f"   Average Detections (0.4): {stats['avg_detections_04']:.1f}")
            print(f"   Average Detections (0.5): {stats['avg_detections_05']:.1f}")
            print(f"   Images above 0.4 confidence: {stats['images_above_04']}/{len(results)}")
            print(f"   Images above 0.5 confidence: {stats['images_above_05']}/{len(results)}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   Best for confidence: {report['recommendations']['best_for_confidence']}")
        print(f"   Best for detections: {report['recommendations']['best_for_detections']}")
        print(f"   Recommended overall: {report['recommendations']['recommended_level']}")
        
        print(f"\n✅ Full report saved to {self.output_dir}/preprocessing_report.json")
        
        return report


def main():
    """Test different preprocessing approaches."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Initialize tester
    tester = ECGPreprocessingTester(device=device)
    
    # Find ECG images
    data_dir = Path("data")
    image_paths = []
    
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
    
    print(f"🚀 PHASE 2: PREPROCESSING ENHANCEMENT")
    print(f"🔬 Testing preprocessing on {len(image_paths)} images...")
    
    # Test preprocessing impact
    results = tester.test_preprocessing_impact(image_paths, num_samples=5)
    
    # Generate comprehensive report
    report = tester.generate_preprocessing_report(results)
    
    print(f"\n🎉 Phase 2 preprocessing testing complete!")
    print(f"📁 All results saved to: {tester.output_dir}")


if __name__ == "__main__":
    main()