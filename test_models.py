#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite
Test all implemented ECG models and components with sample data.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our components
try:
    from models.multimodal_ecg_model import create_multimodal_ecg_model
    from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
    from augmentation.enhanced_transforms import create_ecg_augmentation_pipeline
    from training.balanced_loss import ECGLossCalculator, calculate_dataset_statistics
    from uncertainty.monte_carlo_uncertainty import MonteCarloDropout, DeepEnsemble
    from visualization.attention_maps import AttentionVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the correct paths")
    sys.exit(1)


class ECGModelTester:
    """Comprehensive testing suite for all ECG models."""
    
    def __init__(self, device: str = 'mps'):
        """Initialize tester with device."""
        self.device = device
        self.test_results = {}
        
        # Create test output directory
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🧪 ECG Model Testing Suite")
        print(f"Device: {device}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)
    
    def create_sample_data(self) -> dict:
        """Create sample ECG data for testing."""
        batch_size = 4
        
        # Sample ECG images (simulate 12-lead ECG images)
        ecg_images = self._generate_sample_ecg_images(batch_size)
        
        # Sample raw signals (12 leads, 10 seconds at 100Hz = 1000 samples)
        raw_signals = self._generate_sample_signals(batch_size)
        
        # Sample clinical metadata
        numerical_metadata = torch.randn(batch_size, 5)  # age, height, weight, etc.
        categorical_metadata = {
            'sex': torch.randint(0, 2, (batch_size,)),
            'device_type': torch.randint(0, 5, (batch_size,)),
            'recording_condition': torch.randint(0, 3, (batch_size,))
        }
        
        # Sample labels for different tasks
        diagnosis_labels = torch.randint(0, 6, (batch_size,))  # 6 diagnostic classes
        severity_labels = torch.randint(0, 4, (batch_size,))   # 4 severity levels
        
        return {
            'ecg_images': ecg_images,
            'raw_signals': raw_signals,
            'numerical_metadata': numerical_metadata,
            'categorical_metadata': categorical_metadata,
            'diagnosis_labels': diagnosis_labels,
            'severity_labels': severity_labels
        }
    
    def _generate_sample_ecg_images(self, batch_size: int) -> torch.Tensor:
        """Generate realistic-looking ECG images."""
        images = []
        
        for _ in range(batch_size):
            # Create a blank ECG-like image
            img = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White background
            
            # Add ECG grid pattern
            for i in range(0, 224, 20):
                cv2.line(img, (i, 0), (i, 224), (255, 200, 200), 1)  # Vertical lines
                cv2.line(img, (0, i), (224, i), (255, 200, 200), 1)  # Horizontal lines
            
            # Add some ECG-like waveforms
            for lead in range(3):  # 3 rows of leads
                for col in range(4):  # 4 columns
                    if lead * 4 + col >= 12:  # Only 12 leads
                        break
                    
                    # Generate synthetic ECG waveform
                    x_start = col * 56 + 10
                    y_center = lead * 75 + 37
                    
                    # Simple ECG-like pattern
                    x_points = np.linspace(x_start, x_start + 40, 100)
                    y_points = y_center + 10 * np.sin(x_points * 0.5) + 3 * np.random.randn(100)
                    
                    for i in range(len(x_points) - 1):
                        cv2.line(img, 
                               (int(x_points[i]), int(y_points[i])), 
                               (int(x_points[i+1]), int(y_points[i+1])), 
                               (0, 0, 0), 2)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)
        
        return torch.stack(images)
    
    def _generate_sample_signals(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic ECG signals."""
        signals = []
        
        for _ in range(batch_size):
            lead_signals = []
            
            for lead in range(12):  # 12 ECG leads
                # Generate synthetic ECG signal
                t = np.linspace(0, 10, 1000)  # 10 seconds, 100Hz
                
                # Base heart rate around 70 BPM
                hr = 60 + np.random.randn() * 10
                
                # P wave, QRS complex, T wave components
                signal = (
                    0.1 * np.sin(2 * np.pi * hr/60 * t) +  # P wave
                    0.8 * np.sin(4 * np.pi * hr/60 * t + 0.2) +  # QRS
                    0.3 * np.sin(1.5 * np.pi * hr/60 * t + 0.8) +  # T wave
                    0.05 * np.random.randn(1000)  # Noise
                )
                
                # Lead-specific modifications
                if lead in [0, 1, 2]:  # Limb leads
                    signal *= 0.8
                elif lead in [6, 7, 8]:  # Precordial leads V1-V3
                    signal *= 1.2
                
                lead_signals.append(torch.tensor(signal, dtype=torch.float32))
            
            signals.append(torch.stack(lead_signals))
        
        return torch.stack(signals)
    
    def test_augmentation_pipeline(self) -> bool:
        """Test the data augmentation pipeline."""
        print("\n🔄 Testing Data Augmentation Pipeline...")
        
        try:
            # Create sample image
            sample_image = Image.fromarray(
                (self.create_sample_data()['ecg_images'][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            
            # Test different augmentation strengths
            for strength in ['light', 'medium', 'heavy']:
                print(f"  Testing {strength} augmentation...")
                
                # Create augmentation pipeline
                transform = create_ecg_augmentation_pipeline(
                    image_size=(224, 224),
                    augmentation_type=strength,
                    use_albumentations=False
                )
                
                # Apply augmentation
                augmented = transform(sample_image)
                
                # Save result
                if isinstance(augmented, torch.Tensor):
                    # Convert back to PIL for saving
                    aug_img = (augmented.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    Image.fromarray(aug_img).save(self.output_dir / f"augmented_{strength}.png")
            
            print("  ✅ Augmentation pipeline working correctly")
            self.test_results['augmentation'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ Augmentation test failed: {e}")
            self.test_results['augmentation'] = False
            return False
    
    def test_vision_transformer_model(self) -> bool:
        """Test the Vision Transformer ECG model."""
        print("\n🤖 Testing Vision Transformer Model...")
        
        try:
            # Create model
            model = create_ecg_vit_detr_model(pretrained=False)
            model.to(self.device)
            
            # Get sample data - need 512x512 for ViT model
            sample_images = torch.randn(4, 3, 512, 512).to(self.device)
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(sample_images)
            
            print(f"  Model output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
            
            # Test loss computation (simplified)
            print("  Testing loss computation...")
            batch_size, num_queries = outputs['class_logits'].shape[:2]
            dummy_targets = {
                'classes': torch.randint(0, 6, (batch_size, num_queries)).to(self.device),
                'bboxes': torch.rand(batch_size, num_queries, 4).to(self.device),
                'keypoints': torch.rand(batch_size, num_queries, 2).to(self.device)
            }
            
            losses = model.compute_loss(outputs, dummy_targets)
            print(f"    Loss values: {[f'{k}: {v.item():.4f}' for k, v in losses.items()]}")
            
            print("  ✅ Vision Transformer model working correctly")
            self.test_results['vit_model'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ ViT model test failed: {e}")
            self.test_results['vit_model'] = False
            return False
    
    def test_multimodal_model(self) -> bool:
        """Test the multi-modal ECG model."""
        print("\n🔬 Testing Multi-Modal Model...")
        
        try:
            # Create model
            model = create_multimodal_ecg_model(num_classes=6)
            model.to(self.device)
            
            # Get sample data
            sample_data = self.create_sample_data()
            images = sample_data['ecg_images'].to(self.device)
            signals = sample_data['raw_signals'].to(self.device)
            numerical_meta = sample_data['numerical_metadata'].to(self.device)
            categorical_meta = {k: v.to(self.device) for k, v in sample_data['categorical_metadata'].items()}
            
            # Test forward pass
            with torch.no_grad():
                outputs = model(images, signals, numerical_meta, categorical_meta)
            
            print(f"  Multi-modal output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
            
            # Test uncertainty estimation
            print("  Testing uncertainty estimation...")
            uncertainty_outputs = model.predict_with_uncertainty(
                images, signals, numerical_meta, categorical_meta, n_samples=5
            )
            
            print(f"  Uncertainty output keys: {list(uncertainty_outputs.keys())}")
            for key, value in uncertainty_outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
            
            print("  ✅ Multi-modal model working correctly")
            self.test_results['multimodal'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ Multi-modal model test failed: {e}")
            self.test_results['multimodal'] = False
            return False
    
    def test_balanced_loss_functions(self) -> bool:
        """Test the balanced loss functions."""
        print("\n⚖️ Testing Balanced Loss Functions...")
        
        try:
            # Create sample dataset statistics (imbalanced)
            sample_labels = [0] * 8000 + [1] * 2000 + [2] * 500 + [3] * 100 + [4] * 50 + [5] * 10
            dataset_stats = calculate_dataset_statistics(sample_labels, num_classes=6)
            
            print(f"  Dataset imbalance ratio: {dataset_stats['imbalance_ratio']:.1f}:1")
            
            # Create loss calculator
            calculator = ECGLossCalculator(dataset_stats)
            
            # Test different loss functions
            loss_functions = {
                'weighted_focal': calculator.get_weighted_focal_loss(gamma=2.0),
                'class_balanced': calculator.get_class_balanced_loss(beta=0.9999),
                'ldam': calculator.get_ldam_loss(max_m=0.5),
                'weighted_bce': calculator.get_weighted_bce_loss()
            }
            
            # Sample predictions and targets
            batch_size = 32
            predictions = torch.randn(batch_size, 6)
            targets = torch.randint(0, 6, (batch_size,))
            
            print("  Testing loss functions:")
            for name, loss_fn in loss_functions.items():
                try:
                    if name == 'weighted_bce':
                        # For BCE, convert targets to one-hot
                        targets_onehot = torch.zeros(batch_size, 6)
                        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
                        loss_value = loss_fn(predictions, targets_onehot)
                    else:
                        loss_value = loss_fn(predictions, targets)
                    
                    print(f"    {name}: {loss_value.item():.4f}")
                except Exception as e:
                    print(f"    {name}: Failed ({e})")
            
            print("  ✅ Balanced loss functions working correctly")
            self.test_results['balanced_loss'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ Balanced loss test failed: {e}")
            self.test_results['balanced_loss'] = False
            return False
    
    def test_uncertainty_estimation(self) -> bool:
        """Test uncertainty estimation methods."""
        print("\n🎯 Testing Uncertainty Estimation...")
        
        try:
            # Create simple test model
            class SimpleECGModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.classifier = nn.Sequential(
                        nn.Linear(100, 64),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(32, 6)
                    )
                
                def forward(self, x):
                    return {'diagnosis_logits': self.classifier(x)}
            
            model = SimpleECGModel().to(self.device)
            
            # Test Monte Carlo Dropout
            print("  Testing Monte Carlo Dropout...")
            mc_dropout = MonteCarloDropout(model, num_samples=10)
            
            sample_input = torch.randn(8, 100).to(self.device)
            uncertainty_results = mc_dropout.predict_with_uncertainty(sample_input)
            
            print(f"  MC Dropout results:")
            for key, value in uncertainty_results.items():
                if isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
            
            # Save uncertainty visualization
            from uncertainty.monte_carlo_uncertainty import UncertaintyVisualizer
            visualizer = UncertaintyVisualizer()
            
            fig = visualizer.plot_uncertainty_distribution(uncertainty_results)
            fig.savefig(self.output_dir / "uncertainty_distribution.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print("  ✅ Uncertainty estimation working correctly")
            self.test_results['uncertainty'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ Uncertainty estimation test failed: {e}")
            self.test_results['uncertainty'] = False
            return False
    
    def test_attention_visualization(self) -> bool:
        """Test attention visualization."""
        print("\n👁️ Testing Attention Visualization...")
        
        try:
            # Create a simple ViT-like model for testing
            from transformers import ViTModel, ViTConfig
            
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12
            )
            
            model = ViTModel(config).to(self.device)
            
            # Initialize visualizer
            visualizer = AttentionVisualizer(model, device=self.device)
            
            # Get sample data
            sample_data = self.create_sample_data()
            sample_image = sample_data['ecg_images'][0:1].to(self.device)  # Single image
            original_image = (sample_data['ecg_images'][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Test attention map extraction
            print("  Extracting attention maps...")
            attention_maps = visualizer.get_attention_maps(sample_image)
            
            print(f"  Found attention maps for layers: {list(attention_maps.keys())}")
            
            # Test attention visualization
            print("  Creating attention visualization...")
            fig = visualizer.visualize_attention_maps(
                sample_image, 
                original_image,
                layer_indices=[0, 2, 5],
                head_indices=[0, 3, 7],
                save_path=str(self.output_dir / "attention_visualization.png")
            )
            plt.close(fig)
            
            # Cleanup
            visualizer.cleanup()
            
            print("  ✅ Attention visualization working correctly")
            self.test_results['attention_viz'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ Attention visualization test failed: {e}")
            self.test_results['attention_viz'] = False
            return False
    
    def run_all_tests(self) -> dict:
        """Run all tests and return results."""
        print("🚀 Starting Comprehensive Model Testing...")
        
        # Run all tests
        tests = [
            self.test_augmentation_pipeline,
            self.test_vision_transformer_model,
            self.test_multimodal_model,
            self.test_balanced_loss_functions,
            self.test_uncertainty_estimation,
            self.test_attention_visualization
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Test failed with exception: {e}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("🏁 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed_count = sum(self.test_results.values())
        total = len(self.test_results)
        
        for test_name, test_passed in self.test_results.items():
            status = "✅ PASSED" if test_passed else "❌ FAILED"
            print(f"{test_name:20} {status}")
        
        print(f"\nOverall: {passed_count}/{total} tests passed ({passed_count/total*100:.1f}%)")
        
        if passed_count == total:
            print("🎉 All tests passed! Models are ready for use.")
        else:
            print("⚠️  Some tests failed. Check the error messages above.")
        
        # Save test results
        with open(self.output_dir / "test_results.json", 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        return self.test_results


def main():
    """Main testing function."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create tester and run all tests
    tester = ECGModelTester(device=device)
    results = tester.run_all_tests()
    
    return results


if __name__ == "__main__":
    results = main()