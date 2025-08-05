#!/usr/bin/env python3
"""
Quick Test Script - Test Models Immediately
Simple script to quickly test if all components are working.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def main():
    print("🚀 Quick ECG Model Test")
    print("=" * 40)
    
    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Test 1: Data Augmentation
    print("\n1️⃣ Testing Data Augmentation...")
    try:
        from augmentation.enhanced_transforms import create_ecg_augmentation_pipeline
        
        transform = create_ecg_augmentation_pipeline(
            image_size=(224, 224),
            augmentation_type='medium'
        )
        
        print("   ✅ Data augmentation loaded successfully!")
        
    except Exception as e:
        print(f"   ❌ Data augmentation failed: {e}")
    
    # Test 2: Multi-Modal Model
    print("\n2️⃣ Testing Multi-Modal Model...")
    try:
        from models.multimodal_ecg_model import create_multimodal_ecg_model
        
        # Create model
        model = create_multimodal_ecg_model(num_classes=6)
        model.to(device)
        
        # Create dummy data
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        signals = torch.randn(batch_size, 12, 1000).to(device)
        numerical_meta = torch.randn(batch_size, 5).to(device)
        categorical_meta = {
            'sex': torch.randint(0, 2, (batch_size,)).to(device),
            'device_type': torch.randint(0, 5, (batch_size,)).to(device),
            'recording_condition': torch.randint(0, 3, (batch_size,)).to(device)
        }
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(images, signals, numerical_meta, categorical_meta)
        
        print(f"   ✅ Multi-modal model working! Output keys: {list(outputs.keys())}")
        
    except Exception as e:
        print(f"   ❌ Multi-modal model failed: {e}")
    
    # Test 3: Vision Transformer
    print("\n3️⃣ Testing Vision Transformer...")
    try:
        from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
        
        # Create model (no pretrained to avoid downloads)
        vit_model = create_ecg_vit_detr_model(pretrained=False)
        vit_model.to(device)
        
        # Test with dummy data
        dummy_images = torch.randn(2, 3, 512, 512).to(device)
        
        with torch.no_grad():
            vit_outputs = vit_model(dummy_images)
        
        print(f"   ✅ Vision Transformer working! Output keys: {list(vit_outputs.keys())}")
        
    except Exception as e:
        print(f"   ❌ Vision Transformer failed: {e}")
    
    # Test 4: Loss Functions
    print("\n4️⃣ Testing Balanced Loss Functions...")
    try:
        from training.balanced_loss import ECGLossCalculator, calculate_dataset_statistics
        
        # Create imbalanced dataset stats
        sample_labels = [0] * 1000 + [1] * 200 + [2] * 50 + [3] * 10
        stats = calculate_dataset_statistics(sample_labels, num_classes=4)
        
        calculator = ECGLossCalculator(stats)
        focal_loss = calculator.get_weighted_focal_loss(gamma=2.0)
        
        # Test loss
        predictions = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss_value = focal_loss(predictions, targets)
        
        print(f"   ✅ Loss functions working! Focal loss: {loss_value.item():.4f}")
        
    except Exception as e:
        print(f"   ❌ Loss functions failed: {e}")
    
    # Test 5: Uncertainty Estimation
    print("\n5️⃣ Testing Uncertainty Estimation...")
    try:
        from uncertainty.monte_carlo_uncertainty import MonteCarloDropout
        
        # Simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(32, 6)
                )
            
            def forward(self, x):
                return {'diagnosis_logits': self.classifier(x)}
        
        test_model = TestModel().to(device)
        mc_dropout = MonteCarloDropout(test_model, num_samples=5)
        
        test_input = torch.randn(4, 10).to(device)
        uncertainty_results = mc_dropout.predict_with_uncertainty(test_input)
        
        print(f"   ✅ Uncertainty estimation working! Keys: {list(uncertainty_results.keys())}")
        
    except Exception as e:
        print(f"   ❌ Uncertainty estimation failed: {e}")
    
    # Test 6: Attention Visualization
    print("\n6️⃣ Testing Attention Visualization...")
    try:
        from visualization.attention_maps import AttentionVisualizer
        from transformers import ViTModel, ViTConfig
        
        # Create simple ViT model
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8
        )
        
        vit_model = ViTModel(config).to(device)
        visualizer = AttentionVisualizer(vit_model, device=device)
        
        # Test attention extraction
        test_image = torch.randn(1, 3, 224, 224).to(device)
        attention_maps = visualizer.get_attention_maps(test_image)
        
        visualizer.cleanup()
        
        print(f"   ✅ Attention visualization working! Found {len(attention_maps)} layers")
        
    except Exception as e:
        print(f"   ❌ Attention visualization failed: {e}")
    
    print("\n" + "=" * 40)
    print("🏁 Quick Test Complete!")
    print("\nIf you see ✅ for most tests, you're ready to go!")
    print("\nNext steps:")
    print("• Run full test suite: python test_models.py")
    print("• Train a model: python scripts/train_model.py")
    print("• Process data: python scripts/prepare_dataset.py")
    
    return True

if __name__ == "__main__":
    main()