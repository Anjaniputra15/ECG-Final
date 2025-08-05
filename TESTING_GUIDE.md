# 🧪 ECG Model Testing Guide

This guide shows you exactly how to test all the models we've developed.

## 🚀 Quick Start (30 seconds)

**Run this first to check if everything works:**

```bash
cd /Users/baba/Downloads/ECG-LLM-main/venv/etc/jupyter/nbconfig/notebook.d/ECGfinal/ecg-pqrst-detector

python quick_test.py
```

This will test all major components quickly and tell you what's working.

## 🔬 Comprehensive Testing

### 1. Full Test Suite
```bash
python test_models.py
```

This runs comprehensive tests for:
- ✅ **Data Augmentation Pipeline** (enhanced_transforms.py)
- ✅ **Vision Transformer Model** (vision_transformer_ecg.py)  
- ✅ **Multi-Modal ECG Model** (multimodal_ecg_model.py)
- ✅ **Balanced Loss Functions** (balanced_loss.py)
- ✅ **Uncertainty Estimation** (monte_carlo_uncertainty.py)
- ✅ **Attention Visualization** (attention_maps.py)

### 2. Individual Component Tests

#### Test Data Augmentation
```python
from augmentation.enhanced_transforms import create_ecg_augmentation_pipeline
from PIL import Image
import torch

# Load your ECG image
image = Image.open("path/to/your/ecg.png")

# Create augmentation pipeline
transform = create_ecg_augmentation_pipeline(
    image_size=(224, 224), 
    augmentation_type='medium'
)

# Apply augmentation
augmented = transform(image)
print(f"Augmented tensor shape: {augmented.shape}")
```

#### Test Multi-Modal Model
```python
from models.multimodal_ecg_model import create_multimodal_ecg_model
import torch

# Create model
model = create_multimodal_ecg_model(num_classes=6)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

# Create sample data
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224).to(device)
signals = torch.randn(batch_size, 12, 1000).to(device)  # 12 leads, 1000 samples
numerical_metadata = torch.randn(batch_size, 5).to(device)
categorical_metadata = {
    'sex': torch.randint(0, 2, (batch_size,)).to(device),
    'device_type': torch.randint(0, 5, (batch_size,)).to(device),
    'recording_condition': torch.randint(0, 3, (batch_size,)).to(device)
}

# Forward pass
with torch.no_grad():
    outputs = model(images, signals, numerical_metadata, categorical_metadata)

print("Model outputs:")
for key, value in outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")

# Test uncertainty estimation
uncertainty_outputs = model.predict_with_uncertainty(
    images, signals, numerical_metadata, categorical_metadata, n_samples=10
)

print("\nUncertainty outputs:")
for key, value in uncertainty_outputs.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
```

#### Test Vision Transformer
```python
from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
import torch

# Create model
model = create_ecg_vit_detr_model(pretrained=False)  # Set to True for pretrained weights
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

# Test with ECG images
images = torch.randn(2, 3, 512, 512).to(device)  # Batch of 2 ECG images

with torch.no_grad():
    predictions = model(images)

print("ViT-DETR outputs:")
for key, value in predictions.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
```

#### Test Loss Functions
```python
from training.balanced_loss import ECGLossCalculator, calculate_dataset_statistics
import torch

# Simulate imbalanced dataset
sample_labels = [0] * 8000 + [1] * 2000 + [2] * 500 + [3] * 100 + [4] * 50 + [5] * 10
dataset_stats = calculate_dataset_statistics(sample_labels, num_classes=6)

print(f"Dataset imbalance ratio: {dataset_stats['imbalance_ratio']:.1f}:1")

# Create loss calculator
calculator = ECGLossCalculator(dataset_stats)

# Get different loss functions
focal_loss = calculator.get_weighted_focal_loss(gamma=2.0)
class_balanced_loss = calculator.get_class_balanced_loss(beta=0.9999)
ldam_loss = calculator.get_ldam_loss(max_m=0.5)

# Test losses
predictions = torch.randn(32, 6)
targets = torch.randint(0, 6, (32,))

focal_value = focal_loss(predictions, targets)
cb_value = class_balanced_loss(predictions, targets)
ldam_value = ldam_loss(predictions, targets)

print(f"Focal Loss: {focal_value.item():.4f}")
print(f"Class-Balanced Loss: {cb_value.item():.4f}")
print(f"LDAM Loss: {ldam_value.item():.4f}")
```

#### Test Uncertainty Estimation
```python
from uncertainty.monte_carlo_uncertainty import MonteCarloDropout, UncertaintyVisualizer
import torch
import torch.nn as nn

# Create a model with dropout
class ECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(128, 6)
        )
    
    def forward(self, x):
        return {'diagnosis_logits': self.classifier(x)}

model = ECGModel()
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

# Initialize Monte Carlo Dropout
mc_dropout = MonteCarloDropout(model, num_samples=50)

# Test with sample data
sample_input = torch.randn(16, 768).to(device)
uncertainty_results = mc_dropout.predict_with_uncertainty(sample_input)

print("Uncertainty estimation results:")
for key, value in uncertainty_results.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
        if value.dim() == 1:  # Print some statistics for 1D tensors
            print(f"    Mean: {value.mean().item():.4f}, Std: {value.std().item():.4f}")

# Visualize uncertainty
visualizer = UncertaintyVisualizer()
fig = visualizer.plot_uncertainty_distribution(uncertainty_results)
fig.savefig("uncertainty_analysis.png", dpi=300, bbox_inches='tight')
print("Uncertainty visualization saved as 'uncertainty_analysis.png'")
```

#### Test Attention Visualization
```python
from visualization.attention_maps import AttentionVisualizer
from transformers import ViTModel, ViTConfig
import torch
import numpy as np

# Create ViT model
config = ViTConfig(
    image_size=224,
    patch_size=16,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12
)

model = ViTModel(config)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model.to(device)

# Initialize visualizer
visualizer = AttentionVisualizer(model, device=device)

# Create sample ECG image (you can replace with real ECG)
sample_image = torch.randn(1, 3, 224, 224).to(device)
original_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Extract attention maps
attention_maps = visualizer.get_attention_maps(sample_image)
print(f"Extracted attention from {len(attention_maps)} layers")

# Create visualization
fig = visualizer.visualize_attention_maps(
    sample_image,
    original_image,
    layer_indices=[0, 5, 11],
    head_indices=[0, 5, 11],
    save_path="attention_maps.png"
)

print("Attention visualization saved as 'attention_maps.png'")

# Generate comprehensive report
report_paths = visualizer.generate_comprehensive_report(
    sample_image,
    original_image,
    "test_ecg_001",
    "./attention_reports"
)

print("Comprehensive attention report generated:")
for key, path in report_paths.items():
    print(f"  {key}: {path}")

# Cleanup
visualizer.cleanup()
```

## 📊 Expected Outputs

### Quick Test Output
```
🚀 Quick ECG Model Test
========================================
Device: mps

1️⃣ Testing Data Augmentation...
   ✅ Data augmentation loaded successfully!

2️⃣ Testing Multi-Modal Model...
   ✅ Multi-modal model working! Output keys: ['diagnosis_logits', 'severity_logits', 'confidence_scores', ...]

3️⃣ Testing Vision Transformer...
   ✅ Vision Transformer working! Output keys: ['class_logits', 'bbox_coords', 'keypoint_coords', ...]

4️⃣ Testing Balanced Loss Functions...
   ✅ Loss functions working! Focal loss: 1.8243

5️⃣ Testing Uncertainty Estimation...
   ✅ Uncertainty estimation working! Keys: ['mean_predictions', 'prediction_variance', ...]

6️⃣ Testing Attention Visualization...
   ✅ Attention visualization working! Found 4 layers

========================================
🏁 Quick Test Complete!
```

### Full Test Output
```
🧪 ECG Model Testing Suite
Device: mps
Output directory: test_outputs
==================================================

🚀 Starting Comprehensive Model Testing...

🔄 Testing Data Augmentation Pipeline...
  Testing light augmentation...
  Testing medium augmentation...
  Testing heavy augmentation...
  ✅ Augmentation pipeline working correctly

🤖 Testing Vision Transformer Model...
  Model output keys: ['class_logits', 'bbox_coords', 'keypoint_coords', 'confidence_scores']
    class_logits: torch.Size([4, 60, 6])
    bbox_coords: torch.Size([4, 60, 4])
    keypoint_coords: torch.Size([4, 60, 2])
    confidence_scores: torch.Size([4, 60, 1])
  Testing loss computation...
    Loss values: ['total_loss: 2.1234', 'class_loss: 1.8901', ...]
  ✅ Vision Transformer model working correctly

... [similar output for all tests] ...

==================================================
🏁 TEST RESULTS SUMMARY
==================================================
augmentation         ✅ PASSED
vit_model           ✅ PASSED
multimodal          ✅ PASSED
balanced_loss       ✅ PASSED
uncertainty         ✅ PASSED
attention_viz       ✅ PASSED

Overall: 6/6 tests passed (100.0%)
🎉 All tests passed! Models are ready for use.
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install torch torchvision transformers
   pip install albumentations opencv-python
   pip install matplotlib seaborn
   pip install captum  # For attention visualization
   ```

2. **Memory Issues**
   - Reduce batch size in tests
   - Use CPU instead of GPU/MPS for testing
   - Close other applications

3. **Model Loading Issues**
   - Check internet connection (for pretrained models)
   - Use `pretrained=False` for offline testing
   - Verify file paths are correct

### Device Issues
```python
# Force CPU usage if needed
import torch
torch.cuda.is_available = lambda: False
torch.backends.mps.is_available = lambda: False
```

## 🎯 What Each Test Validates

| Test | What it Checks |
|------|----------------|
| **Augmentation** | Transform pipelines, ECG-specific augmentations, medical integrity |
| **Vision Transformer** | ViT backbone, DETR decoder, loss computation, keypoint detection |
| **Multi-Modal** | Image + signal + metadata fusion, attention mechanisms, uncertainty |
| **Balanced Loss** | Class weights, focal loss, LDAM, imbalanced dataset handling |
| **Uncertainty** | Monte Carlo dropout, epistemic/aleatoric uncertainty, calibration |
| **Attention Viz** | Attention extraction, visualization, interpretability reports |

## 📁 Output Files

After running tests, you'll find:

```
test_outputs/
├── augmented_light.png        # Light augmentation example
├── augmented_medium.png       # Medium augmentation example  
├── augmented_heavy.png        # Heavy augmentation example
├── uncertainty_distribution.png # Uncertainty analysis plot
├── attention_visualization.png  # Attention maps
├── test_results.json          # Detailed test results
└── attention_reports/          # Comprehensive attention analysis
    ├── test_ecg_001_attention_maps.png
    ├── test_ecg_001_gradcam.png
    ├── test_ecg_001_integrated_gradients.png
    └── test_ecg_001_report.json
```

## 🚀 Next Steps

Once tests pass:

1. **Train Models**: Use the training scripts in `scripts/`
2. **Process Real Data**: Use `scripts/prepare_dataset.py` 
3. **Deploy Models**: Use the deployment configs in `deployment/`
4. **Analyze Results**: Use evaluation tools in `evaluation/`

## 💡 Pro Tips

- Run `quick_test.py` first to catch major issues quickly
- Use `test_models.py` for comprehensive validation
- Check `test_outputs/` directory for visual results
- Look at generated images to verify augmentations work correctly
- Monitor memory usage during testing
- Test with your own ECG data after validation passes

Happy testing! 🎉