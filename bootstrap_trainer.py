#!/usr/bin/env python3
"""
Bootstrap R-Peak Trainer
Quick training pipeline to bootstrap ECG detection with synthetic R-peak data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from models.backbones.vision_transformer_ecg import create_ecg_vit_detr_model
from enhanced_preprocessing import ECGImagePreprocessor


class ECGRPeakBootstrapper:
    """Bootstrap ECG R-peak detection with synthetic training data."""
    
    def __init__(self, device='mps'):
        self.device = device
        self.preprocessor = ECGImagePreprocessor()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("experiments") / f"bootstrap_training_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_classes = 6  # P, Q, R, S, T, Background
        self.r_peak_class = 2  # R-peak class index
        
        print(f"🎯 ECG R-Peak Bootstrapper Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {device}")
        print("=" * 60)
    
    def generate_synthetic_ecg_with_rpeaks(self, num_samples=500, image_size=(512, 512)):
        """Generate synthetic ECG images with known R-peak locations."""
        print(f"\n🏭 Generating {num_samples} synthetic ECG samples...")
        
        synthetic_data = []
        
        for i in tqdm(range(num_samples), desc="Generating ECG data"):
            # Create synthetic ECG signal
            sample_data = self.create_synthetic_ecg_sample(i, image_size)
            synthetic_data.append(sample_data)
        
        print(f"✅ Generated {len(synthetic_data)} synthetic ECG samples")
        return synthetic_data
    
    def create_synthetic_ecg_sample(self, sample_id, image_size):
        """Create a single synthetic ECG sample with R-peaks."""
        width, height = image_size
        
        # Create white background image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Generate synthetic ECG parameters
        num_beats = random.randint(3, 8)  # 3-8 heartbeats per image
        heart_rate = random.uniform(60, 120)  # 60-120 BPM
        baseline_y = height // 2
        
        # Generate R-peak locations
        r_peaks = []
        beat_spacing = width / (num_beats + 1)
        
        for beat_idx in range(num_beats):
            # R-peak location with some random variation
            r_x = int((beat_idx + 1) * beat_spacing + random.uniform(-20, 20))
            r_y = int(baseline_y + random.uniform(-30, 30))
            
            # Ensure R-peak is within image bounds
            r_x = max(20, min(width - 20, r_x))
            r_y = max(50, min(height - 50, r_y))
            
            r_peaks.append((r_x, r_y))
            
            # Draw ECG waveform around this R-peak
            self.draw_ecg_beat(draw, r_x, r_y, baseline_y, width, height)
        
        # Draw baseline noise
        self.add_baseline_noise(draw, baseline_y, width, height)
        
        # Convert R-peak coordinates to normalized format
        normalized_rpeaks = []
        for r_x, r_y in r_peaks:
            norm_x = r_x / width
            norm_y = r_y / height
            normalized_rpeaks.append([norm_x, norm_y])
        
        # Create training target
        target = {
            'keypoints': normalized_rpeaks,
            'classes': [self.r_peak_class] * len(normalized_rpeaks),
            'confidences': [1.0] * len(normalized_rpeaks)
        }
        
        return {
            'sample_id': sample_id,
            'image': image,
            'target': target,
            'num_rpeaks': len(r_peaks)
        }
    
    def draw_ecg_beat(self, draw, r_x, r_y, baseline_y, width, height):
        """Draw a realistic ECG beat centered at R-peak location."""
        # ECG beat parameters
        beat_width = 80  # Width of one beat
        amplitude_scale = random.uniform(0.7, 1.3)
        
        # Define ECG waveform points relative to R-peak
        # P-Q-R-S-T wave pattern
        waveform_points = [
            (-40, baseline_y + random.uniform(-5, 5)),      # Start
            (-35, baseline_y + 8 * amplitude_scale),        # P-wave peak
            (-30, baseline_y),                              # P-wave end
            (-25, baseline_y),                              # PR segment
            (-15, baseline_y - 5 * amplitude_scale),        # Q-wave
            (0, r_y),                                       # R-peak (main peak)
            (8, baseline_y - 15 * amplitude_scale),         # S-wave
            (15, baseline_y),                               # ST segment
            (25, baseline_y + 12 * amplitude_scale),        # T-wave peak
            (35, baseline_y),                               # T-wave end
            (40, baseline_y),                               # End
        ]
        
        # Convert relative points to absolute coordinates
        absolute_points = []
        for rel_x, y in waveform_points:
            abs_x = r_x + rel_x
            abs_y = int(y)
            
            # Keep within bounds
            abs_x = max(0, min(width - 1, abs_x))
            abs_y = max(0, min(height - 1, abs_y))
            
            absolute_points.append((abs_x, abs_y))
        
        # Draw the ECG waveform
        for i in range(len(absolute_points) - 1):
            draw.line([absolute_points[i], absolute_points[i + 1]], 
                     fill='black', width=random.randint(2, 4))
        
        # Emphasize R-peak
        draw.ellipse([r_x - 2, r_y - 2, r_x + 2, r_y + 2], fill='red', outline='red')
    
    def add_baseline_noise(self, draw, baseline_y, width, height):
        """Add realistic ECG baseline and noise."""
        # Draw baseline with slight variations
        prev_x, prev_y = 0, baseline_y
        
        for x in range(5, width, 5):
            noise_y = baseline_y + random.uniform(-2, 2)
            noise_y = max(0, min(height - 1, noise_y))
            
            draw.line([(prev_x, prev_y), (x, noise_y)], fill='gray', width=1)
            prev_x, prev_y = x, noise_y
        
        # Add some grid lines (faint)
        grid_spacing = 20
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill='lightgray', width=1)
        
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill='lightgray', width=1)
    
    def create_training_dataset(self, synthetic_data):
        """Convert synthetic data to training format."""
        print(f"\n📦 Creating training dataset...")
        
        training_samples = []
        
        for sample in tqdm(synthetic_data, desc="Processing samples"):
            # Convert synthetic image directly to tensor
            image_tensor = torch.from_numpy(np.array(sample['image'])).permute(2, 0, 1).float() / 255.0
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Create DETR-style targets
            target = self.create_detr_target(sample['target'])
            
            training_samples.append({
                'image': image_tensor,
                'target': target,
                'sample_id': sample['sample_id']
            })
        
        print(f"✅ Created {len(training_samples)} training samples")
        return training_samples
    
    def create_detr_target(self, sample_target):
        """Create DETR-style training target."""
        keypoints = torch.tensor(sample_target['keypoints'], dtype=torch.float32)
        classes = torch.tensor(sample_target['classes'], dtype=torch.long)
        
        # Pad to fixed number of queries (60 for DETR)
        max_queries = 60
        num_actual = len(keypoints)
        
        # Create padded tensors
        padded_keypoints = torch.zeros((max_queries, 2), dtype=torch.float32)
        padded_classes = torch.full((max_queries,), self.num_classes - 1, dtype=torch.long)  # Background class
        padded_confidences = torch.zeros(max_queries, dtype=torch.float32)
        
        # Fill in actual data
        if num_actual > 0:
            padded_keypoints[:num_actual] = keypoints
            padded_classes[:num_actual] = classes
            padded_confidences[:num_actual] = 1.0
        
        return {
            'keypoints': padded_keypoints,
            'classes': padded_classes,
            'confidences': padded_confidences,
            'num_objects': num_actual
        }
    
    def create_training_dataloader(self, training_samples, batch_size=8):
        """Create PyTorch DataLoader for training."""
        from torch.utils.data import Dataset, DataLoader
        
        class ECGDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                return self.samples[idx]
        
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            targets = []
            for item in batch:
                targets.append(item['target'])
            
            return {'images': images, 'targets': targets}
        
        dataset = ECGDataset(training_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        return dataloader
    
    def create_simple_detection_model(self):
        """Create a simple CNN model for R-peak detection."""
        class SimpleECGDetector(nn.Module):
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
        
        return SimpleECGDetector(self.num_classes, 60)
    
    def train_model(self, training_dataloader, num_epochs=50, learning_rate=1e-4):
        """Train the ECG detection model."""
        print(f"\n🎯 Training ECG R-Peak Detection Model...")
        print(f"📊 Epochs: {num_epochs}")
        print(f"📈 Learning Rate: {learning_rate}")
        
        # Create simple CNN model for bootstrapping
        model = self.create_simple_detection_model()
        model.to(self.device)
        model.train()
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        training_history = {
            'losses': [],
            'epoch_losses': [],
            'learning_rates': []
        }
        
        for epoch in range(num_epochs):
            epoch_losses = []
            progress_bar = tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['images'].to(self.device)
                targets = batch['targets']
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                
                # Compute loss
                loss = self.compute_training_loss(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track loss
                loss_value = loss.item()
                epoch_losses.append(loss_value)
                training_history['losses'].append(loss_value)
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss_value:.4f}'})
            
            # End of epoch
            avg_epoch_loss = np.mean(epoch_losses)
            training_history['epoch_losses'].append(avg_epoch_loss)
            training_history['learning_rates'].append(scheduler.get_last_lr()[0])
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model_checkpoint(model, optimizer, epoch, avg_epoch_loss)
        
        # Save final model
        final_model_path = self.output_dir / 'final_trained_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': training_history,
            'training_complete': True
        }, final_model_path)
        
        print(f"✅ Training complete! Model saved to {final_model_path}")
        
        # Plot training history
        self.plot_training_history(training_history)
        
        return model, training_history
    
    def compute_training_loss(self, outputs, targets):
        """Compute training loss for ECG detection."""
        batch_size = len(targets)
        
        # Get model outputs
        class_logits = outputs['class_logits']  # [batch_size, num_queries, num_classes]
        keypoint_coords = outputs['keypoint_coords']  # [batch_size, num_queries, 2]
        confidence_scores = outputs['confidence_scores']  # [batch_size, num_queries, 1]
        
        total_loss = 0.0
        
        for b in range(batch_size):
            target = targets[b]
            target_classes = target['classes'].to(self.device)  # [num_queries]
            target_keypoints = target['keypoints'].to(self.device)  # [num_queries, 2]
            target_confidences = target['confidences'].to(self.device)  # [num_queries]
            
            # Classification loss
            class_loss = nn.CrossEntropyLoss()(class_logits[b], target_classes)
            
            # Keypoint regression loss (only for positive samples)
            positive_mask = target_classes != (self.num_classes - 1)  # Not background
            if positive_mask.sum() > 0:
                pred_keypoints = keypoint_coords[b][positive_mask]
                gt_keypoints = target_keypoints[positive_mask]
                keypoint_loss = nn.MSELoss()(pred_keypoints, gt_keypoints)
            else:
                keypoint_loss = torch.tensor(0.0, device=self.device)
            
            # Confidence loss
            pred_confidences = confidence_scores[b].squeeze(-1)
            confidence_loss = nn.MSELoss()(pred_confidences, target_confidences)
            
            # Combined loss
            sample_loss = class_loss + 5.0 * keypoint_loss + confidence_loss
            total_loss += sample_loss
        
        return total_loss / batch_size
    
    def save_model_checkpoint(self, model, optimizer, epoch, loss):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
    
    def plot_training_history(self, history):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ECG R-Peak Detection Training History', fontsize=16, fontweight='bold')
        
        # Loss over iterations
        axes[0, 0].plot(history['losses'], alpha=0.7)
        axes[0, 0].set_title('Training Loss (per iteration)')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Epoch losses
        axes[0, 1].plot(history['epoch_losses'], 'o-', linewidth=2, markersize=6)
        axes[0, 1].set_title('Average Loss per Epoch')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[1, 0].plot(history['learning_rates'], 'o-', linewidth=2, markersize=6, color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss smoothed
        if len(history['losses']) > 100:
            # Moving average
            window = len(history['losses']) // 20
            smoothed = np.convolve(history['losses'], np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smoothed, linewidth=2, color='red')
            axes[1, 1].set_title(f'Smoothed Training Loss (window={window})')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].plot(history['losses'], linewidth=2, color='red')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history saved: training_history.png")
    
    def test_trained_model(self, model, real_ecg_images, num_samples=5):
        """Test the trained model on real ECG images."""
        print(f"\n🧪 Testing Trained Model on Real ECG Images...")
        
        model.eval()
        test_results = []
        
        for i, image_path in enumerate(real_ecg_images[:num_samples]):
            print(f"  Testing image {i+1}: {Path(image_path).name}")
            
            # Load and preprocess
            processed_tensor, original_pil = self.preprocessor.load_and_preprocess_ecg(
                image_path, target_size=(512, 512), preprocessing_level='aggressive'
            )
            
            if processed_tensor is None:
                continue
            
            # Forward pass
            images = processed_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = model(images)
            
            # Analyze results
            confidence_scores = outputs['confidence_scores'].squeeze(-1)
            keypoint_coords = outputs['keypoint_coords']
            class_logits = outputs['class_logits']
            
            # Test different thresholds
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            threshold_results = {}
            
            for threshold in thresholds:
                if len(confidence_scores.shape) == 2:
                    high_conf_mask = confidence_scores.squeeze(0) > threshold
                else:
                    high_conf_mask = confidence_scores > threshold
                num_detections = high_conf_mask.sum().item()
                threshold_results[threshold] = num_detections
            
            result = {
                'image_name': Path(image_path).name,
                'max_confidence': float(confidence_scores.max().item()),
                'mean_confidence': float(confidence_scores.mean().item()),
                'threshold_detections': threshold_results
            }
            
            test_results.append(result)
            
            print(f"    Max confidence: {result['max_confidence']:.3f}")
            print(f"    Detections at 0.5: {threshold_results[0.5]}")
            print(f"    Detections at 0.6: {threshold_results[0.6]}")
        
        # Save test results
        with open(self.output_dir / 'trained_model_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return test_results
    
    def generate_bootstrap_report(self, training_history, test_results):
        """Generate comprehensive bootstrap training report."""
        print(f"\n📋 Generating Bootstrap Training Report...")
        
        # Calculate improvements
        final_loss = training_history['epoch_losses'][-1]
        initial_loss = training_history['epoch_losses'][0]
        loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        # Test performance analysis
        avg_max_confidence = np.mean([r['max_confidence'] for r in test_results])
        avg_detections_05 = np.mean([r['threshold_detections'][0.5] for r in test_results])
        avg_detections_06 = np.mean([r['threshold_detections'][0.6] for r in test_results])
        
        report = {
            'bootstrap_summary': {
                'training_date': datetime.now().isoformat(),
                'training_epochs': len(training_history['epoch_losses']),
                'final_training_loss': final_loss,
                'loss_improvement_percent': loss_improvement
            },
            'model_performance': {
                'average_max_confidence': avg_max_confidence,
                'average_detections_at_05': avg_detections_05,
                'average_detections_at_06': avg_detections_06,
                'images_tested': len(test_results)
            },
            'comparison_to_random': {
                'random_model_confidence': 0.395,  # From Phase 3
                'trained_model_confidence': avg_max_confidence,
                'confidence_improvement': ((avg_max_confidence - 0.395) / 0.395) * 100 if avg_max_confidence > 0.395 else 0
            },
            'next_steps': [
                "Model shows trained weights vs random initialization",
                "Further fine-tuning on real ECG data recommended",
                "Consider data augmentation for better generalization",
                "Evaluate on larger test set for robust metrics"
            ]
        }
        
        # Save report
        with open(self.output_dir / 'bootstrap_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print(f"\n🎯 BOOTSTRAP TRAINING REPORT")
        print("=" * 60)
        print(f"📚 Training Epochs: {report['bootstrap_summary']['training_epochs']}")
        print(f"📉 Loss Improvement: {loss_improvement:.1f}%")
        print(f"🎯 Final Training Loss: {final_loss:.4f}")
        
        print(f"\n🧪 MODEL PERFORMANCE ON REAL DATA:")
        print(f"   Average Max Confidence: {avg_max_confidence:.3f}")
        print(f"   Detections at 0.5: {avg_detections_05:.1f}")
        print(f"   Detections at 0.6: {avg_detections_06:.1f}")
        
        if report['comparison_to_random']['confidence_improvement'] > 0:
            print(f"\n🚀 IMPROVEMENT OVER RANDOM MODEL:")
            print(f"   Confidence Boost: +{report['comparison_to_random']['confidence_improvement']:.1f}%")
        else:
            print(f"\n⚠️  Model needs more training or better data")
        
        print(f"\n✅ Full report saved to {self.output_dir}/bootstrap_report.json")
        
        return report


def main():
    """Run complete bootstrap training pipeline."""
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Initialize bootstrapper
    bootstrapper = ECGRPeakBootstrapper(device=device)
    
    print(f"🚀 PHASE 4: BOOTSTRAP R-PEAK TRAINING")
    
    # 1. Generate synthetic data
    synthetic_data = bootstrapper.generate_synthetic_ecg_with_rpeaks(num_samples=200)
    
    # Save sample synthetic images
    sample_dir = bootstrapper.output_dir / 'synthetic_samples'
    sample_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(synthetic_data[:5]):
        sample['image'].save(sample_dir / f'synthetic_ecg_{i+1}.png')
    
    print(f"💾 Saved sample synthetic images to {sample_dir}")
    
    # 2. Create training dataset
    training_samples = bootstrapper.create_training_dataset(synthetic_data)
    training_dataloader = bootstrapper.create_training_dataloader(training_samples, batch_size=4)
    
    # 3. Train model
    trained_model, training_history = bootstrapper.train_model(
        training_dataloader, num_epochs=10, learning_rate=5e-5
    )
    
    # 4. Test on real ECG images
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
    
    if image_paths:
        test_results = bootstrapper.test_trained_model(trained_model, image_paths, num_samples=5)
    else:
        print("⚠️  No real ECG images found for testing")
        test_results = []
    
    # 5. Generate comprehensive report
    final_report = bootstrapper.generate_bootstrap_report(training_history, test_results)
    
    print(f"\n🎉 Phase 4 bootstrap training complete!")
    print(f"📁 All results saved to: {bootstrapper.output_dir}")
    print(f"\n💡 Model is now trained and can be loaded for consistent performance!")


if __name__ == "__main__":
    main()