#!/usr/bin/env python3
"""
HuBERT-ECG: Transfer Learning from Speech Processing to ECG Analysis
Innovative approach using HuBERT (speech model) for ECG PQRST detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, HubertConfig
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import cv2

class ECGToAudioConverter:
    """
    Convert ECG images/signals to audio-like representations for HuBERT processing
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 10.0,
                 freq_range: Tuple[int, int] = (80, 8000)):
        """
        Initialize ECG to Audio converter
        
        Args:
            sample_rate: Target sample rate for audio
            duration: Duration of generated audio in seconds
            freq_range: Frequency range for ECG-to-audio mapping
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        self.freq_range = freq_range
    
    def ecg_image_to_signal(self, image: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extract 1D signal from ECG image
        
        Args:
            image: ECG image tensor [C, H, W] or [H, W]
            mask: Optional mask tensor
            
        Returns:
            1D ECG signal
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            if image.shape[0] == 3:
                image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                image = image[0]
        
        height, width = image.shape
        signal = []
        
        # Extract signal column by column
        for x in range(width):
            column = image[:, x]
            
            if mask is not None:
                # Use mask to focus on ECG traces
                mask_column = mask[:, x]
                if mask_column.sum() > 0:
                    masked_pixels = column[mask_column > 0.5]
                    if len(masked_pixels) > 0:
                        # Find ECG trace (darkest pixel)
                        trace_value = masked_pixels.min()
                        y_coord = height - (trace_value * height)
                        signal.append(y_coord.item())
                    else:
                        signal.append(height / 2)
                else:
                    signal.append(height / 2)
            else:
                # Simple center-of-mass approach
                weights = 1.0 - column  # Invert so dark pixels have high weight
                indices = torch.arange(height, dtype=torch.float32)
                center_of_mass = torch.sum(weights * indices) / torch.sum(weights)
                signal.append(center_of_mass.item())
        
        return torch.tensor(signal, dtype=torch.float32)
    
    def signal_to_audio(self, ecg_signal: torch.Tensor) -> torch.Tensor:
        """
        Convert ECG signal to audio-like representation
        
        Args:
            ecg_signal: 1D ECG signal
            
        Returns:
            Audio-like waveform compatible with HuBERT
        """
        # Normalize ECG signal
        ecg_signal = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-8)
        
        # Resample to target length
        if len(ecg_signal) != self.target_length:
            ecg_signal = F.interpolate(
                ecg_signal.unsqueeze(0).unsqueeze(0),
                size=self.target_length,
                mode='linear',
                align_corners=False
            ).squeeze()
        
        # Method 1: Direct mapping (treat ECG as audio)
        audio_direct = ecg_signal * 0.5  # Scale to reasonable audio range
        
        # Method 2: Frequency modulation
        # Map ECG values to frequency changes
        base_freq = (self.freq_range[0] + self.freq_range[1]) / 2
        freq_deviation = (self.freq_range[1] - self.freq_range[0]) / 4
        
        time_samples = torch.arange(self.target_length, dtype=torch.float32) / self.sample_rate
        
        # Create time-varying frequency based on ECG
        instantaneous_freq = base_freq + freq_deviation * ecg_signal
        
        # Generate phase
        phase = torch.cumsum(2 * np.pi * instantaneous_freq / self.sample_rate, dim=0)
        
        # Generate audio signal
        audio_fm = 0.3 * torch.sin(phase)
        
        # Method 3: Amplitude modulation
        carrier_freq = base_freq
        carrier_phase = 2 * np.pi * carrier_freq * time_samples
        carrier = torch.sin(carrier_phase)
        
        # Modulate amplitude with ECG signal
        modulation = 0.5 + 0.5 * ecg_signal  # Ensure positive modulation
        audio_am = 0.3 * modulation * carrier
        
        # Combine methods (ensemble approach)
        audio_combined = 0.4 * audio_direct + 0.3 * audio_fm + 0.3 * audio_am
        
        # Add slight noise for robustness
        noise = torch.randn_like(audio_combined) * 0.01
        audio_final = audio_combined + noise
        
        # Ensure proper range for HuBERT
        audio_final = torch.clamp(audio_final, -1.0, 1.0)
        
        return audio_final
    
    def batch_convert(self, images: torch.Tensor, 
                     masks: torch.Tensor = None) -> torch.Tensor:
        """
        Convert batch of ECG images to audio representations
        
        Args:
            images: Batch of ECG images [batch, C, H, W]
            masks: Optional batch of masks [batch, H, W]
            
        Returns:
            Batch of audio waveforms [batch, target_length]
        """
        batch_size = images.size(0)
        audio_batch = []
        
        for i in range(batch_size):
            mask = masks[i] if masks is not None else None
            
            # Extract ECG signal
            ecg_signal = self.ecg_image_to_signal(images[i], mask)
            
            # Convert to audio
            audio = self.signal_to_audio(ecg_signal)
            
            audio_batch.append(audio)
        
        return torch.stack(audio_batch)

class ECGHubertFeatureExtractor(nn.Module):
    """
    HuBERT-based feature extractor adapted for ECG analysis
    """
    
    def __init__(self, 
                 hubert_model_name: str = "facebook/hubert-base-ls960",
                 freeze_layers: int = 8,
                 output_dim: int = 768):
        super().__init__()
        
        # Load pretrained HuBERT
        self.hubert_config = HubertConfig.from_pretrained(hubert_model_name)
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        
        # Freeze early layers (they capture low-level audio features)
        if freeze_layers > 0:
            for i, layer in enumerate(self.hubert.encoder.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # ECG-specific adaptation layers
        self.ecg_adapter = nn.Sequential(
            nn.Linear(self.hubert_config.hidden_size, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Temporal attention for sequence pooling
        self.temporal_attention = nn.MultiheadAttention(
            output_dim, 8, dropout=0.1, batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, audio_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from audio-converted ECG signals
        
        Args:
            audio_inputs: Audio waveforms [batch, sequence_length]
            
        Returns:
            Dictionary of extracted features
        """
        # HuBERT forward pass
        hubert_outputs = self.hubert(audio_inputs)
        
        # Get sequence features
        sequence_features = hubert_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Adapt features for ECG
        adapted_features = self.ecg_adapter(sequence_features)  # [batch, seq_len, output_dim]
        
        # Temporal attention for pooling
        attended_features, attention_weights = self.temporal_attention(
            adapted_features, adapted_features, adapted_features
        )
        
        # Global pooling
        pooled_features = torch.mean(attended_features, dim=1)  # [batch, output_dim]
        
        # Final projection
        final_features = self.final_proj(pooled_features)
        
        return {
            'sequence_features': adapted_features,
            'pooled_features': final_features,
            'attention_weights': attention_weights,
            'hubert_features': sequence_features
        }

class ECGHubertPQRSTDetector(nn.Module):
    """
    PQRST detection head for HuBERT-ECG model
    """
    
    def __init__(self, input_dim: int = 768, num_points: int = 5):
        super().__init__()
        
        self.num_points = num_points
        
        # Point detection heads
        self.point_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_points + 1)  # +1 for background
        )
        
        self.point_regressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_points * 2)  # x, y for each point
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, num_points)
        )
        
        # Temporal sequence modeling for fine-grained detection
        self.sequence_decoder = nn.LSTM(
            input_dim, input_dim // 2, 
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        self.sequence_classifier = nn.Linear(input_dim, num_points + 1)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Detect PQRST points from HuBERT features
        
        Args:
            features: Features from ECGHubertFeatureExtractor
            
        Returns:
            Dictionary of PQRST predictions
        """
        pooled_features = features['pooled_features']
        sequence_features = features['sequence_features']
        
        # Global predictions from pooled features
        point_classes = self.point_classifier(pooled_features)
        point_coords = torch.sigmoid(self.point_regressor(pooled_features))
        point_confidence = torch.sigmoid(self.confidence_estimator(pooled_features))
        
        # Reshape coordinates
        batch_size = pooled_features.size(0)
        point_coords = point_coords.view(batch_size, self.num_points, 2)
        
        # Sequence-level predictions
        lstm_output, _ = self.sequence_decoder(sequence_features)
        sequence_classes = self.sequence_classifier(lstm_output)
        
        return {
            'point_classes': point_classes,
            'point_coords': point_coords,
            'point_confidence': point_confidence,
            'sequence_classes': sequence_classes
        }

class HubertECGModel(nn.Module):
    """
    Complete HuBERT-ECG model for PQRST detection
    """
    
    def __init__(self, 
                 hubert_model_name: str = "facebook/hubert-base-ls960",
                 freeze_layers: int = 8,
                 feature_dim: int = 768,
                 num_points: int = 5):
        super().__init__()
        
        # ECG to Audio converter
        self.ecg_to_audio = ECGToAudioConverter()
        
        # HuBERT feature extractor
        self.feature_extractor = ECGHubertFeatureExtractor(
            hubert_model_name=hubert_model_name,
            freeze_layers=freeze_layers,
            output_dim=feature_dim
        )
        
        # PQRST detector
        self.pqrst_detector = ECGHubertPQRSTDetector(
            input_dim=feature_dim,
            num_points=num_points
        )
        
        # Additional heads for multi-task learning
        self.rhythm_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 10)  # Common rhythm types
        )
        
        self.quality_assessor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)  # Signal quality score
        )
    
    def forward(self, images: torch.Tensor, 
                masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HuBERT-ECG model
        
        Args:
            images: ECG images [batch, channels, height, width]
            masks: Optional masks [batch, height, width]
            
        Returns:
            Complete set of predictions
        """
        # Convert ECG images to audio
        audio_inputs = self.ecg_to_audio.batch_convert(images, masks)
        
        # Extract features using HuBERT
        features = self.feature_extractor(audio_inputs)
        
        # PQRST detection
        pqrst_predictions = self.pqrst_detector(features)
        
        # Additional predictions
        rhythm_logits = self.rhythm_classifier(features['pooled_features'])
        quality_scores = torch.sigmoid(self.quality_assessor(features['pooled_features']))
        
        # Combine all predictions
        predictions = {
            **pqrst_predictions,
            'rhythm_logits': rhythm_logits,
            'quality_scores': quality_scores,
            'audio_features': features['pooled_features'],
            'sequence_features': features['sequence_features'],
            'attention_weights': features['attention_weights']
        }
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task losses for HuBERT-ECG model
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # PQRST point classification loss
        if 'point_classes' in targets:
            point_class_loss = F.cross_entropy(
                predictions['point_classes'], 
                targets['point_classes']
            )
            losses['point_class_loss'] = point_class_loss
        
        # PQRST coordinate regression loss
        if 'point_coords' in targets:
            coord_loss = F.mse_loss(
                predictions['point_coords'], 
                targets['point_coords']
            )
            losses['coord_loss'] = coord_loss
        
        # Confidence loss
        if 'point_confidence' in targets:
            confidence_loss = F.binary_cross_entropy(
                predictions['point_confidence'], 
                targets['point_confidence']
            )
            losses['confidence_loss'] = confidence_loss
        
        # Sequence classification loss
        if 'sequence_classes' in targets:
            seq_loss = F.cross_entropy(
                predictions['sequence_classes'].flatten(0, 1), 
                targets['sequence_classes'].flatten()
            )
            losses['sequence_loss'] = seq_loss
        
        # Rhythm classification loss
        if 'rhythm_labels' in targets:
            rhythm_loss = F.cross_entropy(
                predictions['rhythm_logits'], 
                targets['rhythm_labels']
            )
            losses['rhythm_loss'] = rhythm_loss
        
        # Quality assessment loss
        if 'quality_scores' in targets:
            quality_loss = F.mse_loss(
                predictions['quality_scores'], 
                targets['quality_scores']
            )
            losses['quality_loss'] = quality_loss
        
        # Total loss (weighted combination)
        total_loss = (
            losses.get('point_class_loss', 0) * 2.0 +
            losses.get('coord_loss', 0) * 5.0 +
            losses.get('confidence_loss', 0) * 1.0 +
            losses.get('sequence_loss', 0) * 1.5 +
            losses.get('rhythm_loss', 0) * 1.0 +
            losses.get('quality_loss', 0) * 0.5
        )
        
        losses['total_loss'] = total_loss
        
        return losses

def create_hubert_ecg_model(model_size: str = "base") -> HubertECGModel:
    """
    Create HuBERT-ECG model with specified size
    
    Args:
        model_size: HuBERT model size ("base", "large")
        
    Returns:
        HubertECGModel instance
    """
    model_name_map = {
        "base": "facebook/hubert-base-ls960",
        "large": "facebook/hubert-large-ls960-ft"
    }
    
    model_name = model_name_map.get(model_size, model_name_map["base"])
    
    model = HubertECGModel(
        hubert_model_name=model_name,
        freeze_layers=8,
        feature_dim=768,
        num_points=5
    )
    
    print(f"Created HuBERT-ECG model with {model_name}")
    print(f"Frozen first 8 layers for transfer learning")
    
    return model

if __name__ == "__main__":
    # Test HuBERT-ECG model
    print("Creating HuBERT-ECG model...")
    
    model = create_hubert_ecg_model(model_size="base")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 256, 512)  # Batch of ECG images
    dummy_masks = torch.randn(2, 256, 512)
    
    print("Testing ECG to Audio conversion...")
    converter = ECGToAudioConverter()
    audio_batch = converter.batch_convert(dummy_images, dummy_masks)
    print(f"Audio batch shape: {audio_batch.shape}")
    
    print("Testing full model...")
    with torch.no_grad():
        predictions = model(dummy_images, dummy_masks)
    
    print("\nHuBERT-ECG model output shapes:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("HuBERT-ECG model created successfully!")