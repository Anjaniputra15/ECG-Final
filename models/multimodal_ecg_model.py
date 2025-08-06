#!/usr/bin/env python3
"""
Multi-Modal ECG Model Architecture
Combines visual ECG analysis with raw signal processing and clinical metadata.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math

try:
    from transformers import ViTModel, ViTConfig
except ImportError:
    print("Transformers library not available, using simple implementation")
    
    class SimpleViTConfig:
        def __init__(self, image_size=224, patch_size=16, hidden_size=768, **kwargs):
            self.image_size = image_size
            self.patch_size = patch_size
            self.hidden_size = hidden_size
    
    class SimpleViTModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embeddings = nn.Linear(config.patch_size * config.patch_size * 3, config.hidden_size)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(config.hidden_size, 8, batch_first=True),
                num_layers=6
            )
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            
        def forward(self, pixel_values):
            # Simple patch embedding
            B, C, H, W = pixel_values.shape
            patch_size = self.config.patch_size
            patches = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
            patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * patch_size * patch_size)
            
            embeddings = self.embeddings(patches)
            encoded = self.encoder(embeddings)
            pooled = self.pooler(encoded.mean(dim=1))
            
            class Output:
                def __init__(self, last_hidden_state, pooler_output):
                    self.last_hidden_state = last_hidden_state
                    self.pooler_output = pooler_output
            
            return Output(encoded, pooled)
    
    ViTModel = SimpleViTModel
    ViTConfig = SimpleViTConfig

# Import existing components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


class SignalEncoder(nn.Module):
    """
    1D CNN encoder for raw ECG signals.
    Processes 12-lead ECG signals simultaneously.
    """
    
    def __init__(self, 
                 input_length: int = 1000,  # 10 seconds at 100Hz
                 num_leads: int = 12,
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        """
        Initialize signal encoder.
        
        Args:
            input_length: Length of input signal
            num_leads: Number of ECG leads
            hidden_dim: Hidden dimension size
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.input_length = input_length
        self.num_leads = num_leads
        
        # Multi-scale 1D convolutions for each lead
        self.lead_encoders = nn.ModuleList([
            self._create_lead_encoder(input_length, hidden_dim)
            for _ in range(num_leads)
        ])
        
        # Cross-lead attention mechanism
        self.cross_lead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Global temporal pooling
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * num_leads, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
    
    def _create_lead_encoder(self, input_length: int, hidden_dim: int) -> nn.Module:
        """Create encoder for single ECG lead."""
        return nn.Sequential(
            # Multi-scale convolutions
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(input_length // 8)  # Consistent output length
        )
    
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for signal encoder.
        
        Args:
            signals: Raw ECG signals [batch_size, num_leads, signal_length]
            
        Returns:
            Encoded signal features [batch_size, output_dim]
        """
        batch_size = signals.shape[0]
        lead_features = []
        
        # Encode each lead separately
        for i in range(self.num_leads):
            lead_signal = signals[:, i:i+1, :]  # [batch_size, 1, signal_length]
            lead_feat = self.lead_encoders[i](lead_signal)  # [batch_size, hidden_dim, seq_len]
            lead_feat = lead_feat.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
            lead_features.append(lead_feat)
        
        # Stack lead features
        all_leads = torch.stack(lead_features, dim=1)  # [batch_size, num_leads, seq_len, hidden_dim]
        batch_size, num_leads, seq_len, hidden_dim = all_leads.shape
        
        # Reshape for cross-lead attention
        all_leads_reshaped = all_leads.view(batch_size * seq_len, num_leads, hidden_dim)
        
        # Apply cross-lead attention
        attended_leads, _ = self.cross_lead_attention(
            all_leads_reshaped, all_leads_reshaped, all_leads_reshaped
        )
        attended_leads = attended_leads.view(batch_size, seq_len, num_leads, hidden_dim)
        
        # Temporal attention pooling for each lead
        lead_representations = []
        for i in range(num_leads):
            lead_temporal = attended_leads[:, :, i, :]  # [batch_size, seq_len, hidden_dim]
            attention_weights = self.temporal_attention(lead_temporal)  # [batch_size, seq_len, 1]
            weighted_features = (lead_temporal * attention_weights).sum(dim=1)  # [batch_size, hidden_dim]
            lead_representations.append(weighted_features)
        
        # Concatenate all lead representations
        combined_features = torch.cat(lead_representations, dim=1)  # [batch_size, num_leads * hidden_dim]
        
        # Final projection
        output = self.output_projection(combined_features)
        
        return output


class ClinicalMetadataEncoder(nn.Module):
    """
    Encoder for clinical metadata (age, sex, medical history, etc.).
    """
    
    def __init__(self, 
                 numerical_features: int = 5,  # age, height, weight, heart_rate, etc.
                 categorical_features: Dict[str, int] = None,  # {feature: num_categories}
                 output_dim: int = 128):
        """
        Initialize clinical metadata encoder.
        
        Args:
            numerical_features: Number of numerical features
            categorical_features: Dictionary mapping categorical features to number of categories
            output_dim: Output dimension
        """
        super().__init__()
        
        if categorical_features is None:
            categorical_features = {
                'sex': 2,
                'device_type': 5,
                'recording_condition': 3
            }
        
        self.categorical_features = categorical_features
        
        # Numerical feature processing
        self.numerical_encoder = nn.Sequential(
            nn.Linear(numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_cats, 16)
            for name, num_cats in categorical_features.items()
        })
        
        # Combined processing
        total_categorical_dim = sum(16 for _ in categorical_features)
        combined_dim = 64 + total_categorical_dim
        
        self.final_encoder = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, 
                numerical_data: torch.Tensor,
                categorical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for clinical metadata encoder.
        
        Args:
            numerical_data: Numerical features [batch_size, num_numerical]
            categorical_data: Dictionary of categorical features
            
        Returns:
            Encoded metadata features [batch_size, output_dim]
        """
        # Process numerical features
        numerical_features = self.numerical_encoder(numerical_data)
        
        # Process categorical features
        categorical_features = []
        for name, embedding in self.categorical_embeddings.items():
            if name in categorical_data:
                cat_embed = embedding(categorical_data[name])
                categorical_features.append(cat_embed)
        
        if categorical_features:
            categorical_features = torch.cat(categorical_features, dim=1)
        else:
            categorical_features = torch.zeros(
                numerical_data.shape[0], 0, 
                device=numerical_data.device
            )
        
        # Combine all features
        if categorical_features.shape[1] > 0:
            combined_features = torch.cat([numerical_features, categorical_features], dim=1)
        else:
            combined_features = numerical_features
        
        # Final encoding
        output = self.final_encoder(combined_features)
        
        return output


class MultiModalFusionModule(nn.Module):
    """
    Advanced fusion module for combining visual, signal, and clinical features.
    Uses attention-based fusion with modality-specific gating.
    """
    
    def __init__(self, 
                 visual_dim: int = 768,
                 signal_dim: int = 512,
                 clinical_dim: int = 128,
                 fusion_dim: int = 512,
                 num_attention_heads: int = 8):
        """
        Initialize fusion module.
        
        Args:
            visual_dim: Dimension of visual features
            signal_dim: Dimension of signal features
            clinical_dim: Dimension of clinical features
            fusion_dim: Dimension of fused features
            num_attention_heads: Number of attention heads
        """
        super().__init__()
        
        # Projection layers to common dimension
        self.visual_projection = nn.Linear(visual_dim, fusion_dim)
        self.signal_projection = nn.Linear(signal_dim, fusion_dim)
        self.clinical_projection = nn.Linear(clinical_dim, fusion_dim)
        
        # Modality-specific gates
        self.visual_gate = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.Sigmoid()
        )
        self.signal_gate = nn.Sequential(
            nn.Linear(signal_dim, fusion_dim),
            nn.Sigmoid()
        )
        self.clinical_gate = nn.Sequential(
            nn.Linear(clinical_dim, fusion_dim),
            nn.Sigmoid()
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_attention_heads,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final output projection
        self.output_projection = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(self, 
                visual_features: torch.Tensor,
                signal_features: torch.Tensor,
                clinical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for fusion module.
        
        Args:
            visual_features: Visual features from ECG images
            signal_features: Features from raw ECG signals
            clinical_features: Clinical metadata features
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        # Project to common dimension
        visual_proj = self.visual_projection(visual_features)
        signal_proj = self.signal_projection(signal_features)
        clinical_proj = self.clinical_projection(clinical_features)
        
        # Apply modality-specific gating
        visual_gated = visual_proj * self.visual_gate(visual_features)
        signal_gated = signal_proj * self.signal_gate(signal_features)
        clinical_gated = clinical_proj * self.clinical_gate(clinical_features)
        
        # Stack modalities for attention
        modalities = torch.stack([visual_gated, signal_gated, clinical_gated], dim=1)
        
        # Apply cross-modal attention
        attended_modalities, attention_weights = self.cross_attention(
            modalities, modalities, modalities
        )
        
        # Flatten for fusion
        flattened = attended_modalities.flatten(start_dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion_layers(flattened)
        
        # Final projection
        output = self.output_projection(fused_features)
        
        return {
            'fused_features': output,
            'attention_weights': attention_weights,
            'modality_contributions': {
                'visual': visual_gated,
                'signal': signal_gated,
                'clinical': clinical_gated
            }
        }


class MultiModalECGModel(nn.Module):
    """
    Complete multi-modal ECG analysis model.
    Combines visual ECG analysis, raw signal processing, and clinical metadata.
    """
    
    def __init__(self, 
                 num_classes: int = 6,
                 visual_model_config: Optional[Dict] = None,
                 signal_config: Optional[Dict] = None,
                 clinical_config: Optional[Dict] = None,
                 fusion_config: Optional[Dict] = None):
        """
        Initialize multi-modal ECG model.
        
        Args:
            num_classes: Number of diagnostic classes
            visual_model_config: Configuration for visual encoder
            signal_config: Configuration for signal encoder
            clinical_config: Configuration for clinical encoder
            fusion_config: Configuration for fusion module
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Default configurations
        if visual_model_config is None:
            visual_model_config = {
                'image_size': 224,
                'patch_size': 16,
                'hidden_size': 768
            }
        
        if signal_config is None:
            signal_config = {
                'input_length': 1000,
                'num_leads': 12,
                'output_dim': 512
            }
        
        if clinical_config is None:
            clinical_config = {
                'numerical_features': 5,
                'output_dim': 128
            }
        
        if fusion_config is None:
            fusion_config = {
                'fusion_dim': 512,
                'num_attention_heads': 8
            }
        
        # Visual encoder (ViT-based)
        vit_config = ViTConfig(
            image_size=visual_model_config['image_size'],
            patch_size=visual_model_config['patch_size'],
            hidden_size=visual_model_config['hidden_size'],
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        self.visual_encoder = ViTModel(vit_config)
        
        # Signal encoder
        self.signal_encoder = SignalEncoder(**signal_config)
        
        # Clinical metadata encoder
        self.clinical_encoder = ClinicalMetadataEncoder(**clinical_config)
        
        # Fusion module
        self.fusion_module = MultiModalFusionModule(
            visual_dim=visual_model_config['hidden_size'],
            signal_dim=signal_config['output_dim'],
            clinical_dim=clinical_config['output_dim'],
            **fusion_config
        )
        
        # Classification heads
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(fusion_config['fusion_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Additional task-specific heads
        self.severity_classifier = nn.Sequential(
            nn.Linear(fusion_config['fusion_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # Normal, Mild, Moderate, Severe
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(fusion_config['fusion_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                images: torch.Tensor,
                signals: torch.Tensor,
                numerical_metadata: torch.Tensor,
                categorical_metadata: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal model.
        
        Args:
            images: ECG images [batch_size, 3, height, width]
            signals: Raw ECG signals [batch_size, 12, signal_length]
            numerical_metadata: Numerical clinical features [batch_size, num_features]
            categorical_metadata: Categorical clinical features
            
        Returns:
            Dictionary with all predictions and intermediate features
        """
        # Extract features from each modality
        visual_outputs = self.visual_encoder(images)
        visual_features = visual_outputs.pooler_output  # [batch_size, hidden_size]
        
        signal_features = self.signal_encoder(signals)  # [batch_size, signal_dim]
        
        clinical_features = self.clinical_encoder(
            numerical_metadata, categorical_metadata
        )  # [batch_size, clinical_dim]
        
        # Fuse modalities
        fusion_output = self.fusion_module(
            visual_features, signal_features, clinical_features
        )
        fused_features = fusion_output['fused_features']
        
        # Generate predictions
        diagnosis_logits = self.diagnosis_classifier(fused_features)
        severity_logits = self.severity_classifier(fused_features)
        confidence_scores = self.confidence_estimator(fused_features)
        
        return {
            # Main predictions
            'diagnosis_logits': diagnosis_logits,
            'severity_logits': severity_logits,
            'confidence_scores': confidence_scores,
            
            # Intermediate features
            'visual_features': visual_features,
            'signal_features': signal_features,
            'clinical_features': clinical_features,
            'fused_features': fused_features,
            
            # Attention information
            'fusion_attention': fusion_output['attention_weights'],
            'modality_contributions': fusion_output['modality_contributions']
        }
    
    def predict_with_uncertainty(self, 
                                images: torch.Tensor,
                                signals: torch.Tensor,
                                numerical_metadata: torch.Tensor,
                                categorical_metadata: Dict[str, torch.Tensor],
                                n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Prediction with uncertainty estimation using Monte Carlo Dropout.
        
        Args:
            inputs: Same as forward method
            n_samples: Number of MC samples
            
        Returns:
            Predictions with uncertainty estimates
        """
        self.train()  # Enable dropout
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(images, signals, numerical_metadata, categorical_metadata)
                predictions.append(torch.softmax(output['diagnosis_logits'], dim=1))
                confidences.append(output['confidence_scores'])
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [n_samples, batch_size, num_classes]
        confidences = torch.stack(confidences)  # [n_samples, batch_size, 1]
        
        # Calculate statistics
        mean_predictions = predictions.mean(0)
        prediction_uncertainty = predictions.var(0).sum(1)  # Total variance
        mean_confidence = confidences.mean(0)
        confidence_uncertainty = confidences.var(0)
        
        return {
            'mean_predictions': mean_predictions,
            'prediction_uncertainty': prediction_uncertainty,
            'mean_confidence': mean_confidence,
            'confidence_uncertainty': confidence_uncertainty,
            'epistemic_uncertainty': prediction_uncertainty,  # Model uncertainty
            'aleatoric_uncertainty': 1.0 - mean_confidence  # Data uncertainty
        }


def create_multimodal_ecg_model(num_classes: int = 6, **kwargs) -> MultiModalECGModel:
    """
    Factory function to create multi-modal ECG model.
    
    Args:
        num_classes: Number of diagnostic classes
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured MultiModalECGModel
    """
    return MultiModalECGModel(num_classes=num_classes, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("🔬 Multi-Modal ECG Model")
    print("=" * 50)
    
    # Create model
    model = create_multimodal_ecg_model(num_classes=6)
    
    # Create dummy inputs
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    signals = torch.randn(batch_size, 12, 1000)
    numerical_metadata = torch.randn(batch_size, 5)
    categorical_metadata = {
        'sex': torch.randint(0, 2, (batch_size,)),
        'device_type': torch.randint(0, 5, (batch_size,)),
        'recording_condition': torch.randint(0, 3, (batch_size,))
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, signals, numerical_metadata, categorical_metadata)
    
    print("Model Architecture:")
    print(f"  📸 Visual Encoder: ViT-based")
    print(f"  📊 Signal Encoder: Multi-lead 1D CNN")
    print(f"  🏥 Clinical Encoder: Embedding + MLP")
    print(f"  🔗 Fusion Module: Cross-modal attention")
    
    print(f"\nOutput Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nModel Parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Test uncertainty estimation
    print(f"\nTesting uncertainty estimation...")
    uncertainty_output = model.predict_with_uncertainty(
        images, signals, numerical_metadata, categorical_metadata, n_samples=5
    )
    print(f"  Mean predictions: {uncertainty_output['mean_predictions'].shape}")
    print(f"  Prediction uncertainty: {uncertainty_output['prediction_uncertainty'].shape}")
    
    print("\n✅ Multi-modal ECG model created successfully!")