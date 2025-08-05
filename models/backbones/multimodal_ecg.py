#!/usr/bin/env python3
"""
Multi-Modal ECG Analysis
Combines image features and signal features for superior PQRST detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import Wav2Vec2Model, Wav2Vec2Config
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class SignalDigitizer:
    """Convert ECG images to 1D signals for multi-modal processing"""
    
    def __init__(self, target_length: int = 5000):
        self.target_length = target_length
    
    def image_to_signal(self, image: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Convert ECG image to 1D signal
        
        Args:
            image: ECG image tensor [C, H, W]
            mask: Optional mask tensor [H, W]
            
        Returns:
            1D signal tensor [target_length]
        """
        if len(image.shape) == 3:
            # Convert to grayscale if RGB
            if image.shape[0] == 3:
                image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            else:
                image = image[0]
        
        height, width = image.shape
        
        # Extract signal by column-wise processing
        signal = []
        
        for x in range(width):
            column = image[:, x]
            
            if mask is not None:
                # Use mask to focus on ECG traces
                mask_column = mask[:, x]
                if mask_column.sum() > 0:
                    # Find the darkest pixel in masked region (ECG trace)
                    masked_pixels = column[mask_column > 0.5]
                    if len(masked_pixels) > 0:
                        # Convert to y-coordinate (invert ECG)
                        trace_value = masked_pixels.min()
                        y_coord = height - (trace_value * height)
                        signal.append(y_coord)
                    else:
                        signal.append(height / 2)
                else:
                    signal.append(height / 2)
            else:
                # Without mask, use simple edge detection
                edges = torch.gradient(column, dim=0)[0]
                peak_idx = torch.argmax(torch.abs(edges))
                signal.append(height - peak_idx.float())
        
        signal = torch.tensor(signal, dtype=torch.float32)
        
        # Resample to target length
        if len(signal) != self.target_length:
            signal = F.interpolate(
                signal.unsqueeze(0).unsqueeze(0), 
                size=self.target_length, 
                mode='linear', 
                align_corners=False
            ).squeeze()
        
        # Normalize signal
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        return signal

class ImageFeatureExtractor(nn.Module):
    """CNN-based feature extractor for ECG images"""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        
        # Use ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        
        # Modify first layer for potential grayscale input
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove classification head
        self.backbone.fc = nn.Identity()
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ECG images
        
        Args:
            x: Input images [batch, channels, height, width]
            
        Returns:
            Image features [batch, output_dim]
        """
        # Extract features through backbone
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply spatial attention
        attention_weights = self.spatial_attention(features)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = features * attention_weights
        
        # Project to output dimension
        features = self.feature_proj(features)
        
        return features

class SignalFeatureExtractor(nn.Module):
    """1D CNN + Transformer for ECG signal feature extraction"""
    
    def __init__(self, signal_length: int = 5000, output_dim: int = 768):
        super().__init__()
        
        # 1D CNN layers for local pattern extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)  # Fixed length output
        )
        
        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ECG signals
        
        Args:
            x: Input signals [batch, signal_length]
            
        Returns:
            Signal features [batch, output_dim]
        """
        # Add channel dimension
        x = x.unsqueeze(1)  # [batch, 1, signal_length]
        
        # CNN feature extraction
        conv_features = self.conv_layers(x)  # [batch, 512, 100]
        
        # Transpose for transformer (batch, seq_len, features)
        conv_features = conv_features.transpose(1, 2)  # [batch, 100, 512]
        
        # Transformer encoding
        transformer_features = self.transformer(conv_features)
        
        # Transpose back and final projection
        transformer_features = transformer_features.transpose(1, 2)  # [batch, 512, 100]
        
        # Final projection
        features = self.final_proj(transformer_features)
        
        return features

class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and signal features"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Query, Key, Value projections
        self.image_to_query = nn.Linear(feature_dim, feature_dim)
        self.signal_to_key = nn.Linear(feature_dim, feature_dim)
        self.signal_to_value = nn.Linear(feature_dim, feature_dim)
        
        self.signal_to_query = nn.Linear(feature_dim, feature_dim)
        self.image_to_key = nn.Linear(feature_dim, feature_dim)
        self.image_to_value = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, image_features: torch.Tensor, 
                signal_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention between image and signal features
        
        Args:
            image_features: [batch, feature_dim]
            signal_features: [batch, feature_dim]
            
        Returns:
            Tuple of enhanced (image_features, signal_features)
        """
        batch_size = image_features.size(0)
        
        # Image attending to signal
        img_query = self.image_to_query(image_features)  # [batch, feature_dim]
        sig_key = self.signal_to_key(signal_features)    # [batch, feature_dim]
        sig_value = self.signal_to_value(signal_features) # [batch, feature_dim]
        
        # Attention scores
        img_to_sig_scores = torch.matmul(img_query.unsqueeze(1), sig_key.unsqueeze(2))  # [batch, 1, 1]
        img_to_sig_weights = F.softmax(img_to_sig_scores / np.sqrt(self.feature_dim), dim=-1)
        
        # Attended signal features
        img_attended_sig = img_to_sig_weights * sig_value  # [batch, feature_dim]
        
        # Signal attending to image
        sig_query = self.signal_to_query(signal_features)
        img_key = self.image_to_key(image_features)
        img_value = self.image_to_value(image_features)
        
        sig_to_img_scores = torch.matmul(sig_query.unsqueeze(1), img_key.unsqueeze(2))
        sig_to_img_weights = F.softmax(sig_to_img_scores / np.sqrt(self.feature_dim), dim=-1)
        
        sig_attended_img = sig_to_img_weights * img_value
        
        # Residual connections and layer norm
        enhanced_image_features = self.layer_norm(image_features + self.dropout(img_attended_sig))
        enhanced_signal_features = self.layer_norm(signal_features + self.dropout(sig_attended_img))
        
        return enhanced_image_features, enhanced_signal_features

class MultiModalFusion(nn.Module):
    """Multi-modal fusion module"""
    
    def __init__(self, feature_dim: int = 768, fusion_method: str = "attention"):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                feature_dim * 2, 8, dropout=0.1, batch_first=True
            )
            self.fusion_proj = nn.Linear(feature_dim * 2, feature_dim)
            
        elif fusion_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(feature_dim * 2, feature_dim)
            
        elif fusion_method == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim)
            )
    
    def forward(self, image_features: torch.Tensor, 
                signal_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse image and signal features
        
        Args:
            image_features: [batch, feature_dim]
            signal_features: [batch, feature_dim]
            
        Returns:
            Fused features [batch, feature_dim]
        """
        # Concatenate features
        combined = torch.cat([image_features, signal_features], dim=-1)  # [batch, feature_dim * 2]
        
        if self.fusion_method == "attention":
            # Self-attention on concatenated features
            combined = combined.unsqueeze(1)  # [batch, 1, feature_dim * 2]
            attended, _ = self.fusion_attention(combined, combined, combined)
            fused = self.fusion_proj(attended.squeeze(1))
            
        elif self.fusion_method == "gated":
            # Gated fusion
            gate_weights = self.gate(combined)
            weighted_img = gate_weights * image_features
            weighted_sig = (1 - gate_weights) * signal_features
            fused = self.fusion_proj(torch.cat([weighted_img, weighted_sig], dim=-1))
            
        elif self.fusion_method == "concat":
            # Simple concatenation + MLP
            fused = self.fusion_proj(combined)
        
        return fused

class MultiModalECGModel(nn.Module):
    """
    Complete Multi-Modal ECG model combining image and signal analysis
    """
    
    def __init__(self, 
                 num_classes: int = 6,  # P, Q, R, S, T + background
                 feature_dim: int = 768,
                 signal_length: int = 5000,
                 fusion_method: str = "attention"):
        super().__init__()
        
        self.signal_digitizer = SignalDigitizer(target_length=signal_length)
        
        # Feature extractors
        self.image_extractor = ImageFeatureExtractor(output_dim=feature_dim)
        self.signal_extractor = SignalFeatureExtractor(
            signal_length=signal_length, 
            output_dim=feature_dim
        )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(feature_dim)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(feature_dim, fusion_method)
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        self.keypoint_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 10)  # 5 points × 2 coordinates
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 5)  # Confidence for each PQRST point
        )
    
    def forward(self, images: torch.Tensor, 
                masks: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal model
        
        Args:
            images: ECG images [batch, channels, height, width]
            masks: Optional masks [batch, height, width]
            
        Returns:
            Dictionary of predictions
        """
        batch_size = images.size(0)
        
        # Extract image features
        image_features = self.image_extractor(images)
        
        # Convert images to signals and extract signal features
        signals = []
        for i in range(batch_size):
            mask = masks[i] if masks is not None else None
            signal = self.signal_digitizer.image_to_signal(images[i], mask)
            signals.append(signal)
        
        signals = torch.stack(signals).to(images.device)
        signal_features = self.signal_extractor(signals)
        
        # Cross-modal attention
        enhanced_image_features, enhanced_signal_features = self.cross_attention(
            image_features, signal_features
        )
        
        # Multi-modal fusion
        fused_features = self.fusion(enhanced_image_features, enhanced_signal_features)
        
        # Predictions
        class_logits = self.classification_head(fused_features)
        keypoint_coords = torch.sigmoid(self.keypoint_head(fused_features))
        confidence_scores = torch.sigmoid(self.confidence_head(fused_features))
        
        # Reshape keypoint coordinates
        keypoint_coords = keypoint_coords.view(batch_size, 5, 2)  # [batch, 5_points, xy]
        
        return {
            'class_logits': class_logits,
            'keypoint_coords': keypoint_coords,
            'confidence_scores': confidence_scores,
            'image_features': enhanced_image_features,
            'signal_features': enhanced_signal_features,
            'fused_features': fused_features
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-modal losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Classification loss
        class_loss = F.cross_entropy(
            predictions['class_logits'], 
            targets['classes']
        )
        
        # Keypoint loss (L1 + L2)
        keypoint_l1 = F.l1_loss(
            predictions['keypoint_coords'], 
            targets['keypoints']
        )
        keypoint_l2 = F.mse_loss(
            predictions['keypoint_coords'], 
            targets['keypoints']
        )
        keypoint_loss = keypoint_l1 + keypoint_l2
        
        # Confidence loss
        confidence_loss = F.binary_cross_entropy(
            predictions['confidence_scores'], 
            targets['confidence']
        )
        
        # Multi-modal consistency loss (encourages similar predictions from both modalities)
        img_pred = self.classification_head(predictions['image_features'])
        sig_pred = self.classification_head(predictions['signal_features'])
        consistency_loss = F.kl_div(
            F.log_softmax(img_pred, dim=-1),
            F.softmax(sig_pred, dim=-1),
            reduction='batchmean'
        )
        
        # Total loss
        total_loss = (
            class_loss + 
            5.0 * keypoint_loss + 
            2.0 * confidence_loss + 
            0.5 * consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'keypoint_loss': keypoint_loss,
            'confidence_loss': confidence_loss,
            'consistency_loss': consistency_loss
        }

def create_multimodal_ecg_model(fusion_method: str = "attention") -> MultiModalECGModel:
    """
    Create multi-modal ECG model
    
    Args:
        fusion_method: Method for fusing modalities ("attention", "gated", "concat")
        
    Returns:
        MultiModalECGModel instance
    """
    model = MultiModalECGModel(
        num_classes=6,
        feature_dim=768,
        signal_length=5000,
        fusion_method=fusion_method
    )
    
    return model

if __name__ == "__main__":
    # Test multi-modal model
    print("Creating Multi-Modal ECG model...")
    
    model = create_multimodal_ecg_model(fusion_method="attention")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_masks = torch.randn(2, 224, 224)
    
    with torch.no_grad():
        predictions = model(dummy_images, dummy_masks)
    
    print("Multi-Modal model output shapes:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Multi-Modal ECG model created successfully!")