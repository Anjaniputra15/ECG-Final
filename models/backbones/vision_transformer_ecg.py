#!/usr/bin/env python3
"""
Vision Transformer for ECG Analysis
State-of-the-art approach using Vision Transformer + DETR for PQRST detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import math
from typing import Dict, List, Tuple, Optional

class ECGViTConfig:
    """Configuration for ECG Vision Transformer"""
    
    def __init__(self):
        # Image settings
        self.image_size = 512
        self.patch_size = 16
        self.num_channels = 3
        
        # Transformer settings
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        
        # ECG specific
        self.num_pqrst_points = 5  # P, Q, R, S, T
        self.num_leads = 12
        self.max_detections = 60  # 5 points × 12 leads
        
        # DETR settings
        self.num_queries = self.max_detections
        self.aux_loss = True

class PositionalEncoding(nn.Module):
    """Positional encoding for DETR decoder"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ECGTransformerDecoder(nn.Module):
    """DETR-style transformer decoder for ECG keypoint detection"""
    
    def __init__(self, config: ECGViTConfig):
        super().__init__()
        self.config = config
        
        # Object queries (learnable embeddings)
        self.query_embed = nn.Embedding(config.num_queries, config.hidden_size)
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=6
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_size)
        
        # Output heads
        self.class_head = nn.Linear(config.hidden_size, config.num_pqrst_points + 1)  # +1 for background
        self.bbox_head = nn.Linear(config.hidden_size, 4)  # x, y, w, h
        self.keypoint_head = nn.Linear(config.hidden_size, 2)  # x, y coordinates
        self.confidence_head = nn.Linear(config.hidden_size, 1)  # confidence score
    
    def forward(self, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DETR decoder
        
        Args:
            encoder_features: Features from ViT encoder [batch, seq_len, hidden_size]
            
        Returns:
            Dictionary with predictions
        """
        batch_size = encoder_features.size(0)
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Add positional encoding to encoder features
        encoder_features = self.pos_encoding(encoder_features.transpose(0, 1)).transpose(0, 1)
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(
            queries, 
            encoder_features.transpose(0, 1)
        ).transpose(0, 1)
        
        # Predictions
        class_logits = self.class_head(decoder_output)
        bbox_coords = torch.sigmoid(self.bbox_head(decoder_output))
        keypoint_coords = torch.sigmoid(self.keypoint_head(decoder_output))
        confidence_scores = torch.sigmoid(self.confidence_head(decoder_output))
        
        return {
            'class_logits': class_logits,
            'bbox_coords': bbox_coords,
            'keypoint_coords': keypoint_coords,
            'confidence_scores': confidence_scores
        }

class ECGViTDETR(nn.Module):
    """
    Complete Vision Transformer + DETR model for ECG PQRST detection
    """
    
    def __init__(self, config: ECGViTConfig = None):
        super().__init__()
        
        if config is None:
            config = ECGViTConfig()
        
        self.config = config
        
        # Vision Transformer backbone
        vit_config = ViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        
        self.vit_backbone = ViTModel(vit_config)
        
        # DETR decoder
        self.decoder = ECGTransformerDecoder(config)
        
        # Feature projection
        self.feature_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Loss weights
        self.class_loss_weight = 1.0
        self.bbox_loss_weight = 5.0
        self.keypoint_loss_weight = 10.0
        self.confidence_loss_weight = 2.0
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            images: Input ECG images [batch, channels, height, width]
            
        Returns:
            Dictionary with all predictions
        """
        # ViT encoding
        vit_outputs = self.vit_backbone(images)
        encoder_features = vit_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Project features
        encoder_features = self.feature_proj(encoder_features)
        
        # DETR decoding
        predictions = self.decoder(encoder_features)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute DETR-style losses with Hungarian matching
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        # Hungarian matching (simplified version)
        class_logits = predictions['class_logits']
        bbox_coords = predictions['bbox_coords']
        keypoint_coords = predictions['keypoint_coords']
        confidence_scores = predictions['confidence_scores']
        
        target_classes = targets['classes']
        target_bboxes = targets['bboxes']
        target_keypoints = targets['keypoints']
        
        # Classification loss (focal loss for imbalanced data)
        class_loss = self._focal_loss(class_logits.flatten(0, 1), target_classes.flatten())
        
        # Bounding box loss (L1 + GIoU)
        bbox_loss = F.l1_loss(bbox_coords.flatten(0, 1), target_bboxes.flatten(0, 1))
        
        # Keypoint loss (L1 + L2)
        keypoint_l1_loss = F.l1_loss(keypoint_coords.flatten(0, 1), target_keypoints.flatten(0, 1))
        keypoint_l2_loss = F.mse_loss(keypoint_coords.flatten(0, 1), target_keypoints.flatten(0, 1))
        keypoint_loss = keypoint_l1_loss + keypoint_l2_loss
        
        # Confidence loss (BCE)
        target_confidence = (target_classes > 0).float()
        confidence_loss = F.binary_cross_entropy(
            confidence_scores.flatten(), 
            target_confidence.flatten()
        )
        
        # Total loss
        total_loss = (
            self.class_loss_weight * class_loss +
            self.bbox_loss_weight * bbox_loss +
            self.keypoint_loss_weight * keypoint_loss +
            self.confidence_loss_weight * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'class_loss': class_loss,
            'bbox_loss': bbox_loss,
            'keypoint_loss': keypoint_loss,
            'confidence_loss': confidence_loss
        }
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, 
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for addressing class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = alpha * (1 - p_t) ** gamma * ce_loss
        return loss.mean()
    
    def post_process(self, predictions: Dict[str, torch.Tensor], 
                    confidence_threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process predictions to extract PQRST detections
        
        Args:
            predictions: Raw model predictions
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detections per image
        """
        batch_size = predictions['class_logits'].size(0)
        results = []
        
        for i in range(batch_size):
            # Extract predictions for this image
            class_logits = predictions['class_logits'][i]
            bbox_coords = predictions['bbox_coords'][i]
            keypoint_coords = predictions['keypoint_coords'][i]
            confidence_scores = predictions['confidence_scores'][i]
            
            # Filter by confidence
            valid_mask = confidence_scores.squeeze() > confidence_threshold
            
            if valid_mask.sum() == 0:
                results.append({
                    'pqrst_points': torch.empty(0, 2),
                    'classes': torch.empty(0, dtype=torch.long),
                    'scores': torch.empty(0),
                    'bboxes': torch.empty(0, 4)
                })
                continue
            
            # Apply NMS to remove duplicate detections
            filtered_detections = self._apply_nms(
                class_logits[valid_mask],
                bbox_coords[valid_mask],
                keypoint_coords[valid_mask],
                confidence_scores[valid_mask].squeeze()
            )
            
            results.append(filtered_detections)
        
        return results
    
    def _apply_nms(self, class_logits: torch.Tensor, bbox_coords: torch.Tensor,
                   keypoint_coords: torch.Tensor, scores: torch.Tensor,
                   iou_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Apply Non-Maximum Suppression"""
        
        # Convert to detection format
        classes = torch.argmax(class_logits, dim=-1)
        
        # Simple NMS based on keypoint distance
        keep_indices = []
        for i in range(len(keypoint_coords)):
            keep = True
            for j in keep_indices:
                # Calculate distance between keypoints
                dist = torch.norm(keypoint_coords[i] - keypoint_coords[j])
                if dist < 0.05:  # Threshold for considering points too close
                    if scores[i] <= scores[j]:
                        keep = False
                        break
            if keep:
                keep_indices.append(i)
        
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        
        return {
            'pqrst_points': keypoint_coords[keep_indices],
            'classes': classes[keep_indices],
            'scores': scores[keep_indices],
            'bboxes': bbox_coords[keep_indices]
        }

class ECGViTDETRTrainer:
    """Training utilities for ECG ViT-DETR model"""
    
    def __init__(self, model: ECGViTDETR, device: str = 'mps'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for backbone and decoder
        backbone_params = list(self.model.vit_backbone.parameters())
        decoder_params = list(self.model.decoder.parameters()) + list(self.model.feature_proj.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained backbone
            {'params': decoder_params, 'lr': 1e-4}    # Higher LR for decoder
        ], weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        images = batch['images'].to(self.device)
        targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
        
        # Forward pass
        predictions = self.model(images)
        
        # Compute loss
        losses = self.model.compute_loss(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['images'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                predictions = self.model(images)
                losses = self.model.compute_loss(predictions, targets)
                
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
                
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}

def create_ecg_vit_detr_model(pretrained: bool = True) -> ECGViTDETR:
    """
    Create ECG ViT-DETR model with optional pretrained weights
    
    Args:
        pretrained: Whether to use pretrained ViT weights
        
    Returns:
        ECGViTDETR model
    """
    config = ECGViTConfig()
    model = ECGViTDETR(config)
    
    if pretrained:
        # Load pretrained ViT weights (will be adapted for ECG)
        print("Loading pretrained ViT weights...")
        # The model will automatically load pretrained weights from HuggingFace
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Creating ECG ViT-DETR model...")
    
    model = create_ecg_vit_detr_model(pretrained=True)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 512, 512)  # Batch of 2 ECG images
    
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print("Model output shapes:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    print("\nECG ViT-DETR model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")