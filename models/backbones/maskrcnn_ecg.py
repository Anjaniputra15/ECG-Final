#!/usr/bin/env python3
"""
Mask R-CNN + Keypoint Detection for ECG Analysis
Practical advanced approach using instance segmentation + keypoint detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

class ECGKeypointRCNN(nn.Module):
    """
    Custom Keypoint R-CNN for ECG PQRST detection
    Combines instance segmentation with keypoint detection
    """
    
    def __init__(self, 
                 num_classes: int = 2,  # Background + ECG lead
                 num_keypoints: int = 5,  # P, Q, R, S, T
                 pretrained: bool = True,
                 trainable_backbone_layers: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        
        # Load pretrained Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers
        )
        
        # Modify box predictor for ECG classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Modify mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_dim = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_dim, num_classes
        )
        
        # Add keypoint predictor
        self.model.roi_heads.keypoint_predictor = ECGKeypointPredictor(
            in_features, num_keypoints
        )
        
        # Custom ECG-specific components
        self.ecg_feature_enhancer = ECGFeatureEnhancer()
        self.pqrst_classifier = PQRSTClassifier(in_features, num_keypoints)
        
    def forward(self, images: List[torch.Tensor], 
                targets: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ECG Keypoint R-CNN
        
        Args:
            images: List of ECG images
            targets: Optional training targets
            
        Returns:
            Dictionary of predictions or losses (training mode)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Enhance ECG-specific features
        enhanced_images = [self.ecg_feature_enhancer(img) for img in images]
        
        # Standard Mask R-CNN forward pass
        if self.training:
            # Add keypoint targets if not present
            targets = self._prepare_keypoint_targets(targets)
            losses = self.model(enhanced_images, targets)
            
            # Add custom ECG losses
            custom_losses = self._compute_custom_losses(enhanced_images, targets)
            losses.update(custom_losses)
            
            return losses
        else:
            # Inference mode
            predictions = self.model(enhanced_images)
            
            # Post-process for ECG-specific outputs
            processed_predictions = self._post_process_predictions(predictions)
            
            return processed_predictions
    
    def _prepare_keypoint_targets(self, targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Prepare keypoint targets for training"""
        enhanced_targets = []
        
        for target in targets:
            enhanced_target = target.copy()
            
            # Ensure keypoints are in the correct format
            if 'keypoints' in target:
                keypoints = target['keypoints']  # [num_objects, num_keypoints, 3]
                
                # Convert PQRST coordinates to keypoint format
                if keypoints.shape[-1] == 2:
                    # Add visibility flag (all keypoints visible)
                    visibility = torch.ones(keypoints.shape[:-1] + (1,))
                    keypoints = torch.cat([keypoints, visibility], dim=-1)
                
                enhanced_target['keypoints'] = keypoints
            
            enhanced_targets.append(enhanced_target)
        
        return enhanced_targets
    
    def _compute_custom_losses(self, images: List[torch.Tensor], 
                             targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute ECG-specific losses"""
        custom_losses = {}
        
        # PQRST ordering consistency loss
        # Ensure P-Q-R-S-T temporal ordering
        ordering_loss = 0.0
        count = 0
        
        for target in targets:
            if 'keypoints' in target:
                keypoints = target['keypoints']  # [num_objects, 5, 3]
                
                for obj_keypoints in keypoints:
                    if obj_keypoints.shape[0] == 5:  # P, Q, R, S, T
                        x_coords = obj_keypoints[:, 0]  # X coordinates
                        
                        # P < Q < R < S < T (generally, with some tolerance)
                        expected_order = torch.argsort(x_coords)
                        actual_order = torch.arange(5)
                        
                        # Penalize deviations from expected order
                        order_diff = torch.abs(expected_order.float() - actual_order.float()).sum()
                        ordering_loss += order_diff
                        count += 1
        
        if count > 0:
            custom_losses['pqrst_ordering_loss'] = ordering_loss / count * 0.1
        
        return custom_losses
    
    def _post_process_predictions(self, predictions: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions for ECG analysis"""
        processed = []
        
        for pred in predictions:
            processed_pred = pred.copy()
            
            # Group keypoints by PQRST type
            if 'keypoints' in pred:
                keypoints = pred['keypoints']  # [num_detections, 5, 3]
                scores = pred['scores']
                
                # Filter high-confidence detections
                high_conf_mask = scores > 0.5
                if high_conf_mask.sum() > 0:
                    filtered_keypoints = keypoints[high_conf_mask]
                    
                    # Organize PQRST points
                    pqrst_points = self._organize_pqrst_points(filtered_keypoints)
                    processed_pred['pqrst_points'] = pqrst_points
            
            processed.append(processed_pred)
        
        return processed
    
    def _organize_pqrst_points(self, keypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Organize detected keypoints into PQRST structure"""
        pqrst_labels = ['P', 'Q', 'R', 'S', 'T']
        organized_points = {}
        
        if len(keypoints) > 0:
            # Average keypoints from multiple detections
            avg_keypoints = torch.mean(keypoints, dim=0)  # [5, 3]
            
            for i, label in enumerate(pqrst_labels):
                if i < avg_keypoints.shape[0]:
                    point = avg_keypoints[i, :2]  # x, y coordinates
                    visibility = avg_keypoints[i, 2]  # visibility score
                    
                    organized_points[label] = {
                        'coordinates': point,
                        'visibility': visibility
                    }
        
        return organized_points

class ECGFeatureEnhancer(nn.Module):
    """Enhance ECG-specific features before Mask R-CNN processing"""
    
    def __init__(self):
        super().__init__()
        
        # ECG-specific preprocessing
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # Edge enhancement for ECG traces
        self.edge_enhancer = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        
        # Initialize edge detection kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Set edge detection weights
        with torch.no_grad():
            for i in range(3):  # For each channel
                self.edge_enhancer.weight[i, i] = sobel_x
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Enhance ECG image features
        
        Args:
            image: Input ECG image [C, H, W]
            
        Returns:
            Enhanced image
        """
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Contrast enhancement
        enhanced = self.contrast_enhancer(image)
        
        # Edge enhancement
        edges = self.edge_enhancer(image)
        
        # Combine original and enhanced features
        combined = 0.7 * enhanced + 0.3 * edges
        
        # Remove batch dimension if added
        if combined.shape[0] == 1:
            combined = combined.squeeze(0)
        
        return combined

class ECGKeypointPredictor(nn.Module):
    """Custom keypoint predictor for PQRST detection"""
    
    def __init__(self, in_channels: int, num_keypoints: int):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        
        # Keypoint detection layers
        self.kps_score_lowres = nn.Conv2d(in_channels, num_keypoints, 1, 1, 0)
        
        # Upsampling and refinement
        self.up_scale = 2
        self.kps_score_highres = nn.ConvTranspose2d(
            num_keypoints, num_keypoints, 
            self.up_scale, self.up_scale, 0
        )
        
        # PQRST-specific refinement
        self.pqrst_refiner = nn.Sequential(
            nn.Conv2d(num_keypoints, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 1, 1, 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict PQRST keypoints
        
        Args:
            x: Feature maps from ROI
            
        Returns:
            Keypoint heatmaps
        """
        # Initial keypoint prediction
        kps_score = self.kps_score_lowres(x)
        
        # Upsample for higher resolution
        kps_score = self.kps_score_highres(kps_score)
        
        # PQRST-specific refinement
        kps_score = self.pqrst_refiner(kps_score)
        
        return kps_score

class PQRSTClassifier(nn.Module):
    """Additional classifier for PQRST point types"""
    
    def __init__(self, in_features: int, num_keypoints: int):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_keypoints)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify PQRST point types"""
        return self.classifier(features)

class ECGMaskRCNNTrainer:
    """Training utilities for ECG Mask R-CNN"""
    
    def __init__(self, model: ECGKeypointRCNN, device: str = 'mps'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with different learning rates for different components
        params = [
            {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-4},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-3}
        ]
        
        self.optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
    
    def train_step(self, images: List[torch.Tensor], 
                  targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = self.model(images, targets)
        
        # Compute total loss
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in loss_dict.items()}
    
    def evaluate(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Evaluate model on images"""
        self.model.eval()
        
        images = [img.to(self.device) for img in images]
        
        with torch.no_grad():
            predictions = self.model(images)
        
        return predictions

def create_ecg_maskrcnn_model(pretrained: bool = True) -> ECGKeypointRCNN:
    """
    Create ECG Mask R-CNN model
    
    Args:
        pretrained: Whether to use pretrained weights
        
    Returns:
        ECGKeypointRCNN model
    """
    model = ECGKeypointRCNN(
        num_classes=2,  # Background + ECG lead
        num_keypoints=5,  # P, Q, R, S, T
        pretrained=pretrained,
        trainable_backbone_layers=3
    )
    
    return model

def prepare_ecg_data_for_maskrcnn(images: torch.Tensor, 
                                 masks: torch.Tensor,
                                 pqrst_points: torch.Tensor) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Prepare ECG data in Mask R-CNN format
    
    Args:
        images: Batch of ECG images [batch, C, H, W]
        masks: Batch of ECG masks [batch, H, W]
        pqrst_points: Batch of PQRST coordinates [batch, 5, 2]
        
    Returns:
        Tuple of (image_list, target_list)
    """
    image_list = []
    target_list = []
    
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # Image
        image = images[i]
        image_list.append(image)
        
        # Target
        mask = masks[i]
        points = pqrst_points[i]
        
        # Create bounding box from mask
        mask_coords = torch.nonzero(mask > 0.5)
        if len(mask_coords) > 0:
            y_min, x_min = torch.min(mask_coords, dim=0)[0]
            y_max, x_max = torch.max(mask_coords, dim=0)[0]
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
        else:
            # Default bbox if no mask
            h, w = mask.shape
            bbox = torch.tensor([0, 0, w, h], dtype=torch.float32)
        
        # Prepare keypoints (add visibility)
        keypoints = torch.cat([points, torch.ones(5, 1)], dim=1)  # [5, 3]
        keypoints = keypoints.unsqueeze(0)  # [1, 5, 3] - one object
        
        target = {
            'boxes': bbox.unsqueeze(0),  # [1, 4]
            'labels': torch.tensor([1], dtype=torch.int64),  # ECG lead class
            'masks': mask.unsqueeze(0).byte(),  # [1, H, W]
            'keypoints': keypoints
        }
        
        target_list.append(target)
    
    return image_list, target_list

if __name__ == "__main__":
    # Test ECG Mask R-CNN model
    print("Creating ECG Mask R-CNN model...")
    
    model = create_ecg_maskrcnn_model(pretrained=True)
    
    # Test data preparation
    dummy_images = torch.randn(2, 3, 512, 512)
    dummy_masks = torch.randn(2, 512, 512)
    dummy_points = torch.randn(2, 5, 2) * 512  # Random PQRST coordinates
    
    print("Preparing data for Mask R-CNN...")
    image_list, target_list = prepare_ecg_data_for_maskrcnn(
        dummy_images, dummy_masks, dummy_points
    )
    
    print(f"Prepared {len(image_list)} images and {len(target_list)} targets")
    
    # Test forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        predictions = model(image_list)
    
    print(f"\nMask R-CNN predictions for {len(predictions)} images:")
    for i, pred in enumerate(predictions):
        print(f"  Image {i}:")
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("ECG Mask R-CNN model created successfully!")