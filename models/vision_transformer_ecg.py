#!/usr/bin/env python3
"""
Vision Transformer (ViT) for ECG Classification
Advanced model architecture for improved ECG diagnosis accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
import numpy as np
from typing import Optional, Tuple


class ECGViTClassifier(nn.Module):
    """
    Vision Transformer for ECG Classification with medical-grade accuracy.
    
    Features:
    - Pre-trained ViT-B/16 backbone
    - Multi-head attention for ECG pattern recognition
    - Dropout and regularization for medical reliability
    - Multi-label classification for 11 cardiac conditions
    """
    
    def __init__(
        self, 
        num_classes: int = 11,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.3,
        attention_dropout: float = 0.1
    ):
        super(ECGViTClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained Vision Transformer
        self.backbone = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=""  # Remove global pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.embed_dim
        
        # Medical-grade classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # Attention weights for medical interpretability
        self.attention_weights = None
        
        # Medical condition labels
        self.condition_labels = [
            'Normal', 'Myocardial_Infarction', 'ST_T_Changes',
            'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
            'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
        ]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention extraction for medical interpretability.
        
        Args:
            x: Input ECG images [batch_size, channels, height, width]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        # Extract patch embeddings
        x = self.backbone.patch_embed(x)
        
        # Add class token and positional encoding
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        
        # Apply transformer blocks with attention extraction
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            
            # Store attention weights from last layer for interpretability
            if i == len(self.backbone.blocks) - 1:
                # Get attention weights from the last attention layer
                self.attention_weights = block.attn.attention_weights if hasattr(block.attn, 'attention_weights') else None
        
        # Global average pooling over sequence dimension (excluding class token)
        x = x[:, 1:].mean(dim=1)  # Skip class token, average over patches
        
        # Apply classification head
        logits = self.classifier(x)
        
        # Apply sigmoid for multi-label classification
        return torch.sigmoid(logits)
    
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get attention weights for medical visualization."""
        return self.attention_weights
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with confidence scores for medical decision support.
        
        Returns:
            predictions: Binary predictions [batch_size, num_classes]
            confidences: Confidence scores [batch_size, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            
            # Convert to binary predictions (threshold at 0.5)
            predictions = (logits > 0.5).float()
            
            # Calculate confidence as distance from threshold
            confidences = torch.abs(logits - 0.5) + 0.5
            
            return predictions, confidences
    
    def get_medical_report(self, x: torch.Tensor, patient_id: str = None) -> dict:
        """
        Generate comprehensive medical report from ECG analysis.
        
        Returns:
            report: Dictionary with medical findings and recommendations
        """
        predictions, confidences = self.predict_with_confidence(x)
        
        # Convert to numpy for processing
        pred_np = predictions.cpu().numpy()[0]  # First sample
        conf_np = confidences.cpu().numpy()[0]
        
        # Generate medical findings
        findings = []
        recommendations = []
        
        for i, (condition, pred, conf) in enumerate(zip(self.condition_labels, pred_np, conf_np)):
            if pred == 1:  # Condition detected
                findings.append({
                    'condition': condition,
                    'status': 'DETECTED',
                    'confidence': float(conf),
                    'clinical_significance': self._get_clinical_significance(condition)
                })
                
                # Add recommendations based on condition
                recommendations.extend(self._get_recommendations(condition))
        
        # If no conditions detected, mark as normal
        if not findings:
            findings.append({
                'condition': 'Normal ECG',
                'status': 'CONFIRMED',
                'confidence': float(conf_np[0]),  # Use Normal confidence
                'clinical_significance': 'No acute cardiac abnormalities detected'
            })
        
        report = {
            'patient_id': patient_id or 'Unknown',
            'analysis_type': 'ECG-ViT Analysis',
            'model_version': 'ViT-B/16 Medical',
            'findings': findings,
            'recommendations': list(set(recommendations)),  # Remove duplicates
            'overall_risk': self._calculate_risk_level(findings),
            'requires_urgent_attention': any(f['condition'] in ['Myocardial_Infarction', 'VT'] for f in findings)
        }
        
        return report
    
    def _get_clinical_significance(self, condition: str) -> str:
        """Get clinical significance for each condition."""
        significance_map = {
            'Normal': 'No cardiac abnormalities detected',
            'Myocardial_Infarction': 'CRITICAL: Heart attack detected, immediate medical attention required',
            'ST_T_Changes': 'Non-specific ST-T wave changes, may indicate ischemia',
            'Conduction_Disturbance': 'Abnormal electrical conduction in the heart',
            'Hypertrophy': 'Heart muscle thickening, possible hypertension',
            'Atrial_Fibrillation': 'Irregular heart rhythm, stroke risk increased',
            'Atrial_Flutter': 'Regular but rapid atrial rhythm, requires monitoring',
            'SVT': 'Fast heart rate originating above ventricles',
            'VT': 'CRITICAL: Ventricular tachycardia, immediate intervention needed',
            'Bradycardia': 'Slow heart rate, may cause symptoms',
            'Tachycardia': 'Fast heart rate, underlying cause needs evaluation'
        }
        return significance_map.get(condition, 'Clinical significance requires evaluation')
    
    def _get_recommendations(self, condition: str) -> list:
        """Get medical recommendations based on detected condition."""
        recommendations_map = {
            'Myocardial_Infarction': [
                'URGENT: Contact emergency services immediately',
                'Administer aspirin if not contraindicated',
                'Prepare for immediate cardiac catheterization'
            ],
            'ST_T_Changes': [
                'Cardiology consultation recommended',
                'Consider serial ECGs',
                'Evaluate for cardiac enzymes'
            ],
            'Conduction_Disturbance': [
                'Monitor heart rate and rhythm',
                'Consider electrophysiology consultation',
                'Evaluate need for pacemaker'
            ],
            'Hypertrophy': [
                'Blood pressure monitoring',
                'Echocardiogram recommended',
                'Lifestyle modifications advised'
            ],
            'Atrial_Fibrillation': [
                'Anticoagulation therapy consideration',
                'Rate control medication',
                'Stroke risk assessment (CHA2DS2-VASc)'
            ],
            'Atrial_Flutter': [
                'Rhythm monitoring required',
                'Consider cardioversion',
                'Anticoagulation evaluation'
            ],
            'SVT': [
                'Vagal maneuvers if stable',
                'Adenosine for acute management',
                'EP study consideration'
            ],
            'VT': [
                'URGENT: Immediate cardioversion if unstable',
                'IV antiarrhythmic therapy',
                'ICD evaluation'
            ],
            'Bradycardia': [
                'Monitor symptoms and hemodynamics',
                'Consider pacemaker evaluation',
                'Review medications'
            ],
            'Tachycardia': [
                'Identify underlying cause',
                'Rate control if needed',
                'Holter monitor consideration'
            ]
        }
        return recommendations_map.get(condition, ['General cardiology consultation recommended'])
    
    def _calculate_risk_level(self, findings: list) -> str:
        """Calculate overall cardiovascular risk level."""
        high_risk_conditions = ['Myocardial_Infarction', 'VT']
        medium_risk_conditions = ['Atrial_Fibrillation', 'SVT', 'Conduction_Disturbance']
        
        detected_conditions = [f['condition'] for f in findings if f['status'] == 'DETECTED']
        
        if any(cond in high_risk_conditions for cond in detected_conditions):
            return 'HIGH RISK'
        elif any(cond in medium_risk_conditions for cond in detected_conditions):
            return 'MODERATE RISK'
        elif len(detected_conditions) > 2:
            return 'MODERATE RISK'
        else:
            return 'LOW RISK'


class ECGViTEnsemble(nn.Module):
    """
    Ensemble of multiple ViT models for improved medical reliability.
    """
    
    def __init__(self, model_configs: list, num_classes: int = 11):
        super(ECGViTEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            ECGViTClassifier(num_classes=num_classes, **config) 
            for config in model_configs
        ])
        
        self.num_models = len(self.models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with voting."""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Average ensemble predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty quantification using ensemble disagreement.
        
        Returns:
            predictions: Ensemble predictions
            uncertainty: Prediction uncertainty (standard deviation)
        """
        with torch.no_grad():
            predictions = []
            
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            predictions = torch.stack(predictions)  # [num_models, batch_size, num_classes]
            
            # Calculate mean and uncertainty
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.std(dim=0)
            
            return mean_pred, uncertainty


def create_ecg_vit_model(
    num_classes: int = 11,
    model_type: str = "base",
    ensemble: bool = False
) -> nn.Module:
    """
    Factory function to create ECG ViT models.
    
    Args:
        num_classes: Number of cardiac conditions to classify
        model_type: Model size - "base", "large", or "huge"
        ensemble: Whether to create ensemble model
        
    Returns:
        ECG ViT model
    """
    
    model_configs = {
        "base": {"model_name": "vit_base_patch16_224", "dropout": 0.3},
        "large": {"model_name": "vit_large_patch16_224", "dropout": 0.4},
        "huge": {"model_name": "vit_huge_patch14_224", "dropout": 0.5}
    }
    
    if ensemble:
        # Create ensemble with different configurations
        configs = [
            model_configs["base"],
            {**model_configs["base"], "dropout": 0.2},
            {**model_configs["base"], "dropout": 0.4}
        ]
        return ECGViTEnsemble(configs, num_classes)
    else:
        config = model_configs.get(model_type, model_configs["base"])
        return ECGViTClassifier(num_classes=num_classes, **config)


if __name__ == "__main__":
    # Test the ViT model
    model = create_ecg_vit_model(num_classes=11, model_type="base")
    
    # Test input
    x = torch.randn(1, 3, 224, 224)  # Batch of 1 ECG image
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities: {output}")
    
    # Generate medical report
    report = model.get_medical_report(x, patient_id="TEST001")
    print(f"\nMedical Report:")
    for key, value in report.items():
        print(f"{key}: {value}")