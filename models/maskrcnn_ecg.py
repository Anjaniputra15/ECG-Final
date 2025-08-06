#!/usr/bin/env python3
"""
Mask R-CNN for ECG Waveform Detection and Segmentation
Precise P, QRS, T wave detection with clinical timing measurements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from scipy import signal
from scipy.signal import find_peaks


class ECGMaskRCNN(nn.Module):
    """
    Mask R-CNN adapted for ECG waveform detection and segmentation.
    
    Detects and segments:
    - P waves
    - QRS complexes  
    - T waves
    - Background
    """
    
    def __init__(self):
        super(ECGMaskRCNN, self).__init__()
        
        # Load pre-trained Mask R-CNN
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # ECG wave classes
        self.wave_classes = {
            'background': 0,
            'P_wave': 1,
            'QRS_complex': 2,
            'T_wave': 3
        }
        
        self.num_classes = len(self.wave_classes)
        
        # Replace classifier heads for ECG-specific classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.num_classes
        )
        
        # ECG-specific parameters for clinical measurements
        self.sampling_rate = 100  # Hz (PTB-XL standard)
        self.image_width = 1024   # ECG image width
        self.image_height = 768   # ECG image height
        self.time_duration = 10   # seconds
        
    def forward(self, images, targets=None):
        """
        Forward pass for training and inference.
        
        Args:
            images: List of ECG images [batch_size, 3, H, W]
            targets: Training targets (optional)
            
        Returns:
            Training: losses dict
            Inference: predictions dict
        """
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def detect_waves(self, image: torch.Tensor, confidence_threshold: float = 0.7) -> Dict:
        """
        Detect ECG waves in a single image with clinical measurements.
        
        Args:
            image: ECG image tensor [3, H, W]
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Detection results with clinical measurements
        """
        self.eval()
        
        with torch.no_grad():
            predictions = self.model([image])
            prediction = predictions[0]
        
        # Filter by confidence
        keep = prediction['scores'] > confidence_threshold
        boxes = prediction['boxes'][keep]
        labels = prediction['labels'][keep] 
        scores = prediction['scores'][keep]
        masks = prediction['masks'][keep]
        
        # Group detections by wave type
        wave_detections = {
            'P_waves': [],
            'QRS_complexes': [],
            'T_waves': []
        }
        
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            wave_type = self._get_wave_name(label.item())
            
            if wave_type != 'background':
                detection = {
                    'bbox': box.cpu().numpy(),
                    'confidence': score.item(),
                    'mask': mask.squeeze().cpu().numpy(),
                    'timing': self._calculate_timing(box),
                    'morphology': self._analyze_morphology(mask.squeeze(), box)
                }
                
                if wave_type == 'P_wave':
                    wave_detections['P_waves'].append(detection)
                elif wave_type == 'QRS_complex':
                    wave_detections['QRS_complexes'].append(detection)
                elif wave_type == 'T_wave':
                    wave_detections['T_waves'].append(detection)
        
        # Sort by temporal order (left to right)
        for wave_type in wave_detections:
            wave_detections[wave_type].sort(key=lambda x: x['bbox'][0])
        
        # Calculate clinical intervals
        clinical_measurements = self._calculate_clinical_intervals(wave_detections)
        
        return {
            'wave_detections': wave_detections,
            'clinical_measurements': clinical_measurements,
            'total_detections': len(boxes),
            'detection_summary': self._create_detection_summary(wave_detections)
        }
    
    def _get_wave_name(self, label_id: int) -> str:
        """Convert label ID to wave name."""
        for name, id in self.wave_classes.items():
            if id == label_id:
                return name
        return 'unknown'
    
    def _calculate_timing(self, bbox: torch.Tensor) -> Dict:
        """
        Calculate timing information from bounding box coordinates.
        
        Args:
            bbox: [x1, y1, x2, y2] in image coordinates
            
        Returns:
            Timing measurements in milliseconds and seconds
        """
        x1, y1, x2, y2 = bbox.cpu().numpy()
        
        # Convert pixel coordinates to time
        # Image represents 10 seconds across width
        start_time_s = (x1 / self.image_width) * self.time_duration
        end_time_s = (x2 / self.image_width) * self.time_duration
        duration_s = end_time_s - start_time_s
        
        # Convert to milliseconds
        start_time_ms = start_time_s * 1000
        end_time_ms = end_time_s * 1000
        duration_ms = duration_s * 1000
        
        return {
            'start_time_s': start_time_s,
            'end_time_s': end_time_s,
            'duration_s': duration_s,
            'start_time_ms': start_time_ms,
            'end_time_ms': end_time_ms,
            'duration_ms': duration_ms,
            'center_time_s': (start_time_s + end_time_s) / 2
        }
    
    def _analyze_morphology(self, mask: np.ndarray, bbox: torch.Tensor) -> Dict:
        """
        Analyze wave morphology from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            bbox: Bounding box coordinates
            
        Returns:
            Morphological characteristics
        """
        x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
        
        # Extract wave region
        wave_mask = mask[y1:y2, x1:x2]
        
        if wave_mask.sum() == 0:
            return {'area': 0, 'height': 0, 'width': 0}
        
        # Calculate basic morphology
        area = np.sum(wave_mask > 0.5)
        height = y2 - y1
        width = x2 - x1
        
        # Find wave amplitude (highest point)
        wave_coords = np.where(wave_mask > 0.5)
        if len(wave_coords[0]) > 0:
            # Amplitude relative to baseline (assuming middle of image is baseline)
            baseline_y = self.image_height // 2
            min_y = np.min(wave_coords[0]) + y1
            amplitude_pixels = abs(baseline_y - min_y)
            
            # Convert to approximate mV (assuming standard ECG calibration)
            amplitude_mv = amplitude_pixels * (1.0 / 100)  # Rough calibration
        else:
            amplitude_mv = 0
        
        return {
            'area_pixels': area,
            'height_pixels': height,
            'width_pixels': width,
            'amplitude_mv': amplitude_mv,
            'aspect_ratio': width / height if height > 0 else 0
        }
    
    def _calculate_clinical_intervals(self, wave_detections: Dict) -> Dict:
        """
        Calculate standard ECG intervals (PR, QRS, QT, RR).
        
        Args:
            wave_detections: Dictionary of detected waves
            
        Returns:
            Clinical interval measurements
        """
        measurements = {
            'PR_interval_ms': [],
            'QRS_duration_ms': [],
            'QT_interval_ms': [],
            'RR_interval_ms': [],
            'heart_rate_bpm': None
        }
        
        p_waves = wave_detections['P_waves']
        qrs_complexes = wave_detections['QRS_complexes']
        t_waves = wave_detections['T_waves']
        
        # PR Interval: P wave start to QRS start
        for p_wave in p_waves:
            for qrs in qrs_complexes:
                if qrs['timing']['start_time_ms'] > p_wave['timing']['start_time_ms']:
                    pr_interval = qrs['timing']['start_time_ms'] - p_wave['timing']['start_time_ms']
                    if 120 <= pr_interval <= 300:  # Normal PR interval range
                        measurements['PR_interval_ms'].append(pr_interval)
                    break
        
        # QRS Duration
        for qrs in qrs_complexes:
            qrs_duration = qrs['timing']['duration_ms']
            if 60 <= qrs_duration <= 200:  # Normal QRS duration range
                measurements['QRS_duration_ms'].append(qrs_duration)
        
        # QT Interval: QRS start to T wave end
        for i, qrs in enumerate(qrs_complexes):
            if i < len(t_waves):
                t_wave = t_waves[i]
                qt_interval = t_wave['timing']['end_time_ms'] - qrs['timing']['start_time_ms']
                if 300 <= qt_interval <= 500:  # Normal QT interval range
                    measurements['QT_interval_ms'].append(qt_interval)
        
        # RR Interval: QRS to QRS
        for i in range(len(qrs_complexes) - 1):
            rr_interval = (qrs_complexes[i+1]['timing']['start_time_ms'] - 
                          qrs_complexes[i]['timing']['start_time_ms'])
            if 400 <= rr_interval <= 1500:  # Normal RR interval range
                measurements['RR_interval_ms'].append(rr_interval)
        
        # Heart Rate from RR intervals
        if measurements['RR_interval_ms']:
            avg_rr_ms = np.mean(measurements['RR_interval_ms'])
            heart_rate_bpm = 60000 / avg_rr_ms  # Convert to BPM
            measurements['heart_rate_bpm'] = heart_rate_bpm
        
        return measurements
    
    def _create_detection_summary(self, wave_detections: Dict) -> Dict:
        """Create summary of detection results."""
        return {
            'P_waves_detected': len(wave_detections['P_waves']),
            'QRS_complexes_detected': len(wave_detections['QRS_complexes']),
            'T_waves_detected': len(wave_detections['T_waves']),
            'total_beats': len(wave_detections['QRS_complexes']),
            'rhythm_regularity': self._assess_rhythm_regularity(wave_detections['QRS_complexes'])
        }
    
    def _assess_rhythm_regularity(self, qrs_complexes: List) -> str:
        """Assess rhythm regularity from QRS timing."""
        if len(qrs_complexes) < 3:
            return 'insufficient_data'
        
        # Calculate RR interval variability
        rr_intervals = []
        for i in range(len(qrs_complexes) - 1):
            rr = (qrs_complexes[i+1]['timing']['center_time_s'] - 
                  qrs_complexes[i]['timing']['center_time_s']) * 1000
            rr_intervals.append(rr)
        
        if not rr_intervals:
            return 'insufficient_data'
        
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        
        # Coefficient of variation
        cv = rr_std / rr_mean if rr_mean > 0 else 0
        
        if cv < 0.1:
            return 'regular'
        elif cv < 0.2:
            return 'slightly_irregular'
        else:
            return 'irregular'
    
    def clinical_report(self, image: torch.Tensor, patient_id: str = None) -> Dict:
        """
        Generate comprehensive clinical report from ECG analysis.
        
        Args:
            image: ECG image tensor
            patient_id: Optional patient identifier
            
        Returns:
            Comprehensive clinical report
        """
        # Detect waves and calculate measurements
        results = self.detect_waves(image)
        
        wave_detections = results['wave_detections']
        clinical_measurements = results['clinical_measurements']
        detection_summary = results['detection_summary']
        
        # Clinical interpretation
        interpretation = self._interpret_measurements(clinical_measurements, detection_summary)
        
        return {
            'patient_id': patient_id or 'Unknown',
            'analysis_type': 'ECG Wave Detection & Measurement',
            'model': 'Mask R-CNN ECG',
            'wave_detections': wave_detections,
            'clinical_measurements': clinical_measurements,
            'detection_summary': detection_summary,
            'clinical_interpretation': interpretation,
            'recommendations': self._generate_recommendations(interpretation)
        }
    
    def _interpret_measurements(self, measurements: Dict, summary: Dict) -> Dict:
        """Interpret clinical measurements."""
        interpretation = {
            'heart_rate_assessment': 'normal',
            'rhythm_assessment': 'normal',
            'conduction_assessment': 'normal',
            'findings': []
        }
        
        # Heart rate assessment
        if measurements['heart_rate_bpm']:
            hr = measurements['heart_rate_bpm']
            if hr < 60:
                interpretation['heart_rate_assessment'] = 'bradycardia'
                interpretation['findings'].append(f'Bradycardia: {hr:.0f} BPM')
            elif hr > 100:
                interpretation['heart_rate_assessment'] = 'tachycardia'
                interpretation['findings'].append(f'Tachycardia: {hr:.0f} BPM')
            else:
                interpretation['findings'].append(f'Normal heart rate: {hr:.0f} BPM')
        
        # PR interval assessment
        if measurements['PR_interval_ms']:
            avg_pr = np.mean(measurements['PR_interval_ms'])
            if avg_pr > 200:
                interpretation['conduction_assessment'] = 'abnormal'
                interpretation['findings'].append(f'Prolonged PR interval: {avg_pr:.0f} ms')
            elif avg_pr < 120:
                interpretation['conduction_assessment'] = 'abnormal'
                interpretation['findings'].append(f'Short PR interval: {avg_pr:.0f} ms')
        
        # QRS duration assessment
        if measurements['QRS_duration_ms']:
            avg_qrs = np.mean(measurements['QRS_duration_ms'])
            if avg_qrs > 120:
                interpretation['conduction_assessment'] = 'abnormal'
                interpretation['findings'].append(f'Wide QRS complex: {avg_qrs:.0f} ms')
        
        # Rhythm assessment
        rhythm = summary['rhythm_regularity']
        if rhythm == 'irregular':
            interpretation['rhythm_assessment'] = 'abnormal'
            interpretation['findings'].append('Irregular rhythm detected')
        
        return interpretation
    
    def _generate_recommendations(self, interpretation: Dict) -> List[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []
        
        if interpretation['heart_rate_assessment'] == 'bradycardia':
            recommendations.extend([
                'Evaluate for symptomatic bradycardia',
                'Consider pacemaker evaluation if symptomatic',
                'Review medications that may cause bradycardia'
            ])
        
        if interpretation['heart_rate_assessment'] == 'tachycardia':
            recommendations.extend([
                'Evaluate underlying cause of tachycardia',
                'Consider rate control if appropriate',
                'Assess hemodynamic stability'
            ])
        
        if interpretation['conduction_assessment'] == 'abnormal':
            recommendations.extend([
                'Cardiology consultation recommended',
                'Consider electrophysiology evaluation',
                'Serial ECGs for trend monitoring'
            ])
        
        if interpretation['rhythm_assessment'] == 'abnormal':
            recommendations.extend([
                'Holter monitor for rhythm analysis',
                'Evaluate for atrial fibrillation',
                'Consider anticoagulation assessment'
            ])
        
        if not recommendations:
            recommendations = [
                'Normal ECG wave detection and timing',
                'Continue routine cardiac monitoring',
                'Regular follow-up as clinically indicated'
            ]
        
        return recommendations


def create_ecg_maskrcnn(pretrained: bool = True) -> ECGMaskRCNN:
    """
    Factory function to create ECG Mask R-CNN model.
    
    Args:
        pretrained: Whether to use pre-trained weights
        
    Returns:
        ECG Mask R-CNN model
    """
    model = ECGMaskRCNN()
    
    if not pretrained:
        # Initialize with random weights if not using pretrained
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)
    
    return model


if __name__ == "__main__":
    # Test the Mask R-CNN model
    model = create_ecg_maskrcnn(pretrained=True)
    
    # Test input
    x = torch.randn(1, 3, 768, 1024)  # ECG image
    
    print(f"Created ECG Mask R-CNN model")
    print(f"Input shape: {x.shape}")
    
    # Test detection (requires trained model)
    model.eval()
    with torch.no_grad():
        try:
            results = model.detect_waves(x[0])
            print(f"Detection test successful")
            print(f"Detected waves: {results['detection_summary']}")
        except Exception as e:
            print(f"Detection test failed (expected without training): {e}")
    
    print("ECG Mask R-CNN model ready for training!")