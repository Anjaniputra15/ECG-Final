#!/usr/bin/env python3
"""
ECG Wave Annotation System
Automated P, QRS, T wave detection and annotation for Mask R-CNN training.
"""

import numpy as np
import cv2
import torch
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image, ImageDraw


class ECGWaveAnnotator:
    """
    Automated ECG wave detection and annotation system.
    Creates bounding boxes and segmentation masks for P, QRS, T waves.
    """
    
    def __init__(self):
        self.wave_classes = {
            'background': 0,
            'P_wave': 1,
            'QRS_complex': 2,
            'T_wave': 3
        }
        
        # ECG analysis parameters
        self.sampling_rate = 100  # Hz
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Wave detection parameters
        self.qrs_params = {
            'height': 0.3,      # Minimum peak height
            'distance': 40,     # Minimum distance between peaks (0.4s at 100Hz)
            'width': (5, 25),   # QRS width range (50-250ms)
            'prominence': 0.2   # Peak prominence
        }
        
        self.p_wave_params = {
            'height': 0.1,
            'distance': 20,
            'width': (8, 25),   # P wave width range (80-250ms)
            'prominence': 0.05
        }
        
        self.t_wave_params = {
            'height': 0.1,
            'distance': 30,
            'width': (15, 50),  # T wave width range (150-500ms)
            'prominence': 0.05
        }
    
    def annotate_ecg_image(self, image_path: str, signal_data: Optional[np.ndarray] = None) -> Dict:
        """
        Create annotations for ECG image with P, QRS, T wave detection.
        
        Args:
            image_path: Path to ECG image
            signal_data: Optional ECG signal data [leads, samples]
            
        Returns:
            Annotation data in COCO format for Mask R-CNN training
        """
        # Load ECG image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # If signal data not provided, extract from image
        if signal_data is None:
            signal_data = self._extract_signal_from_image(image)
        
        # Detect waves in each lead
        all_annotations = []
        annotation_id = 1
        
        # Process each lead (assuming 4x3 grid layout)
        for lead_idx, lead_name in enumerate(self.leads):
            if lead_idx >= signal_data.shape[0]:
                continue
                
            lead_signal = signal_data[lead_idx]
            
            # Get lead position in image
            lead_region = self._get_lead_region(lead_idx, width, height)
            
            # Detect waves in this lead
            wave_detections = self._detect_waves_in_lead(lead_signal, lead_region)
            
            # Convert to annotations
            for wave_type, detections in wave_detections.items():
                for detection in detections:
                    annotation = self._create_annotation(
                        detection, wave_type, annotation_id, lead_name, width, height
                    )
                    all_annotations.append(annotation)
                    annotation_id += 1
        
        # Create COCO format annotation
        coco_annotation = {
            'info': {
                'description': 'ECG Wave Detection Dataset',
                'version': '1.0',
                'contributor': 'ECG-LLM System'
            },
            'images': [{
                'id': 1,
                'file_name': Path(image_path).name,
                'width': width,
                'height': height
            }],
            'annotations': all_annotations,
            'categories': [
                {'id': 1, 'name': 'P_wave', 'supercategory': 'wave'},
                {'id': 2, 'name': 'QRS_complex', 'supercategory': 'wave'},
                {'id': 3, 'name': 'T_wave', 'supercategory': 'wave'}
            ]
        }
        
        return coco_annotation
    
    def _extract_signal_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract ECG signal from image using image processing.
        This is a simplified approach - in practice, use actual signal data.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract signals from each lead region
        height, width = gray.shape
        signals = []
        
        for lead_idx in range(12):  # 12 leads
            lead_region = self._get_lead_region(lead_idx, width, height)
            
            # Extract horizontal line (ECG trace) from region
            y_start, y_end, x_start, x_end = lead_region
            lead_image = gray[y_start:y_end, x_start:x_end]
            
            # Find the ECG trace (darkest horizontal line)
            signal_trace = self._extract_trace_from_region(lead_image)
            signals.append(signal_trace)
        
        return np.array(signals)
    
    def _get_lead_region(self, lead_idx: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Get image coordinates for each lead region (4x3 grid).
        
        Returns:
            (y_start, y_end, x_start, x_end)
        """
        rows, cols = 4, 3
        
        row = lead_idx // cols
        col = lead_idx % cols
        
        lead_height = height // rows
        lead_width = width // cols
        
        y_start = row * lead_height + 20  # Small margin
        y_end = (row + 1) * lead_height - 20
        x_start = col * lead_width + 40
        x_end = (col + 1) * lead_width - 40
        
        return (y_start, y_end, x_start, x_end)
    
    def _extract_trace_from_region(self, region: np.ndarray) -> np.ndarray:
        """
        Extract ECG trace from lead region using edge detection.
        """
        # Apply edge detection
        edges = cv2.Canny(region, 50, 150)
        
        # Find horizontal lines (ECG traces)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Extract signal by finding y-coordinates of detected lines
        signal_points = []
        region_height, region_width = region.shape
        
        for x in range(0, region_width, 2):  # Sample every 2 pixels
            col = detected_lines[:, x] if x < region_width else detected_lines[:, -1]
            
            # Find y-coordinate of ECG trace
            trace_points = np.where(col > 0)[0]
            if len(trace_points) > 0:
                # Use median to handle multiple detections
                y_coord = np.median(trace_points)
                # Convert to signal amplitude (invert y-axis)
                amplitude = (region_height / 2 - y_coord) / (region_height / 2)
                signal_points.append(amplitude)
            else:
                # Use previous value or zero
                signal_points.append(signal_points[-1] if signal_points else 0)
        
        return np.array(signal_points)
    
    def _detect_waves_in_lead(self, signal: np.ndarray, lead_region: Tuple) -> Dict:
        """
        Detect P, QRS, T waves in ECG signal.
        
        Args:
            signal: 1D ECG signal
            lead_region: (y_start, y_end, x_start, x_end) in image coordinates
            
        Returns:
            Dictionary of detected waves with positions
        """
        # Preprocess signal
        filtered_signal = self._preprocess_signal(signal)
        
        # Detect QRS complexes first (most prominent)
        qrs_peaks = self._detect_qrs_peaks(filtered_signal)
        
        # Detect P waves (before QRS)
        p_waves = self._detect_p_waves(filtered_signal, qrs_peaks)
        
        # Detect T waves (after QRS)
        t_waves = self._detect_t_waves(filtered_signal, qrs_peaks)
        
        # Convert to image coordinates
        y_start, y_end, x_start, x_end = lead_region
        
        detections = {
            'QRS_complex': self._convert_peaks_to_boxes(qrs_peaks, signal, lead_region, 'QRS'),
            'P_wave': self._convert_peaks_to_boxes(p_waves, signal, lead_region, 'P'),
            'T_wave': self._convert_peaks_to_boxes(t_waves, signal, lead_region, 'T')
        }
        
        return detections
    
    def _preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal for wave detection."""
        if len(signal) < 10:
            return signal
        
        # Bandpass filter (0.5-40 Hz for ECG)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40 / nyquist
        
        try:
            b, a = butter(3, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
        except:
            filtered = signal  # Use original if filtering fails
        
        # Normalize
        if np.std(filtered) > 0:
            filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        
        return filtered
    
    def _detect_qrs_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect QRS complex peaks."""
        # Use derivative for QRS detection (emphasizes sharp changes)
        derivative = np.diff(signal)
        derivative_squared = derivative ** 2
        
        # Find peaks in derivative signal
        peaks, properties = find_peaks(
            derivative_squared,
            height=self.qrs_params['height'],
            distance=self.qrs_params['distance'],
            prominence=self.qrs_params['prominence']
        )
        
        # Refine peak positions by finding actual R-peaks in original signal
        refined_peaks = []
        for peak in peaks:
            # Search for actual R-peak in neighborhood
            start = max(0, peak - 10)
            end = min(len(signal), peak + 10)
            
            if start < end:
                local_region = signal[start:end]
                if len(local_region) > 0:
                    local_peak = np.argmax(np.abs(local_region))
                    refined_peaks.append(start + local_peak)
        
        return np.array(refined_peaks)
    
    def _detect_p_waves(self, signal: np.ndarray, qrs_peaks: np.ndarray) -> np.ndarray:
        """Detect P waves before QRS complexes."""
        p_peaks = []
        
        for qrs_peak in qrs_peaks:
            # Search for P wave before QRS (typically 80-300ms before)
            search_start = max(0, qrs_peak - 50)  # 500ms before at 100Hz
            search_end = max(0, qrs_peak - 5)   # 50ms before
            
            if search_start < search_end and search_end < len(signal):
                search_region = signal[search_start:search_end]
                
                # Find peaks in search region
                local_peaks, _ = find_peaks(
                    search_region,
                    height=self.p_wave_params['height'],
                    distance=self.p_wave_params['distance'],
                    prominence=self.p_wave_params['prominence']
                )
                
                # Take the closest peak to QRS
                if len(local_peaks) > 0:
                    closest_peak = local_peaks[-1]  # Last peak before QRS
                    p_peaks.append(search_start + closest_peak)
        
        return np.array(p_peaks)
    
    def _detect_t_waves(self, signal: np.ndarray, qrs_peaks: np.ndarray) -> np.ndarray:
        """Detect T waves after QRS complexes."""
        t_peaks = []
        
        for qrs_peak in qrs_peaks:
            # Search for T wave after QRS (typically 200-400ms after)
            search_start = min(len(signal), qrs_peak + 20)  # 200ms after
            search_end = min(len(signal), qrs_peak + 60)    # 600ms after
            
            if search_start < search_end:
                search_region = signal[search_start:search_end]
                
                # Find peaks in search region
                local_peaks, _ = find_peaks(
                    search_region,
                    height=self.t_wave_params['height'],
                    distance=self.t_wave_params['distance'],
                    prominence=self.t_wave_params['prominence']
                )
                
                # Take the first significant peak after QRS
                if len(local_peaks) > 0:
                    first_peak = local_peaks[0]
                    t_peaks.append(search_start + first_peak)
        
        return np.array(t_peaks)
    
    def _convert_peaks_to_boxes(
        self, 
        peaks: np.ndarray, 
        signal: np.ndarray, 
        lead_region: Tuple, 
        wave_type: str
    ) -> List[Dict]:
        """Convert peak positions to bounding boxes and masks."""
        if len(peaks) == 0:
            return []
        
        y_start, y_end, x_start, x_end = lead_region
        region_width = x_end - x_start
        region_height = y_end - y_start
        
        boxes = []
        
        # Define wave-specific widths
        wave_widths = {
            'QRS': 25,  # ~250ms
            'P': 20,    # ~200ms  
            'T': 40     # ~400ms
        }
        
        half_width = wave_widths.get(wave_type, 25)
        
        for peak in peaks:
            if peak >= len(signal):
                continue
                
            # Convert signal index to x-coordinate in image  
            x_center = x_start + int((peak / len(signal)) * region_width)
            
            # Convert signal amplitude to y-coordinate
            amplitude = signal[peak]
            y_center = y_start + int(region_height / 2 - amplitude * region_height / 4)
            
            # Create bounding box
            x1 = max(x_start, x_center - half_width)
            x2 = min(x_end, x_center + half_width)
            y1 = max(y_start, y_center - 20)
            y2 = min(y_end, y_center + 20)
            
            # Create segmentation mask
            mask = self._create_wave_mask(x1, y1, x2, y2, wave_type)
            
            boxes.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                'peak_position': peak,
                'amplitude': amplitude,
                'mask': mask
            })
        
        return boxes
    
    def _create_wave_mask(self, x1: int, y1: int, x2: int, y2: int, wave_type: str) -> List:
        """Create segmentation mask for wave."""
        # Create simple rectangular mask for now
        # In practice, could create more sophisticated wave-shaped masks
        
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            return []
        
        # Create mask as list of polygon points
        if wave_type == 'QRS':
            # Sharp, narrow mask for QRS
            mask_points = [
                x1 + width//4, y2,           # Bottom left
                x1 + width//2, y1,           # Top center
                x1 + 3*width//4, y2,         # Bottom right
                x1 + width//4, y2            # Close polygon
            ]
        elif wave_type == 'P':
            # Rounded mask for P wave
            mask_points = [
                x1, y1 + height//2,          # Left center
                x1 + width//4, y1,           # Top left
                x1 + 3*width//4, y1,         # Top right
                x2, y1 + height//2,          # Right center
                x1 + 3*width//4, y2,         # Bottom right
                x1 + width//4, y2,           # Bottom left
                x1, y1 + height//2           # Close polygon
            ]
        else:  # T wave
            # Broader, curved mask for T wave
            mask_points = [
                x1, y2,                      # Bottom left
                x1, y1 + height//3,          # Upper left
                x1 + width//3, y1,           # Top left
                x1 + 2*width//3, y1,         # Top right
                x2, y1 + height//3,          # Upper right
                x2, y2,                      # Bottom right
                x1, y2                       # Close polygon
            ]
        
        return mask_points
    
    def _create_annotation(
        self, 
        detection: Dict, 
        wave_type: str, 
        annotation_id: int, 
        lead_name: str,
        image_width: int,
        image_height: int
    ) -> Dict:
        """Create COCO format annotation."""
        bbox = detection['bbox']
        mask_points = detection['mask']
        
        # Calculate area
        area = bbox[2] * bbox[3]  # width * height
        
        annotation = {
            'id': annotation_id,
            'image_id': 1,
            'category_id': self.wave_classes[wave_type],
            'bbox': bbox,
            'area': area,
            'segmentation': [mask_points] if mask_points else [],
            'iscrowd': 0,
            'attributes': {
                'lead': lead_name,
                'wave_type': wave_type,
                'peak_position': int(detection['peak_position']),
                'amplitude': float(detection['amplitude'])
            }
        }
        
        return annotation
    
    def save_annotations(self, annotations: Dict, output_path: str):
        """Save annotations to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def visualize_annotations(self, image_path: str, annotations: Dict, output_path: str):
        """Visualize annotations on ECG image."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Colors for different wave types
        colors = {
            1: 'red',    # P wave
            2: 'blue',   # QRS complex
            3: 'green'   # T wave
        }
        
        # Draw annotations
        for ann in annotations['annotations']:
            bbox = ann['bbox']
            category_id = ann['category_id']
            color = colors.get(category_id, 'yellow')
            
            # Draw bounding box
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            
            # Draw label
            wave_type = ann['attributes']['wave_type']
            lead = ann['attributes']['lead']
            draw.text((x, y - 15), f"{wave_type}-{lead}", fill=color)
        
        # Save visualization
        image.save(output_path)


def create_ecg_annotations(image_path: str, signal_data: Optional[np.ndarray] = None) -> Dict:
    """
    Convenience function to create ECG annotations.
    
    Args:
        image_path: Path to ECG image
        signal_data: Optional ECG signal data
        
    Returns:
        COCO format annotations
    """
    annotator = ECGWaveAnnotator()
    return annotator.annotate_ecg_image(image_path, signal_data)


if __name__ == "__main__":
    # Test the annotation system
    annotator = ECGWaveAnnotator()
    print("ECG Wave Annotator ready!")
    print(f"Wave classes: {annotator.wave_classes}")
    print("Ready to create P, QRS, T wave annotations for Mask R-CNN training!")