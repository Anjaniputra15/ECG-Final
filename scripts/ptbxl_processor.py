#!/usr/bin/env python3
"""
PTB-XL Dataset Processor
Converts PTB-XL WFDB format ECG signals to standardized ECG images for training.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import wfdb
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PTBXLProcessor:
    """Process PTB-XL dataset and convert to ECG images."""
    
    def __init__(self, ptbxl_path: str, output_path: str):
        self.ptbxl_path = Path(ptbxl_path)
        self.output_path = Path(output_path)
        self.records_base_path = self.ptbxl_path / "ptb-xl-records/physionet.org/files/ptb-xl/1.0.3"
        
        # Load metadata
        self.df = pd.read_csv(self.ptbxl_path / "ptbxl_database.csv", index_col='ecg_id')
        self.scp_statements = pd.read_csv(self.ptbxl_path / "scp_statements.csv", index_col=0)
        
        # ECG parameters
        self.sampling_rate = 100  # Hz
        self.duration = 10  # seconds
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Image parameters
        self.img_width = 1024
        self.img_height = 768
        self.dpi = 100
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directory structure."""
        dirs = [
            'images', 'masks', 'annotations', 'metadata'
        ]
        for dir_name in dirs:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
            
    def get_diagnostic_labels(self, scp_codes: str) -> Dict[str, float]:
        """Convert SCP codes to diagnostic labels."""
        if pd.isna(scp_codes):
            return {}
            
        # Parse SCP codes
        scp_dict = eval(scp_codes) if isinstance(scp_codes, str) else scp_codes
        
        # Map to our 11 diagnostic categories
        diagnostic_mapping = {
            'NORM': 'Normal',
            'MI': 'Myocardial_Infarction', 
            'STTC': 'ST_T_Changes',
            'CD': 'Conduction_Disturbance',
            'HYP': 'Hypertrophy',
            'PAC': 'Atrial_Fibrillation',
            'PVC': 'Atrial_Flutter', 
            'AFIB': 'Atrial_Fibrillation',
            'AFLT': 'Atrial_Flutter',
            'SVTAC': 'SVT',
            'BIGU': 'VT',
            'TRIGU': 'VT',
            'BRADY': 'Bradycardia',
            'TACHY': 'Tachycardia'
        }
        
        labels = {}
        for scp_code, confidence in scp_dict.items():
            if scp_code in diagnostic_mapping:
                labels[diagnostic_mapping[scp_code]] = confidence
                
        return labels
        
    def load_ecg_record(self, record_path: str) -> Optional[np.ndarray]:
        """Load ECG record from WFDB format."""
        try:
            # Remove file extension if present
            record_name = record_path.replace('.hea', '').replace('.dat', '')
            
            # Load the record
            record = wfdb.rdrecord(record_name)
            
            # Get signal data (shape: [samples, leads])
            signal = record.p_signal
            
            if signal is None or signal.shape[1] != 12:
                return None
                
            return signal.T  # Return as [leads, samples]
            
        except Exception as e:
            print(f"Error loading record {record_path}: {e}")
            return None
            
    def create_ecg_image(self, signal: np.ndarray, record_id: str) -> np.ndarray:
        """Create standardized 12-lead ECG image from signal data."""
        fig = plt.figure(figsize=(self.img_width/self.dpi, self.img_height/self.dpi), dpi=self.dpi)
        fig.patch.set_facecolor('white')
        
        # Create 4x3 grid layout for 12 leads
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        # Time axis
        time_axis = np.linspace(0, self.duration, signal.shape[1])
        
        # Plot each lead
        for i, lead_name in enumerate(self.leads):
            row = i // 3
            col = i % 3
            
            ax = fig.add_subplot(gs[row, col])
            
            # Plot ECG signal
            ax.plot(time_axis, signal[i], 'k-', linewidth=1.2)
            
            # Add grid
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3, color='red')
            ax.grid(True, which='minor', linestyle='-', linewidth=0.25, alpha=0.2, color='red')
            
            # Set minor ticks for fine grid
            ax.set_xticks(np.arange(0, self.duration + 0.2, 0.2), minor=True)
            ax.set_yticks(np.arange(-3, 4, 0.1), minor=True)
            
            # Formatting
            ax.set_xlim(0, self.duration)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(lead_name, fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('mV', fontsize=8)
            
            # Make it look more like medical ECG paper
            ax.set_facecolor('#ffe6e6')  # Light pink background
            
        plt.tight_layout()
        
        # Convert to image array
        fig.canvas.draw()
        # Use buffer_rgba for compatibility with macOS
        buf = fig.canvas.buffer_rgba()
        img_array = np.asarray(buf)[..., :3]  # Remove alpha channel
        
        plt.close(fig)
        return img_array
        
    def create_segmentation_mask(self, signal: np.ndarray) -> np.ndarray:
        """Create segmentation mask highlighting ECG waveforms."""
        # Create a simplified mask based on signal amplitude
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        
        # This is a simplified approach - in practice, you'd want more sophisticated
        # signal processing to accurately identify waveform regions
        
        # For now, create regions where ECG signals would be plotted
        lead_height = self.img_height // 4
        lead_width = self.img_width // 3
        
        for i in range(4):  # 4 rows
            for j in range(3):  # 3 columns
                if i * 3 + j < 12:  # Only 12 leads
                    y_start = i * lead_height + 30
                    y_end = (i + 1) * lead_height - 30
                    x_start = j * lead_width + 50
                    x_end = (j + 1) * lead_width - 50
                    
                    # Create mask region for this lead
                    mask[y_start:y_end, x_start:x_end] = 255
                    
        return mask
        
    def process_record(self, record_id: int) -> bool:
        """Process a single ECG record."""
        try:
            # Get record info from dataframe
            if record_id not in self.df.index:
                return False
                
            record_info = self.df.loc[record_id]
            filename_lr = record_info['filename_lr']
            
            # Construct full path
            record_path = self.records_base_path / filename_lr
            
            # Load ECG signal
            signal = self.load_ecg_record(str(record_path).replace('_lr.dat', '_lr'))
            if signal is None:
                return False
                
            # Create ECG image
            ecg_image = self.create_ecg_image(signal, str(record_id))
            
            # Create segmentation mask  
            seg_mask = self.create_segmentation_mask(signal)
            
            # Save image
            img_path = self.output_path / f"images/ecg_{record_id:05d}.png"
            cv2.imwrite(str(img_path), cv2.cvtColor(ecg_image, cv2.COLOR_RGB2BGR))
            
            # Save mask
            mask_path = self.output_path / f"masks/ecg_{record_id:05d}.png"
            cv2.imwrite(str(mask_path), seg_mask)
            
            # Create annotation data (convert numpy types to Python types for JSON)
            annotation = {
                'record_id': int(record_id),
                'image_path': f"images/ecg_{record_id:05d}.png",
                'mask_path': f"masks/ecg_{record_id:05d}.png",
                'age': int(record_info['age']) if not pd.isna(record_info['age']) else None,
                'sex': int(record_info['sex']) if not pd.isna(record_info['sex']) else None,
                'height': float(record_info['height']) if not pd.isna(record_info['height']) else None,
                'weight': float(record_info['weight']) if not pd.isna(record_info['weight']) else None,
                'diagnostic_labels': self.get_diagnostic_labels(record_info['scp_codes']),
                'heart_axis': str(record_info['heart_axis']) if not pd.isna(record_info['heart_axis']) else None,
                'device': str(record_info['device']) if not pd.isna(record_info['device']) else None,
                'report': str(record_info['report']) if not pd.isna(record_info['report']) else None
            }
            
            # Save annotation
            ann_path = self.output_path / f"annotations/ecg_{record_id:05d}.json"
            with open(ann_path, 'w') as f:
                json.dump(annotation, f, indent=2)
                
            return True
            
        except Exception as e:
            print(f"Error processing record {record_id}: {e}")
            return False
            
    def process_dataset(self, max_records: Optional[int] = None, batch_size: int = 1000) -> None:
        """Process the entire dataset or a subset with optimized batching."""
        print(f"🚀 Processing PTB-XL dataset (FULL SCALE)...")
        print(f"Found {len(self.df)} records in database")
        
        # Get available record files from all directories
        available_records = set()
        records_path = self.records_base_path / "records100"  # 100Hz sampling
        records_500_path = self.records_base_path / "records500"  # 500Hz sampling
        
        # Check both 100Hz and 500Hz directories
        for path_name, records_dir in [("100Hz", records_path), ("500Hz", records_500_path)]:
            if records_dir.exists():
                print(f"Scanning {path_name} records directory...")
                for subdir in records_dir.iterdir():
                    if subdir.is_dir():
                        for file in subdir.iterdir():
                            if file.suffix == '.dat' and '_lr' in file.name:
                                # Extract record ID from filename
                                record_name = file.stem.replace('_lr', '')
                                try:
                                    record_id = int(record_name)
                                    available_records.add(record_id)
                                except ValueError:
                                    continue
        
        print(f"✅ Found {len(available_records)} available record files")
        
        # Filter to available records and sort for consistent processing
        valid_records = sorted(list(set(self.df.index) & available_records))
        
        # Apply record limit if specified
        if max_records:
            valid_records = valid_records[:max_records]
            print(f"📊 Limited to {max_records} records for processing")
        else:
            print(f"🎯 Processing ALL {len(valid_records)} available records")
            
        # Process in batches for memory efficiency
        total_successful = 0
        total_failed = 0
        
        for batch_start in range(0, len(valid_records), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_records))
            batch_records = valid_records[batch_start:batch_end]
            
            print(f"\n📦 Processing batch {batch_start//batch_size + 1}: records {batch_start+1}-{batch_end}")
            
            batch_successful = 0
            batch_failed = 0
            
            for record_id in tqdm(batch_records, desc=f"Batch {batch_start//batch_size + 1}"):
                if self.process_record(record_id):
                    batch_successful += 1
                    total_successful += 1
                else:
                    batch_failed += 1
                    total_failed += 1
                    
            print(f"Batch complete: {batch_successful} successful, {batch_failed} failed")
            
            # Memory cleanup after each batch
            plt.close('all')
                
        print(f"\n🎉 FULL DATASET PROCESSING COMPLETE:")
        print(f"  ✅ Successful: {total_successful}")
        print(f"  ❌ Failed: {total_failed}")
        print(f"  📈 Success Rate: {total_successful/(total_successful+total_failed)*100:.1f}%")
        
        # Create comprehensive summary metadata
        self.create_dataset_summary(total_successful, total_failed, valid_records[:total_successful])
        
    def create_dataset_summary(self, successful: int, failed: int, processed_records: List[int]) -> None:
        """Create dataset summary and splits."""
        summary = {
            'dataset': 'PTB-XL',
            'total_records_processed': len(processed_records),
            'successful': successful,
            'failed': failed,
            'image_size': [self.img_width, self.img_height],
            'leads': self.leads,
            'sampling_rate': self.sampling_rate,
            'duration': self.duration
        }
        
        # Save summary
        with open(self.output_path / 'metadata/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Create train/val/test splits (70/15/15)
        np.random.seed(42)
        successful_records = [r for r in processed_records if (self.output_path / f"images/ecg_{r:05d}.png").exists()]
        np.random.shuffle(successful_records)
        
        n_train = int(0.7 * len(successful_records))
        n_val = int(0.15 * len(successful_records))
        
        splits = {
            'train': successful_records[:n_train],
            'val': successful_records[n_train:n_train + n_val],
            'test': successful_records[n_train + n_val:]
        }
        
        # Save splits
        with open(self.output_path / 'metadata/dataset_splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
            
        print(f"Dataset splits created:")
        print(f"  Train: {len(splits['train'])} records")
        print(f"  Val: {len(splits['val'])} records")  
        print(f"  Test: {len(splits['test'])} records")


def main():
    parser = argparse.ArgumentParser(description='Process PTB-XL dataset')
    parser.add_argument('--ptbxl-path', required=True, help='Path to PTB-XL dataset directory')
    parser.add_argument('--output-path', required=True, help='Output directory for processed data')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process')
    
    args = parser.parse_args()
    
    # Check if wfdb is installed
    try:
        import wfdb
    except ImportError:
        print("Error: wfdb package not found. Install with: pip install wfdb")
        sys.exit(1)
        
    # Initialize processor
    processor = PTBXLProcessor(args.ptbxl_path, args.output_path)
    
    # Process dataset
    processor.process_dataset(args.max_records)
    
if __name__ == "__main__":
    main()