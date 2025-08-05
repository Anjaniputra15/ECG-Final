#!/usr/bin/env python3
"""
PTB-XL Integration Script
Integrates processed PTB-XL data with existing project structure.
"""

import os
import sys
import pandas as pd
import json
import shutil
from pathlib import Path
from typing import Dict, List
import argparse

class PTBXLIntegrator:
    """Integrate PTB-XL processed data with existing project structure."""
    
    def __init__(self, ptbxl_processed_path: str, project_data_path: str):
        self.ptbxl_path = Path(ptbxl_processed_path)
        self.project_path = Path(project_data_path)
        
        # Load PTB-XL metadata
        self.load_ptbxl_metadata()
        
    def load_ptbxl_metadata(self):
        """Load PTB-XL processed metadata."""
        # Load dataset summary
        with open(self.ptbxl_path / 'metadata/dataset_summary.json', 'r') as f:
            self.summary = json.load(f)
            
        # Load dataset splits
        with open(self.ptbxl_path / 'metadata/dataset_splits.json', 'r') as f:
            self.splits = json.load(f)
            
        print(f"Loaded PTB-XL data: {self.summary['successful']} records")
        print(f"Train: {len(self.splits['train'])}, Val: {len(self.splits['val'])}, Test: {len(self.splits['test'])}")
        
    def update_project_metadata(self):
        """Update project metadata CSV with PTB-XL records."""
        metadata_file = self.project_path / 'raw/metadata.csv'
        
        # Read existing metadata
        if metadata_file.exists():
            existing_df = pd.read_csv(metadata_file)
            start_idx = len(existing_df)
            print(f"Found {len(existing_df)} existing records")
        else:
            # Create new metadata structure  
            existing_df = pd.DataFrame(columns=[
                'image_id', 'image_path', 'mask_path', 'image_size', 
                'has_mask', 'quality_score', 'leads_detected', 'processing_status'
            ])
            start_idx = 0
            
        # Create new records from PTB-XL data
        new_records = []
        
        # Process all PTB-XL annotations
        annotation_files = list((self.ptbxl_path / 'annotations').glob('*.json'))
        
        for i, ann_file in enumerate(annotation_files):
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
                
            record_id = annotation['record_id']
            new_record = {
                'image_id': f'ptbxl_{record_id:05d}',
                'image_path': f'processed/ptbxl/images/ecg_{record_id:05d}.png',
                'mask_path': f'processed/ptbxl/masks/ecg_{record_id:05d}.png',
                'image_size': f"{self.summary['image_size'][0]}x{self.summary['image_size'][1]}",
                'has_mask': True,
                'quality_score': 'high',
                'leads_detected': '12-lead',
                'processing_status': 'processed'
            }
            new_records.append(new_record)
            
        # Add new records to dataframe
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Save updated metadata
        combined_df.to_csv(metadata_file, index=False)
        print(f"Updated metadata with {len(new_records)} PTB-XL records")
        
    def update_dataset_splits(self):
        """Update existing dataset splits with PTB-XL data."""
        # Update train.json
        train_file = self.project_path / 'splits/train.json'
        val_file = self.project_path / 'splits/val.json' 
        test_file = self.project_path / 'splits/test.json'
        
        # Create splits directory if it doesn't exist
        (self.project_path / 'splits').mkdir(exist_ok=True)
        
        # Convert PTB-XL record IDs to new format
        def convert_record_ids(record_list):
            return [f'ptbxl_{record_id:05d}' for record_id in record_list]
            
        # Load existing splits or create new ones
        def load_or_create_split(split_file):
            if split_file.exists():
                with open(split_file, 'r') as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'records' in data:
                        return data['records']
                    else:
                        return []
            return []
            
        # Update each split
        train_split = load_or_create_split(train_file)
        val_split = load_or_create_split(val_file)
        test_split = load_or_create_split(test_file)
        
        # Add PTB-XL records
        train_split.extend(convert_record_ids(self.splits['train']))
        val_split.extend(convert_record_ids(self.splits['val']))
        test_split.extend(convert_record_ids(self.splits['test']))
        
        # Save updated splits
        with open(train_file, 'w') as f:
            json.dump(train_split, f, indent=2)
        with open(val_file, 'w') as f:
            json.dump(val_split, f, indent=2)
        with open(test_file, 'w') as f:
            json.dump(test_split, f, indent=2)
            
        print(f"Updated splits - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
        
    def create_classification_labels(self):
        """Create classification labels file for PTB-XL data."""
        labels_file = self.project_path / 'processed/ptbxl/classification_labels.csv'
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process all annotations to create labels
        annotation_files = list((self.ptbxl_path / 'annotations').glob('*.json'))
        label_records = []
        
        # Define the 11 diagnostic categories
        diagnostic_categories = [
            'Normal', 'Myocardial_Infarction', 'ST_T_Changes', 
            'Conduction_Disturbance', 'Hypertrophy', 'Atrial_Fibrillation',
            'Atrial_Flutter', 'SVT', 'VT', 'Bradycardia', 'Tachycardia'
        ]
        
        for ann_file in annotation_files:
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
                
            record_id = annotation['record_id']
            diagnostic_labels = annotation.get('diagnostic_labels', {})
            
            # Create label record
            label_record = {
                'image_id': f'ptbxl_{record_id:05d}',
                'image_path': f'processed/ptbxl/images/ecg_{record_id:05d}.png'
            }
            
            # Add binary labels for each category
            for category in diagnostic_categories:
                label_record[category] = 1 if category in diagnostic_labels else 0
                
            # Add confidence scores if available
            for category, confidence in diagnostic_labels.items():
                if category in diagnostic_categories:
                    label_record[f'{category}_confidence'] = confidence
                    
            label_records.append(label_record)
            
        # Save labels
        labels_df = pd.DataFrame(label_records)
        labels_df.to_csv(labels_file, index=False)
        print(f"Created classification labels for {len(label_records)} records")
        
    def create_summary_report(self):
        """Create integration summary report."""
        report = {
            'integration_summary': {
                'source_dataset': 'PTB-XL',
                'records_integrated': self.summary['successful'],
                'image_format': 'PNG',
                'image_size': self.summary['image_size'],
                'leads': self.summary['leads'],
                'sampling_rate': self.summary['sampling_rate'],
                'duration_seconds': self.summary['duration']
            },
            'data_splits': {
                'train_records': len(self.splits['train']),
                'val_records': len(self.splits['val']),
                'test_records': len(self.splits['test'])
            },
            'file_structure': {
                'images': f"data/processed/ptbxl/images/ ({self.summary['successful']} files)",
                'masks': f"data/processed/ptbxl/masks/ ({self.summary['successful']} files)",
                'annotations': f"data/processed/ptbxl/annotations/ ({self.summary['successful']} files)",
                'classification_labels': "data/processed/ptbxl/classification_labels.csv"
            }
        }
        
        # Save integration report
        report_file = self.project_path / 'processed/ptbxl/integration_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print("Integration Summary:")
        print(f"  ✅ Integrated {report['integration_summary']['records_integrated']} PTB-XL records")
        print(f"  ✅ Created train/val/test splits")
        print(f"  ✅ Generated classification labels")
        print(f"  ✅ Updated project metadata")
        print(f"  📁 Data location: {self.project_path / 'processed/ptbxl'}")
        
    def integrate(self):
        """Run full integration process."""
        print("Starting PTB-XL integration...")
        
        # Create necessary directories
        (self.project_path / 'processed/ptbxl').mkdir(parents=True, exist_ok=True)
        
        # Update project metadata
        self.update_project_metadata()
        
        # Update dataset splits  
        self.update_dataset_splits()
        
        # Create classification labels
        self.create_classification_labels()
        
        # Create summary report
        self.create_summary_report()
        
        print("\n🎉 PTB-XL integration completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Integrate PTB-XL processed data')
    parser.add_argument('--ptbxl-processed', required=True, 
                        help='Path to processed PTB-XL data directory')
    parser.add_argument('--project-data', required=True,
                        help='Path to project data directory')
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = PTBXLIntegrator(args.ptbxl_processed, args.project_data)
    
    # Run integration
    integrator.integrate()
    
if __name__ == "__main__":
    main()