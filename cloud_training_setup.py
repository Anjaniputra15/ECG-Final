#!/usr/bin/env python3
"""
🌥️ Cloud Training Setup Script
Complete infrastructure setup for 21k PTB-XL dataset training
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import requests
import yaml

class CloudTrainingSetup:
    """Complete cloud training infrastructure setup."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "config" / "cloud_training_config.yaml"
        self.setup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🌥️ ECG-LLM Cloud Training Setup")
        print("=" * 60)
        print(f"📅 Setup Time: {self.setup_timestamp}")
        print(f"📁 Project Root: {self.project_root}")
        
    def detect_environment(self):
        """Detect current environment and capabilities."""
        env_info = {
            'timestamp': self.setup_timestamp,
            'platform': sys.platform,
            'python_version': sys.version,
            'available_gpus': [],
            'cloud_provider': 'local',
            'estimated_capability': 'development'
        }
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                env_info['available_gpus'] = [
                    f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}"
                    for i in range(torch.cuda.device_count())
                ]
                env_info['estimated_capability'] = 'gpu_training'
            elif torch.backends.mps.is_available():
                env_info['available_gpus'] = ['Apple Silicon MPS']
                env_info['estimated_capability'] = 'mps_training'
        except ImportError:
            pass
        
        # Detect cloud environment
        cloud_metadata_urls = {
            'aws': 'http://169.254.169.254/latest/meta-data/',
            'gcp': 'http://metadata.google.internal/computeMetadata/v1/',
            'azure': 'http://169.254.169.254/metadata/instance'
        }
        
        for provider, url in cloud_metadata_urls.items():
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    env_info['cloud_provider'] = provider
                    env_info['estimated_capability'] = 'cloud_training'
                    break
            except:
                continue
        
        return env_info
    
    def create_directory_structure(self):
        """Create complete directory structure for cloud training."""
        directories = [
            "cloud_training",
            "cloud_training/data",
            "cloud_training/data/ptbxl_full",
            "cloud_training/data/ptbxl_full/raw",
            "cloud_training/data/ptbxl_full/processed",
            "cloud_training/data/ptbxl_full/processed/images",
            "cloud_training/data/ptbxl_full/processed/masks", 
            "cloud_training/data/ptbxl_full/processed/metadata",
            "cloud_training/models",
            "cloud_training/checkpoints",
            "cloud_training/experiments",
            "cloud_training/logs",
            "cloud_training/scripts",
            "cloud_training/configs",
            "config"  # For main config directory
        ]
        
        print("\n📁 Creating Directory Structure...")
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ {directory}")
        
        return [self.project_root / d for d in directories]
    
    def create_cloud_training_config(self, env_info):
        """Create comprehensive training configuration."""
        config = {
            'project': {
                'name': 'ECG-LLM-21K-Training',
                'version': '2.0.0',
                'setup_timestamp': self.setup_timestamp
            },
            
            'environment': env_info,
            
            'dataset': {
                'name': 'PTB-XL',
                'total_records': 21837,
                'url': 'https://physionet.org/content/ptb-xl/1.0.3/',
                'local_path': 'cloud_training/data/ptbxl_full',
                'processed_image_size': [224, 224],
                'batch_processing_size': 1000
            },
            
            'training': {
                'models': {
                    'vision_transformer': {
                        'architecture': 'ViT-B/16',
                        'parameters': 86099723,
                        'input_size': [224, 224],
                        'batch_size': 32,
                        'epochs': 30,
                        'learning_rate': 0.0001
                    },
                    'gpt_clinical': {
                        'architecture': 'GPT-OSS-20B',
                        'context_length': 4096,
                        'batch_size': 8,
                        'epochs': 20,
                        'learning_rate': 0.00001
                    }
                },
                'optimization': {
                    'mixed_precision': True,
                    'gradient_accumulation_steps': 4,
                    'checkpoint_every': 1000,
                    'validation_every': 2000
                }
            },
            
            'cloud': {
                'recommended_instances': {
                    'aws': {
                        'instance_type': 'p3.2xlarge',
                        'gpu': 'V100 16GB',
                        'cost_per_hour': 3.06,
                        'estimated_total_cost': 2200
                    },
                    'gcp': {
                        'instance_type': 'n1-highmem-8',
                        'gpu': 'V100',
                        'cost_per_hour': 2.50,
                        'estimated_total_cost': 1800
                    },
                    'azure': {
                        'instance_type': 'NC12s_v3',
                        'gpu': 'V100',
                        'cost_per_hour': 3.17,
                        'estimated_total_cost': 2300
                    }
                },
                'storage_requirements': {
                    'raw_data': '20GB',
                    'processed_images': '150GB', 
                    'models_checkpoints': '50GB',
                    'total_minimum': '250GB'
                }
            }
        }
        
        # Save configuration
        self.config_file.parent.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n⚙️ Configuration saved: {self.config_file}")
        return config
    
    def install_cloud_dependencies(self):
        """Install all required packages for cloud training."""
        print("\n📦 Installing Cloud Training Dependencies...")
        
        dependencies = [
            # Core ML frameworks
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "transformers>=4.30.0",
            
            # Medical data processing
            "wfdb>=4.1.0",  # For PTB-XL WFDB format
            "scipy>=1.10.0",
            "scikit-image>=0.20.0",
            
            # Cloud and data management
            "boto3",  # AWS
            "google-cloud-storage",  # GCP
            "azure-storage-blob",  # Azure
            "kaggle",  # Kaggle datasets
            
            # Monitoring and visualization
            "wandb",  # Weights & Biases for experiment tracking
            "tensorboard>=2.13.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            
            # Performance optimization
            "accelerate>=0.20.0",  # Hugging Face optimization
            "bitsandbytes",  # 8-bit training
            "datasets>=2.10.0",  # Dataset management
            
            # Utilities
            "tqdm>=4.65.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
            "requests>=2.31.0"
        ]
        
        # Create requirements file
        requirements_file = self.project_root / "cloud_training" / "requirements.txt"
        with open(requirements_file, 'w') as f:
            for dep in dependencies:
                f.write(f"{dep}\n")
        
        print(f"  📝 Requirements file created: {requirements_file}")
        
        # Try to install
        try:
            print("  🔄 Installing packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "-r", str(requirements_file)
            ], check=True, capture_output=True)
            print("  ✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Some packages may need manual installation")
            print(f"  💡 Run: pip install -r {requirements_file}")
        
        return requirements_file
    
    def create_cloud_startup_scripts(self):
        """Create startup scripts for different cloud providers."""
        print("\n🚀 Creating Cloud Startup Scripts...")
        
        scripts_dir = self.project_root / "cloud_training" / "scripts"
        
        # AWS startup script
        aws_script = """#!/bin/bash
# AWS EC2 Startup Script for ECG-LLM Training

echo "🚀 Starting ECG-LLM Cloud Training Setup on AWS"

# Update system
sudo apt-get update -y
sudo apt-get install -y python3-pip git htop nvtop

# Install NVIDIA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
fi

# Clone project (replace with your repo)
# git clone https://github.com/your-repo/ecg-llm.git
# cd ecg-llm

# Install dependencies
pip3 install -r cloud_training/requirements.txt

# Download PTB-XL dataset
python3 cloud_training/scripts/download_ptbxl_full.py

# Start training
python3 cloud_training/scripts/train_cloud_21k.py

echo "✅ Setup complete! Training started."
"""
        
        # Google Cloud startup script  
        gcp_script = """#!/bin/bash
# Google Cloud VM Startup Script for ECG-LLM Training

echo "🚀 Starting ECG-LLM Cloud Training Setup on GCP"

# Install CUDA if needed
if ! command -v nvcc &> /dev/null; then
    curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda
fi

# Install Python dependencies
pip3 install -r cloud_training/requirements.txt

# Setup monitoring
pip3 install wandb
wandb login

# Start training with monitoring
python3 cloud_training/scripts/train_cloud_21k.py --use-wandb

echo "✅ GCP Setup complete!"
"""
        
        # Save startup scripts
        with open(scripts_dir / "aws_startup.sh", 'w') as f:
            f.write(aws_script)
        
        with open(scripts_dir / "gcp_startup.sh", 'w') as f:
            f.write(gcp_script)
        
        # Make executable
        os.chmod(scripts_dir / "aws_startup.sh", 0o755)
        os.chmod(scripts_dir / "gcp_startup.sh", 0o755)
        
        print(f"  ✅ AWS startup script: {scripts_dir}/aws_startup.sh")
        print(f"  ✅ GCP startup script: {scripts_dir}/gcp_startup.sh")
        
        return scripts_dir
    
    def create_monitoring_dashboard(self):
        """Create monitoring and logging setup."""
        print("\n📊 Setting up Monitoring Dashboard...")
        
        monitoring_script = """#!/usr/bin/env python3
'''
Training Monitoring Dashboard for ECG-LLM Cloud Training
'''

import time
import psutil
import GPUtil
from datetime import datetime
import json
from pathlib import Path

class TrainingMonitor:
    def __init__(self, log_dir="cloud_training/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"training_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log_system_stats(self):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
        }
        
        # GPU stats if available
        try:
            gpus = GPUtil.getGPUs()
            stats['gpus'] = []
            for gpu in gpus:
                stats['gpus'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                    'utilization': gpu.load * 100
                })
        except:
            pass
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(stats) + '\\n')
        
        return stats

if __name__ == "__main__":
    monitor = TrainingMonitor()
    while True:
        stats = monitor.log_system_stats()
        print(f"📊 {datetime.now().strftime('%H:%M:%S')} - CPU: {stats['cpu_percent']:.1f}% | RAM: {stats['memory_percent']:.1f}%")
        time.sleep(30)  # Log every 30 seconds
"""
        
        with open(self.project_root / "cloud_training" / "scripts" / "monitor.py", 'w') as f:
            f.write(monitoring_script)
        
        print("  ✅ Monitoring dashboard created")
        return True
    
    def run_setup(self):
        """Run complete setup process."""
        print("\n🔧 Running Complete Cloud Training Setup...")
        
        # 1. Environment detection
        env_info = self.detect_environment()
        print(f"\n🌍 Environment: {env_info['cloud_provider']} ({env_info['estimated_capability']})")
        
        # 2. Create directory structure
        directories = self.create_directory_structure()
        
        # 3. Create configuration
        config = self.create_cloud_training_config(env_info)
        
        # 4. Install dependencies
        requirements_file = self.install_cloud_dependencies()
        
        # 5. Create startup scripts
        scripts_dir = self.create_cloud_startup_scripts()
        
        # 6. Setup monitoring
        self.create_monitoring_dashboard()
        
        # Summary
        print(f"\n🎉 CLOUD TRAINING SETUP COMPLETE!")
        print("=" * 60)
        print(f"📁 Project Structure: Ready")
        print(f"⚙️ Configuration: {self.config_file}")
        print(f"📦 Dependencies: {requirements_file}")
        print(f"🚀 Scripts: {scripts_dir}")
        print(f"\n🎯 NEXT STEPS:")
        print("1. Choose cloud provider (AWS/GCP/Azure)")
        print("2. Launch instance with startup script")
        print("3. Run: python3 cloud_training/scripts/download_ptbxl_full.py")
        print("4. Run: python3 cloud_training/scripts/train_cloud_21k.py")
        print("\n💡 Estimated Training Cost: $1,800 - $2,300")
        print("💡 Estimated Training Time: 2-3 weeks")
        
        return {
            'config': config,
            'directories': directories,
            'requirements': requirements_file,
            'scripts': scripts_dir
        }

def main():
    """Main setup function."""
    setup = CloudTrainingSetup()
    return setup.run_setup()

if __name__ == "__main__":
    main()