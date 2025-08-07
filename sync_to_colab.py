#!/usr/bin/env python3
"""
Sync local Windsurf development to Google Colab for GPU training
"""

import os
import shutil
from pathlib import Path
import zipfile
import json
from datetime import datetime

class WindsurfColabSync:
    def __init__(self, local_project_path, google_drive_path):
        self.local_path = Path(local_project_path)
        self.drive_path = Path(google_drive_path)
        self.sync_manifest = self.local_path / ".colab_sync_manifest.json"
        
    def create_training_package(self):
        """Create a training package for Colab"""
        print("📦 Creating training package for Colab...")
        
        # Files to sync
        essential_files = [
            "bootstrap_trainer.py",
            "training/advanced_trainer.py",
            "models/backbones/*.py",
            "enhanced_preprocessing.py",
            "requirements.txt"
        ]
        
        # Create package directory
        package_dir = self.local_path / "colab_package"
        package_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        for file_pattern in essential_files:
            if "*" in file_pattern:
                # Handle wildcards
                from glob import glob
                files = glob(str(self.local_path / file_pattern))
                for file_path in files:
                    rel_path = Path(file_path).relative_to(self.local_path)
                    dest_path = package_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    print(f"✅ Copied: {rel_path}")
            else:
                src_file = self.local_path / file_pattern
                if src_file.exists():
                    dest_file = package_dir / file_pattern
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)
                    print(f"✅ Copied: {file_pattern}")
        
        # Create training config for Colab
        colab_config = {
            "device": "cuda",
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 50,
            "use_wandb": False,  # Disable for Colab
            "save_every_n_epochs": 10,
            "sync_timestamp": datetime.now().isoformat()
        }
        
        with open(package_dir / "colab_config.json", "w") as f:
            json.dump(colab_config, f, indent=2)
        
        # Zip the package
        zip_path = self.local_path / "ecg_training_package.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir)
                    zipf.write(file_path, arcname)
        
        print(f"📦 Training package created: {zip_path}")
        
        # Clean up temporary directory
        shutil.rmtree(package_dir)
        
        return zip_path
    
    def create_colab_runner(self):
        """Create a Colab notebook to run your training"""
        colab_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# ECG-LLM Training from Windsurf\n", "Auto-generated training runner"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Setup environment\n",
                        "!nvidia-smi\n",
                        "import torch\n",
                        "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                        "print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
                    ]
                },
                {
                    "cell_type": "code", 
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Install dependencies\n",
                        "!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n",
                        "!pip install wfdb neurokit2 pandas numpy matplotlib opencv-python scikit-learn tqdm\n",
                        "!pip install transformers timm efficientnet-pytorch"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None, 
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Mount Google Drive\n",
                        "from google.colab import drive\n",
                        "drive.mount('/content/drive')\n",
                        "\n",
                        "# Extract training package\n",
                        "import zipfile\n",
                        "import os\n",
                        "\n",
                        "package_path = '/content/drive/MyDrive/ECG_LLM_Project/ecg_training_package.zip'\n",
                        "if os.path.exists(package_path):\n",
                        "    with zipfile.ZipFile(package_path, 'r') as zipf:\n",
                        "        zipf.extractall('/content/ecg_training')\n",
                        "    os.chdir('/content/ecg_training')\n",
                        "    print('✅ Training package extracted')\n",
                        "else:\n",
                        "    print('❌ Training package not found. Please upload it to Google Drive.')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {}, 
                    "outputs": [],
                    "source": [
                        "# Download PTB-XL dataset (run once)\n",
                        "import urllib.request\n",
                        "import zipfile\n",
                        "\n",
                        "print('📥 Downloading PTB-XL dataset...')\n",
                        "url = 'https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip'\n",
                        "\n",
                        "if not os.path.exists('data'):\n",
                        "    urllib.request.urlretrieve(url, 'ptb-xl.zip')\n",
                        "    with zipfile.ZipFile('ptb-xl.zip', 'r') as zipf:\n",
                        "        zipf.extractall('data')\n",
                        "    os.remove('ptb-xl.zip')\n",
                        "    print('✅ Dataset downloaded and extracted')\n",
                        "else:\n",
                        "    print('✅ Dataset already exists')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Run your training code\n",
                        "print('🚀 Starting ECG-LLM training...')\n",
                        "\n",
                        "# Load Colab config\n",
                        "import json\n",
                        "with open('colab_config.json', 'r') as f:\n",
                        "    config = json.load(f)\n",
                        "\n",
                        "print(f'Config: {config}')\n",
                        "\n",
                        "# Execute your training script\n",
                        "!python bootstrap_trainer.py --device cuda --batch-size 16 --epochs 50"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Save results back to Google Drive\n",
                        "import shutil\n",
                        "\n",
                        "results_dir = '/content/drive/MyDrive/ECG_LLM_Project/training_results'\n",
                        "os.makedirs(results_dir, exist_ok=True)\n",
                        "\n",
                        "# Copy model files\n",
                        "model_files = ['best_model.pth', 'latest_checkpoint.pth']\n",
                        "for model_file in model_files:\n",
                        "    if os.path.exists(model_file):\n",
                        "        shutil.copy2(model_file, f'{results_dir}/{model_file}')\n",
                        "        print(f'✅ Saved {model_file} to Drive')\n",
                        "\n",
                        "# Copy experiment results\n",
                        "if os.path.exists('experiments'):\n",
                        "    shutil.copytree('experiments', f'{results_dir}/experiments', dirs_exist_ok=True)\n",
                        "    print('✅ Saved experiments to Drive')\n",
                        "\n",
                        "print('🎉 Training complete! Results saved to Google Drive.')"
                    ]
                }
            ],
            "metadata": {
                "colab": {"provenance": [], "gpuType": "T4", "machine_shape": "hm"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "accelerator": "GPU"
            },
            "nbformat": 4,
            "nbformat_minor": 0
        }
        
        notebook_path = self.local_path / "ECG_Training_Runner.ipynb"
        with open(notebook_path, 'w') as f:
            json.dump(colab_notebook, f, indent=2)
        
        print(f"📓 Colab runner created: {notebook_path}")
        return notebook_path

def main():
    """Main sync function"""
    # Adjust these paths for your setup
    local_project = "/Users/baba/Downloads/ECG-LLM-main/venv/etc/jupyter/nbconfig/notebook.d/ECGfinal/ecg-pqrst-detector"
    google_drive = "~/Google Drive/ECG_LLM_Project"  # Adjust for your Google Drive path
    
    syncer = WindsurfColabSync(local_project, google_drive)
    
    # Create training package
    package_path = syncer.create_training_package()
    
    # Create Colab runner notebook
    notebook_path = syncer.create_colab_runner()
    
    print("\n🎯 Next steps:")
    print(f"1. Upload {package_path} to Google Drive/ECG_LLM_Project/")
    print(f"2. Upload {notebook_path} to Google Colab")
    print("3. Run the notebook cells in Colab to start training!")
    print("\n✨ Your Windsurf code will run on Colab GPU!")

if __name__ == "__main__":
    main()