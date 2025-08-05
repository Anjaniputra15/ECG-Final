# ECG PQRST Detector

A comprehensive ECG analysis system for PQRST wave detection optimized for M3 hardware.

## Project Structure

See the directory structure above for detailed organization of components.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Setup M3 environment: `./scripts/setup_m3_env.sh`
3. Prepare your dataset: `python scripts/prepare_dataset.py`
4. Train the model: `python scripts/train_model.py`
5. Evaluate results: `python scripts/evaluate.py`

## Documentation

- [Installation Guide](docs/installation.md)
- [Data Preparation](docs/data_preparation.md)
- [Annotation Workflow](docs/annotation_guide.md)
- [Training Guide](docs/training_guide.md)
- [API Reference](docs/api_reference.md)