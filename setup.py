from setuptools import setup, find_packages

setup(
    name="ecg-pqrst-detector",
    version="0.1.0",
    description="ECG PQRST wave detection system optimized for M3 hardware",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "neurokit2>=0.2.0",
        "opencv-python>=4.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "annotation": [
            "cvat-sdk>=2.5.0",
        ],
        "deployment": [
            "onnx>=1.14.0",
            "coremltools>=6.3.0",
            "streamlit>=1.24.0",
            "fastapi>=0.100.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)