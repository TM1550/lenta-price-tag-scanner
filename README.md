# Design_ML_system

## Setup

```bash
# Clone repository
git clone <your-repo-url>
cd Design_ML_system

# Install poetry dependencies
poetry install

# Install pre-commit hooks
pre-commit install

# Download data with DVC
dvc pull
```

## Train

```bash
# Run training with default config
poetry run python scripts/train.py

# Or use hydra to override parameters
poetry run python scripts/train.py train.epochs=50 train.batch=32

# Or use fire CLI
poetry run price-scanner train
```

## Production Preparation

```bash
# Export model to ONNX and TensorRT
poetry run price-scanner export
```

## Infer

```bash
# Run inference on image
poetry run price-scanner infer source=path/to/image.jpg

# Run inference on video
poetry run price-scanner infer source=path/to/video.mp4

# Run inference on camera
poetry run price-scanner infer source=0
```
