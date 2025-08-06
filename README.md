# SSM Multimodal MVP

A full-stack MVP for multimodal (audio + text) sequence modeling and inference, featuring model training, ONNX export, gRPC-based serving, a Next.js frontend, and monitoring with Prometheus and Grafana.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Training Pipeline](#training-pipeline)
- [Model Architecture](#model-architecture)
- [Inference & Serving](#inference--serving)
- [Frontend](#frontend)
- [Monitoring](#monitoring)
- [Deployment (Docker Compose)](#deployment-docker-compose)
- [Directory Overview](#directory-overview)
- [Requirements](#requirements)
- [License](#license)

---

## Project Structure

```
ssm-multimodal-mvp/
├── data/                # Processed datasets (e.g., synthetic_processed.pt)
├── deployment/          # Dockerfiles, docker-compose, monitoring configs
├── frontend/            # Next.js client app
├── models/              # Model checkpoints and ONNX exports
├── monitoring/          # (empty or for future monitoring scripts)
├── protoc-25.3/         # Protobuf tools
├── serving/             # Go gRPC server and proto definitions
├── training/            # Model training, data prep, and evaluation scripts
├── requirements.txt     # Python dependencies
└── ...
```

---

## Features

- **Multimodal Model**: Audio (mel spectrogram) + text input, trained with Mamba SSM blocks.
- **Training Pipeline**: Data preparation, training, evaluation, and ONNX export.
- **gRPC Inference Server**: Go-based server loads ONNX model and exposes gRPC API.
- **gRPC-Web Proxy**: Bridges browser/frontend to gRPC backend.
- **Next.js Frontend**: User interface for inference.
- **Monitoring**: Prometheus metrics and Grafana dashboards.
- **Dockerized**: All components containerized for easy deployment.

---

## Setup & Installation

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + drivers (for training)
- Python 3.8+ (for local training)
- Node.js 18+ (for frontend dev)

### Quick Start (All-in-One)

```bash
cd deployment
docker-compose up --build
```

- Frontend: [http://localhost:3000](http://localhost:3000)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3001](http://localhost:3001) (admin/admin)

---

## Training Pipeline

- **Data Preparation**: `training/prepare_data.py` downloads and processes LibriSpeech audio, extracts log-mel spectrograms, and tokenizes text using GPT-2 tokenizer.
- **Model Training**: `training/train_model.py` trains the `MultimodalMambaModel` (see below) on the processed data, logs metrics to TensorBoard, and saves checkpoints.
- **Quick Training**: `training/quick_train.py` provides a minimal example for rapid prototyping.
- **ONNX Export**: Export trained models to ONNX format for serving.

### Example: Run Training

```bash
cd training
python prepare_data.py
python train_model.py
python export_onnx.py
```

---

## Model Architecture

Defined in `training/multimodal_mamba.py`:

- **Audio Encoder**: 1D Conv layers project mel spectrograms to model dimension.
- **Text Embedding**: Standard embedding layer for tokenized text.
- **Mamba SSM Blocks**: Two stacked Mamba blocks for sequence modeling.
- **Output Projection**: Linear layer to vocabulary size.

```python
class MultimodalMambaModel(nn.Module):
    ...
    def forward(self, audio_features, text_tokens):
        # Encode audio and text, concatenate, pass through Mamba blocks, project to vocab
```

---

## Inference & Serving

- **Go gRPC Server** (`serving/go-server/`): Loads ONNX model, exposes gRPC API for inference and health checks, and serves Prometheus metrics.
- **gRPC-Web Proxy**: Converts HTTP requests from the frontend to gRPC calls.
- **ONNX Models**: Place exported models in `models/onnx/`.

### API Endpoints

- `/api/inference` (POST): Accepts audio data and text prompt, returns generated text and confidence.
- `/api/health` (GET): Health check endpoint.

---

## Frontend

- **Framework**: Next.js (TypeScript, Tailwind CSS)
- **Location**: `frontend/nextjs-client/`
- **Dev Start**:

```bash
cd frontend/nextjs-client
npm install
npm run dev
```

- **Build & Serve (Docker)**: Handled by `Dockerfile.frontend` and docker-compose.

---

## Monitoring

- **Prometheus**: Scrapes metrics from inference server.
- **Grafana**: Pre-configured dashboards for inference latency, request counts, etc.
- **Config**: See `deployment/prometheus.yml` and `deployment/grafana/`.

---

## Deployment (Docker Compose)

All services are orchestrated via `deployment/docker-compose.yml`:

- `inference-server`: gRPC server (Go, ONNX)
- `grpc-web-proxy`: HTTP-to-gRPC bridge
- `frontend`: Next.js app
- `training`: Model training (GPU required)
- `prometheus`: Metrics collection
- `grafana`: Visualization

### Example

```bash
cd deployment
docker-compose up --build
```

---

## Directory Overview

- **data/**: Processed datasets (e.g., `synthetic_processed.pt`)
- **models/onnx/**: Exported ONNX models for serving
- **models/checkpoints/**: PyTorch model checkpoints
- **training/**: All training, data prep, and evaluation scripts
- **serving/go-server/**: Go gRPC server and proxy
- **frontend/nextjs-client/**: Next.js frontend app
- **deployment/**: Dockerfiles, docker-compose, monitoring configs

---

## Requirements

See `requirements.txt` for Python dependencies, including:

- torch, torchvision, torchaudio
- mamba-ssm
- transformers, datasets
- librosa, soundfile
- onnx, onnxruntime
- tensorboard, prometheus-client
- jupyter, ipywidgets

---

## License

[MIT License](LICENSE) (or specify your license here)

---

**For more details, see the code in each subdirectory and the comments in the scripts.**
