# MHARS — Multi-modal Hybrid Adaptive Response System

> Universal thermal management for IoT environments using Reinforcement Learning, transfer learning, and on-device LLM alerts.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange.svg)](https://pytorch.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.8-green.svg)](https://stable-baselines3.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is MHARS?

MHARS is a research prototype that addresses six gaps identified in a 2025 systematic literature review of 103 AIoT papers (Khadam, Davidsson & Spalazzese, *Internet of Things*, 34, 101779).

Every existing thermal management system is built for **one machine type**. A model trained on CPU data cannot be redeployed on a motor without full retraining. MHARS solves this with a **Machine Adapter** that learns the thermal profile of a new machine from fewer than 100 samples — and a **PPO transfer module** that adapts the decision policy in 50 episodes.

### Architecture: IoT → ML → AI

```
IoT Layer     Thermocouple · MLX90640 thermal camera · MPU6050 vibration · Microphone
     ↓
ML Layer      Isolation Forest · LSTM (12-step) · MobileNetV2 CNN · Autoencoder · Attention Fusion
     ↓
AI Layer      PPO decision agent · RL Router (edge/cloud) · Phi-3 Mini LLM alerts
     ↓
Output        Action · Plain language alert · Incident report · Live dashboard
```

---

## Research Gaps Addressed

| Gap (Khadam et al. 2025) | MHARS Solution |
|---|---|
| 78% of systems use numeric data only | 5-modality fusion: numeric + thermal image + audio |
| 95% use supervised learning only | Isolation Forest (unsupervised) + PPO (RL) |
| Only 15% use hybrid deployment | RL Router: dynamic edge ↔ cloud per urgency |
| Human interaction absent in 99% | Phi-3 Mini LLM generates plain language alerts |
| Single machine type per paper | Machine Adapter + cross-machine PPO transfer |
| Most systems at TRL 3–4 only | Working prototype, path to TRL 5 via hardware |

---

## Key Results

| Metric | Result |
|---|---|
| PPO reward — trained agent | **500 / 500** (maximum possible) |
| PPO reward — random baseline | −280 |
| PPO transfer: adapted (50 episodes) | **500.0** |
| PPO transfer: from scratch (50 episodes) | **13.6** |
| Transfer advantage | **36× faster convergence** |
| Machine Adapter samples needed | **100** (vs 2000 for full retrain) |
| Adaptation time | **< 0.1 seconds** |
| Edge path latency | **< 100 ms** (PPO inference) |
| LLM alert latency | 2.8–4.7 s (Phi-3 Mini, 4-bit, CPU) |

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Navin-kumar0007/MHARS_Project.git
cd MHARS_Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies (cross-platform)
pip install -r requirements-core.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set API key, ports, etc.

# 5. Install MHARS as editable package
pip install -e .
```

### Optional: GPU Acceleration
```bash
pip install -r requirements-gpu.txt
```

### Optional: Phi-3 Mini LLM (for real language alerts)

```bash
pip install llama-cpp-python
# Download model (~2.2 GB):
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
# Save to: models/Phi-3-mini-4k-instruct-q4.gguf
```

---

## Quick Start

```python
from mhars import MHARS
from mhars.schemas import SensorReading

# Initialise for CPU monitoring
system = MHARS(machine_type_id=0)   # 0=CPU 1=Motor 2=Server 3=Engine

# Process a multi-sensor reading
reading = SensorReading(temp_c=72.5, load_pct=0.85, ambient_c=24.0)
result = system.run(reading)

print(result.action)        # "fan+"
print(result.route)         # "both"
print(result.urgency)       # 0.68
print(result.alert)         # plain language alert text

# Backward compatible: single temperature float still works
result = system.run(temp_celsius=72.5)
```

### Run the demo

```bash
python demo.py
```

### Run the full-stack dashboard

```bash
# Terminal 1 — Backend API
source venv/bin/activate

# For local development (HTTP):
uvicorn api.main:app --host 127.0.0.1 --port 8000

# For production deployment (HTTPS/TLS):
# uvicorn api.main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem

# Terminal 2 — Open the static dashboard
open dashboard_web.html     # macOS
# Or open http://localhost:8000 in browser

# Terminal 3 (optional) — Next.js dashboard
cd dashboard
npm install && npm run dev
# Opens at http://localhost:3000
```

### Docker deployment (full stack)

```bash
cp .env.example .env
# Edit .env to set MHARS_API_KEY for production
docker-compose up --build
# Dashboard: http://localhost:3000
# API:       http://localhost:8000/docs
```

---

## Train all models from scratch

```bash
# Stage 1 — validate simulation environment
cd stage1_simulation && python run_stage1.py

# Stage 2 — train ML pipeline (Isolation Forest, LSTM, Autoencoder)
cd ../stage2_ml && python run_stage2.py

# Stage 3 — train PPO agent (500K timesteps, ~30 min)
cd ../stage3_ai && python run_stage3.py

# Stage 5 — run Machine Adapter experiment
cd ../stage5_adapter && python run_stage5.py
```

Or use the trainer API:

```python
from mhars import MHARSTrainer

trainer = MHARSTrainer()
trainer.train_all()           # train everything
trainer.train_ml_only()       # ML layer only
trainer.train_ppo(machine=0)  # PPO for one machine type
```

---

## Project Structure

```
MHARS_Project/
├── mhars/                      # Framework package (import this)
│   ├── __init__.py             # Public API
│   ├── core.py                 # Main MHARS class + MHARSResult
│   ├── config.py               # Central configuration
│   ├── models.py               # PyTorch model definitions
│   ├── llm.py                  # Phi-3 Mini alert generator
│   ├── trainer.py              # MHARSTrainer class
│   └── dashboard.py            # Live terminal dashboard
├── stage1_simulation/          # Gymnasium thermal environment + dataset loader
├── stage2_ml/                  # ML pipeline: IF, LSTM, Autoencoder, Fusion
├── stage3_ai/                  # PPO agent, RL Router, LLM integration
├── stage4_hardware/            # Raspberry Pi deployment (Stage 4)
├── stage5_adapter/             # Machine Adapter + cross-machine transfer
├── benchmarks/                 # Structured benchmark runner + results
├── tests/                      # Pytest test suite
├── models/                     # Trained model files (see models/README.md)
├── data/                       # NASA CMAPSS dataset
├── results/                    # Experiment outputs (JSON, CSV, plots)
├── demo.py                     # Full framework demo
└── requirements.txt
```

---

## Security & Privacy Considerations

When deploying MHARS in a real factory or edge environment, several security and privacy factors must be addressed:

1. **Edge vs. Cloud Processing**: MHARS supports a local execution model (`MHARSResult.route == "edge"`). Critical sensor data and vision/audio streams do not need to be transmitted to the cloud, preserving industrial trade secrets.
2. **Data in Transit**: If the REST/WebSocket API is exposed over a network, ensure it is wrapped in TLS (HTTPS/WSS) using a reverse proxy like Nginx or Traefik.
3. **API Authentication**: The FastAPI backend includes `X-API-Key` header authentication. Do not disable this in production (`MHARS_API_KEY` environment variable).
4. **LLM Data Sanitization**: The Phi-3 Mini agent is designed to run entirely offline on-device via `llama.cpp`. No proprietary machine IDs or schematics are sent to third-party APIs (e.g., OpenAI) unless manually configured to do so.
5. **Data Retention**: The telemetry database (`logs/mhars_events.jsonl`) should be rotated and securely archived according to your organization's data retention policies.

## Datasets

**NASA CMAPSS** — turbofan engine degradation (primary training data)
```
https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
```
Download `train_FD001.txt` and place it in `data/train_FD001.txt`.

**FLIR ADAS Thermal** — for MobileNetV2 fine-tuning (thermal image hotspot detection)
```
https://www.flir.com/oem/adas/adas-dataset-form/
```
*(Alternatively, use the free open-source FLIR Starter dataset or similar thermal samples).*

### Reproducibility & Synthetic Multi-Modal Fallback

If the CMAPSS or FLIR datasets are not present on your machine, **the system is still 100% operational**. 
MHARS includes a built-in synthetic proxy generator that dynamically calculates highly realistic placeholder vibration features (via FFT simulacra) and audio metrics (MFCC-like variance) based on the rate-of-change of the CPU/Motor temperature. This ensures that any researcher can clone the repo, run `python demo.py`, and test the full 5-modality Attention Fusion pipeline out-of-the-box without waiting for large dataset downloads.

---

## Machine Types

| ID | Machine | Safe max | Critical | Idle temp |
|---|---|---|---|---|
| 0 | CPU | 85°C | 100°C | 45°C |
| 1 | Motor | 80°C | 95°C | 40°C |
| 2 | Server | 75°C | 90°C | 35°C |
| 3 | Engine | 100°C | 115°C | 60°C |

---

## References

1. Khadam, U., Davidsson, P., & Spalazzese, R. (2025). A systematic literature review on AI in IoT systems. *Internet of Things*, 34, 101779. https://doi.org/10.1016/j.iot.2025.101779
2. Schulman, J. et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
3. Zouganeli, E. et al. (2025). Health state prediction with RL for predictive maintenance. PMC. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12833388/
4. Kalla, D., & Smith, N. (2024). Integrating AI and IoT for predictive maintenance in Industry 4.0. *Information*, 16(9), 737.
5. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation forest. IEEE ICDM.

---

## Author

**Navin Kumar** — [github.com/Navin-kumar0007](https://github.com/Navin-kumar0007)

*Research project — AIoT Systems, 2025*
