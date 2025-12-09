# Epistemic-Uncertainty-Suppression-for-Robust-Spatiotemporal-Forecasting
This repository provides the reference implementation of **CAST**, a spatiotemporal surrogate model designed for robust forecasting under distribution shift in aerospace-relevant settings.

During peer review, the goal of this repository is to enable **end-to-end execution** (install → run → obtain metrics + saved outputs) with a **small demo configuration**. Full-scale training used in the paper may require additional compute and/or non-public data (see “Datasets”).

> **Double-anonymous review note:**  
> If the journal requires double-anonymous review, please ensure that any manuscript-facing materials (PDF, supplementary files, and links shown to reviewers) use an anonymized archive/snapshot provided via the editorial office. The public repository can remain available, but the manuscript should avoid exposing identifying links during peer review.
---

## System requirements
### OS
- Tested on **Ubuntu Linux (GPU node)** and **Windows 11**.

  
### Language
- Python **3.8.20**
  


### Dependencies: 

- Python: 3.8.20
- PyTorch: torch==2.4.1+cu121
  - torchvision==0.19.1+cu121
  - torchaudio==2.4.1+cu121
- CUDA runtime (via PyTorch wheels): CUDA 12.1 components
  - nvidia-cublas-cu12==12.1.3.1
  - nvidia-cuda-runtime-cu12==12.1.105
  - nvidia-cuda-nvrtc-cu12==12.1.105
  - nvidia-cuda-cupti-cu12==12.1.105
  - nvidia-cudnn-cu12==9.1.0.70
  - nvidia-nccl-cu12==2.20.5
  - nvidia-cufft-cu12==11.0.2.54
  - nvidia-curand-cu12==10.3.2.106
  - nvidia-cusolver-cu12==11.4.5.107
  - nvidia-cusparse-cu12==12.1.0.106
  - nvidia-nvtx-cu12==12.1.105
  - nvidia-nvjitlink-cu12==12.1.105
- Core scientific stack:
  - numpy==1.24.4
  - pandas==2.0.3
  - scipy==1.10.1
  - scikit-learn==1.3.2
- Visualization:
  - matplotlib==3.7.5
  - seaborn==0.13.2
- Utilities:
  - tqdm==4.67.1
  - pyyaml==6.0.2
  - requests==2.32.3
  - packaging==25.0
- Model / attention helpers (if used by the codebase):
  - einops==0.8.1
  - axial-positional-embedding==0.3.12
  - local-attention==1.10.0
  - product-key-memory==0.2.11
  - reformer-pytorch==1.4.4
  - colt5-attention==0.11.1
- Experiment / tuning (optional, if you run the provided scripts):
  - nni==3.0

    
### Tested environment (reference)
- Ubuntu Linux (GPU node), NVIDIA RTX A6000 (48 GB VRAM)  
- NVIDIA driver 570.133.07 (CUDA driver API 12.8)  
- PyTorch 2.4.1+cu121 (CUDA 12.1 build)

### Hardware: 
- GPU: NVIDIA RTX A6000 (48 GB VRAM)
- CPU/RAM: Standard x86_64 workstation/server (not required; any modern CPU is sufficient for the demo)
- Non-standard hardware: An NVIDIA GPU is recommended for training and for reproducing the main experiments.

---

## Installation


```bash
# 1) Create and activate environment
conda create -n pytorch38 python=3.8 -y
conda activate pytorch38

# 2) Upgrade pip
python -m pip install --upgrade pip

# 3) Install PyTorch (CUDA 12.1 build)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 4) Install remaining dependencies
pip install -r requirements.txt
```

## Typical install time

- ~5–15 minutes on a normal desktop/workstation with broadband internet (Conda + pip install).

- First-time PyTorch wheel download may take longer depending on network speed.

## Quick demo (one-command run)
### Run

```bash
python run.py
```

### What the demo does
- Runs a **short training loop** (small number of iterations) on a **small demo dataset/config** to verify the pipeline end-to-end.

- Reports a compact training log and evaluation metrics.

### Expected output
- Console prints:
  - a short training log (loss)
  - evaluation metrics: **MAE, RMSE, sMAPE**
- After completion, outputs are saved to:
  - `outputs/demo_run/`
    
> > **Note:** This quick demo uses a small demo dataset and a minimal number of training iterations for one-command execution. Its purpose is to validate end-to-end setup and reproducible execution (install → run → metrics → saved outputs), so the resulting metrics are **not intended to match** the paper’s full-scale results obtained with longer training and the full experimental configuration. Exact values may also vary slightly across environments (hardware, library versions, and random seeds).


### Typical run time
- Demo: typically finishes in minutes on a modern machine (GPU faster; CPU may take longer).

## Reproducibility

- Random seeds are fixed where applicable (see code for `setup_seed(...)`).
- The demo is configured to run deterministically as much as possible given PyTorch/CUDA behavior.
- All outputs (metrics logs and artifacts) are written under `outputs/demo_run/`.

## Datasets
### Public datasets
Public datasets used in this study are available from:
- NY-Weather (NYC Open Data): https://data.cityofnewyork.us/dataset/Hyperlocal-Temperature-Monitoring/qdq3-9eqn/about_data
- Exchange Rate (Kaggle): https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset
- DMSP-Par (Kaggle): https://www.kaggle.com/datasets/saurabhshahane/dmsp-particle-precipitation-aiready-data
These datasets do not involve human subjects and do not require institutional approvals.
### Proprietary dataset
- The C919 flight-test dataset contains proprietary information and is not publicly available.
- It may be shared for bona fide academic research upon reasonable request to the corresponding author, subject to required approvals and a data-use agreement.
> Demo data: the demo expects a small CSV placed under the repository data folder (see the path used in the demo config/code). You may replace it with the officially downloaded public data or your own formatted sample.

## Code availability, versioning, and license (for checklist compliance)

- **Peer-review access:** A GitHub repository link has been provided to the editorial office for the purpose of peer review. To maintain double-anonymous review, the manuscript and reviewer-facing materials do not include identifying links; access can be mediated by the editorial office if needed.
- **Version identifier (within this archive):** `review-snapshot-v1`  
- **Reproducibility:** Please run the demo exactly as described in this README; outputs will be generated under `outputs/demo_run/`.  
- **DOI:** Not yet available. Upon acceptance, we will archive the exact release used in the manuscript (e.g., via Zenodo) and provide the DOI in the final Code Availability statement.  
- **License:** MIT License (see `LICENSE`).


## Contact

For questions during peer review, please use the journal submission system or editorial correspondence channel.
