# Epistemic-Uncertainty-Suppression-for-Robust-Spatiotemporal-Forecasting
This repository contains the source code for the spatiotemporal surrogate model (CAST) developed for robust forecasting in aerospace systems. The code will be made publicly available to support further research and development in the field of digital airworthiness validation and spatiotemporal forecasting. The code will be made open-source upon acceptance. Contributions to the project are welcome.
##System requirements
###OS: Windows 11

###Language: Python 3.8

###Dependencies: 

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

###Tested on
Ubuntu Linux (GPU node) with NVIDIA RTX A6000 (48 GB VRAM), NVIDIA driver 570.133.07 (CUDA driver API 12.8), and PyTorch 2.4.1+cu121.

###Hardware: 
GPU: NVIDIA RTX A6000 (48 GB VRAM)

CPU/RAM: Standard x86_64 workstation/server (not required; any modern CPU is sufficient for the demo)

Non-standard hardware: An NVIDIA GPU is recommended for training and for reproducing the main experiments.

#Installation
##System requirements

OS: Ubuntu Linux

Python: 3.8.20

GPU (recommended): NVIDIA RTX A6000 (48 GB VRAM)

Driver: NVIDIA 570.133.07 (CUDA driver API 12.8)

PyTorch: 2.4.1+cu121 (CUDA 12.1 build)

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

##Datasets
Public datasets used in this study are available as follows: the NY-Weather dataset from New York City Open Data (https://data.cityofnewyork.us/dataset/Hyperlocal-Temperature-Monitoring/qdq3-9eqn/about_data). Detailed metadata and variable descriptions are provided on the dataset landing page. The Exchange Rate dataset is available at (https://www.kaggle.com/datasets/wentixiaogege/time-series-dataset). The DMSP-Par dataset is available at (https://www.kaggle.com/datasets/saurabhshahane/dmsp-particle-precipitation-aiready-data). These datasets do not involve human subjects and do not require institutional approvals. The C919 flight-test dataset contains proprietary information and is not publicly available; it may be shared by the authors for bona fide academic research upon reasonable request to the corresponding author, subject to any required approvals and a data-use agreement. 
