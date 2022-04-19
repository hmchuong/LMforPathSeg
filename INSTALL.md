## Installation

### Requirements:
- PyTorch 1.8.0
- TorchVision 0.9.0

### Installation

```bash
conda create -n semseg python=3.7.3 pip
source activate semseg
pip install numpy pyyaml opencv-python scipy h5py pandas ipdb albumentations pretrainedmodels scikit-image catalyst torchmetrics
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
