<div align="center">
<h1>TopoLiDM: Topology-Aware LiDAR Diffusion Models for Interpretable and Realistic LiDAR Point Cloud Generation</h1>

<p>
  <a href="https://arxiv.org/abs/2507.22454"><img src="https://img.shields.io/badge/arXiv-2507.22454-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/IROS-2025-green.svg" alt="IROS 2025">
</p>

<p>
  <b>Jiuming Liu</b><sup>†1</sup>,
  <b>Zheng Huang</b><sup>†1</sup>,
  Mengmeng Liu<sup>2</sup>,
  Tianchen Deng<sup>1</sup>,
  Francesco Nex<sup>2</sup>,
  Hao Cheng<sup>2</sup>,
  Hesheng Wang<sup>*1</sup>
</p>

<p>
  <sup>1</sup>Shanghai Jiao Tong University &nbsp;|&nbsp;
  <sup>2</sup>University of Twente
  <br>
  <sup>†</sup>Equal contribution &nbsp;|&nbsp; <sup>*</sup>Corresponding author
</p>

</div>


## Installation

> **Note:** Run the following commands one by one rather than as a single script, as some steps (e.g. conda activate) require shell re-sourcing.

### Step 1 — Install Rust (required by some packages)

```bash
# Optional: set Tsinghua mirror for faster downloads in China
export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
export RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

### Step 2 — Create the conda environment

```bash
conda create -n ld python=3.10.11 -y
conda activate ld
```

### Step 3 — Install PyTorch (CUDA 11.8)

```bash
pip install --upgrade pip
pip install setuptools==69.5.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

### Step 4 — Install core dependencies

```bash
pip install \
    torchmetrics==0.5.0 pytorch-lightning==1.4.2 omegaconf==2.1.1 \
    einops==0.3.0 transformers==4.36.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 \
    opencv-python kornia==0.7.0 wandb more_itertools gdown \
    numpy==1.26 easydict joblib scipy

pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
```

### Step 5 — Install TopologyLayer and Dionysus (required for TopoLiDM)

```bash
pip install -e git+https://github.com/bruel-gabrielsson/TopologyLayer.git --no-build-isolation

apt-get update && apt-get install -y libboost-all-dev
conda install -c conda-forge dionysus
```

After installing TopologyLayer, patch a NumPy API deprecation inside the library:

```bash
# Replace deprecated np.int with int throughout TopologyLayer source
cd path/to/TopologyLayer   # typically inside your pip src/ folder
grep -rl "np.int" . | xargs sed -i 's/np.int/int/g'
```

### Step 6 — Install torchsparse (optional, required for evaluation)

```bash
apt-get install -y libsparsehash-dev

# Clear LD_LIBRARY_PATH to avoid library conflicts (see Troubleshooting below)
LD_LIBRARY_PATH="" pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 \
    --no-build-isolation
```

### Step 7 — Prepare dataset directory and pretrained weights

```bash
mkdir -p dataset/        # symlink or place KITTI-360 data here
mkdir -p pretrained_weights/   # place pretrained model checkpoints here
```

See `lidm/eval/README.md` for details on preparing evaluation checkpoints. If a checkpoint is a `.zip` file, remove the `.zip` suffix before use.


## Pretrained Weights

We release pretrained weights on Hugging Face: **[hhzz0505/TopoLiDM](https://huggingface.co/hhzz0505/TopoLiDM)**

| Model | Description | File |
|-------|-------------|------|
| Topological Autoencoder | Trained on KITTI-360 | `autoencoder/model.ckpt` |
| Diffusion Model | Unconditional LiDAR generation on KITTI-360 | `diffusion/model.ckpt` |

Download with Python:
```python
from huggingface_hub import hf_hub_download

# Autoencoder
hf_hub_download(repo_id="hhzz0505/TopoLiDM", filename="autoencoder/model.ckpt", local_dir="pretrained_weights/")

# Diffusion model
hf_hub_download(repo_id="hhzz0505/TopoLiDM", filename="diffusion/model.ckpt", local_dir="pretrained_weights/")
```

Or via CLI:
```bash
huggingface-cli download hhzz0505/TopoLiDM --local-dir pretrained_weights/
```


## Quick Start

### Training the Topological Autoencoder

```bash
# Train on KITTI with topological regularization (multi-GPU)
python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4_topo.yaml -t --gpus 0,1,2,3

# Resume from a checkpoint  (note: -n and -r are mutually exclusive)
python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4_topo.yaml -t --gpus 0 \
    -r path/to/checkpoint

# Single-GPU debug mode
python main.py -b configs/autoencoder/kitti/autoencoder_c2_p4_topo.yaml -t --gpus 0 -d
```

### Sampling and Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sample.py \
    -d kitti -r models/lidm/kitti/uncond/model.ckpt -n 2000 --eval
```


## Project Structure

```
topolidm/
├── main.py                                    # Main training entry point
├── configs/
│   └── autoencoder/kitti/
│       └── autoencoder_c2_p4_topo.yaml        # Topological autoencoder config
├── lidm/
│   ├── models/
│   │   ├── autoencoder.py                     # Standard LiDM autoencoder
│   │   └── autoencoder_topo.py                # Topological autoencoder
│   └── modules/diffusion/
│       ├── model_lidm.py                      # LiDM UNet components
│       └── model_topolidm.py                  # Graph encoder + decoder
```


## Troubleshooting

### torchsparse Installation Issues

Two errors are commonly encountered when installing torchsparse in a conda environment.

---

#### Error 1: `undefined symbol: ucs_config_doc_nop`

```
ImportError: /opt/hpcx/ucc/lib/libucc.so.1: undefined symbol: ucs_config_doc_nop
```

**Cause:** The system-level `LD_LIBRARY_PATH` includes `/opt/hpcx/ucc/lib/`, whose `libucc.so.1` conflicts with the library expected by PyTorch. Python loads the system library instead of the conda environment's library at startup.

**Fix:**
```bash
# Clear LD_LIBRARY_PATH before running pip
LD_LIBRARY_PATH="" pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 \
    --no-build-isolation

# Alternative: use env -i to fully isolate the subprocess environment
env -i PATH="$PATH" HOME="$HOME" pip install \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 --no-build-isolation
```

---

#### Error 2: `CUDA version mismatch`

```
RuntimeError: The detected CUDA version (12.4) mismatches the version that was used
to compile PyTorch (11.8).
```

**Cause:** The `nvcc` visible to pip comes from a different CUDA toolkit than the one used to build the installed PyTorch. PyTorch's `cpp_extension.py` checks for this mismatch.

**Fix:** Reinstall PyTorch pinned to your system's CUDA version, then reinstall torchsparse:
```bash
pip uninstall -y torch
pip install --no-cache-dir torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

LD_LIBRARY_PATH="" pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 \
    --no-build-isolation
```

---

#### Root Cause: LD_LIBRARY_PATH priority

The underlying problem is that conda's environment variables (set via `conda env config vars set`) only take effect when conda **explicitly activates** the environment in the current shell. A `LD_LIBRARY_PATH` set in `~/.bashrc` or by a system init script takes precedence over conda's settings in any shell that inherits it without re-activating conda.

Diagnosis commands:
```bash
# Check which library paths are currently active
echo $LD_LIBRARY_PATH

# Check that PyTorch is loading from the right place
unset LD_LIBRARY_PATH && python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Verify torchsparse after a clean install
unset LD_LIBRARY_PATH && python -c "import torchsparse; print(torchsparse.__version__)"
```



---

#### Common Errors Quick Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `undefined symbol: ucs_config_doc_nop` | `LD_LIBRARY_PATH` points to wrong `libucc` | Run with `LD_LIBRARY_PATH=""` |
| `CUDA version mismatch` | `nvcc` version differs from PyTorch's CUDA | Reinstall matching PyTorch version |
| `Unknown CUDA arch (8.7)` | GPU arch not supported by this build | Set `TORCH_CUDA_ARCH_LIST="8.0;8.6"` |
| `No module named 'pkg_resources'` | New setuptools removed `pkg_resources` | `pip install setuptools==69.5.1` |
| `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent` | PyTorch version conflicts with system libs | Reinstall CUDA-matched PyTorch |


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2025topolidm,
  title={TopoLiDM: Topology-Aware LiDAR Diffusion Models for Interpretable and Realistic LiDAR Point Cloud Generation},
  author={Liu, Jiuming and Huang, Zheng and Liu, Mengmeng and Deng, Tianchen and Nex, Francesco and Cheng, Hao and Wang, Hesheng},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={8180--8186},
  year={2025},
  organization={IEEE}
}
```


## Acknowledgements

- The diffusion model framework builds heavily on [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
- Topological regularization concepts from [GLiDR](https://github.com/GLiDR-CVPR2024/GLiDR)
- LiDAR data processing pipeline from [LiDM](https://github.com/hancyran/LiDAR-Diffusion)
