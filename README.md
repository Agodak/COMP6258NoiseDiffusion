# NoiseDiffusion Reproduction

This project is a reproduction of the paper:

**"NoiseDiffusion: Correcting Noise for Image Interpolation with Diffusion Models Beyond Spherical Linear Interpolation"**,  
presented at ICLR 2024.

<img src="https://github.com/user-attachments/assets/ca102c49-07a2-4046-89d7-4b3d4b2ed291" title="Image Taken From Paper">


Note: The image is taken from the paper.


We reimplemented the core interpolation strategies using diffusion models and performed empirical studies to evaluate their quality and fidelity. Our work focuses on the *NoiseDiffusion* algorithm as described in the original paper.

📄 Original Paper: [NoiseDiffusion (ICLR 2024)](https://openreview.net/forum?id=6O3Q6AFUTu)
🔗 Official Code: [NoiseDiffusion](https://github.com/Tranquil1ty/NoiseDiffusion)

---

## Overview

- Implements and evaluates different interpolation techniques:
  - Spherical Linear Interpolation (Slerp)
  - Noise Injection (à la SDEdit)
  - NoiseDiffusion (proposed method)

---

## Project Structure

```bash
COMP6258NoiseDiffusion-main/
│
├── cldm/                        # ControlNet related modules
│
│
├── ldm/                         # Latent diffusion model modules
│   
├── controlnet_boundary_mu_experiments.py   # Experiments using ControlNet & mu/clip values
├── ldm_parameter_experiments.py            # Latent diffusion model interpolation experiments
├── parameter_experiments.py                # Runs multiple parameter experiments
├── source_ldm_experiments.py               # Source image interpolation (e.g. LSUN, SD)
├── environment.yaml                        # Conda environment dependencies
├── README.md                               
```

## Setup

### 1. Environment

We recommend using `conda`:

```bash
conda env create -f environment.yaml
conda activate control
```


## Running Code

### 1. Interpolation Using LDM-HuggingFace

```bash
python ldm_parameter_experiments.py --image1 bed1.png --image2 bed2.png --output results/interpolated.png --mu 0.5 0.6 0.7 --method noise_diffusion --timesteps 140
```

### 2. Interpolation with ControlNet

```bash
python controlnet_boundary_mu_experiments.py --image1 bed1.png --image2 bed2.png --output results/interpolated.png --mu 0.5 0.6 0.7 --method noise_diffusion --timesteps 140
```


### 3. LDM from Source Experiments

```bash
source_ldm_experiments.py --ckpt trial/trial.pth --image1 bed1.png --image2 bed2.png --output results/source_interpolated.png --frac 0.1 0.3 0.5 --method noise_diffusion --timesteps 140
```

---

## Some Results

### Varying NoiseDiffusion Parameters
![image](https://github.com/user-attachments/assets/f07e58e8-8926-4c25-8f6c-ea4f7279a538)


More results with varying other parameters can be found in the repository.

### AI and Natural Image Interpolation
![image](https://github.com/user-attachments/assets/68a98bab-a714-4d38-8d85-42aae5880be7)

### Interpolation between faces
![image](https://github.com/user-attachments/assets/9b83817e-1c76-4d17-aa5f-a2b5500407fa)

 
## References

- **NoiseDiffusion (Zheng et al., ICLR 2024)**  
- **Latent Diffusion Models** — Rombach et al., 2022  
- **ControlNet** — Zhang et al., 2023

---

## Reproducibility Notes

- The ControlNet logic is located in `cldm/`.
- Latent diffusion functions use modules from `ldm/`.

---

## Acknowledgements

This work reproduces and builds on the official [NoiseDiffusion](https://openreview.net/forum?id=6O3Q6AFUTu&referrer=%5Bthe%20profile%20of%20PengFei%20Zheng%5D(%2Fprofile%3Fid%3D~PengFei_Zheng2)) paper and codebase. We also utilize Latent Diffusion Models and ControlNet for additional image control and quality.


