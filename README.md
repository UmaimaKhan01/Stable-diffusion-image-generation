Stable Diffusion XL: Turbo vs Base Comparison
Overview

This project compares two variants of Stable Diffusion XL (SDXL) — Turbo and Base — for text-to-image generation.
The objective is to evaluate differences in generation speed, image fidelity, and resource efficiency on a mid-range GPU (RTX 4050, 6 GB VRAM).

Methodology
1. Image Generation

Images were generated using the Hugging Face Diffusers library with both models under identical prompts.

Prompts stored in prompts.txt

Outputs saved in:

data/generated_images/sdxl_base/
data/generated_images/sdxl_turbo/

2. Speed Evaluation

The evaluate_speed.py script measures per-image generation time using Python’s time module.
Results are averaged and visualized with Matplotlib as speed_comparison.png.

3. FID Evaluation

Quality is quantified using the Fréchet Inception Distance (FID) metric.
The script evaluate_fid.py uses a pre-trained InceptionV3 model to extract 2048-dimensional features and computes:

𝐹
𝐼
𝐷
=
∣
∣
𝜇
1
−
𝜇
2
∣
∣
2
+
𝑇
𝑟
(
Σ
1
+
Σ
2
−
2
(
Σ
1
Σ
2
)
1
/
2
)
FID=∣∣μ
1
	​

−μ
2
	​

∣∣
2
+Tr(Σ
1
	​

+Σ
2
	​

−2(Σ
1
	​

Σ
2
	​

)
1/2
)

A lower FID indicates higher similarity between generated and reference image distributions.

Results
Metric	SDXL Turbo	SDXL Base	Ratio
Avg Time (s/img)	15.47	705.58	×45.6 faster
Images per second	0.0646	0.0014	–
FID (Turbo vs Base)	247.09	–	–

SDXL Turbo generated images ~45× faster than SDXL Base.

Turbo maintained comparable semantic accuracy with minor loss in fine detail.

SDXL Base achieved higher fidelity but required significantly more GPU time and memory.

Repository Structure
Stable-diffusion-image-generation/
│
├── generate_images.py
├── evaluate_speed.py
├── evaluate_fid.py
├── run_comparison.py
├── generate_report.py
│
├── data/
│   ├── prompts.txt
│   ├── generated_images/
│   └── evaluation_results/
│
├── requirements_exact.txt
└── main.tex

Environment

Python 3.10

PyTorch 2.1.0 + CUDA 11.8

Diffusers 0.25.1

Transformers 4.35.2

NumPy, SciPy, Matplotlib, Pillow

Hardware: NVIDIA RTX 4050 (6 GB VRAM)
OS: Windows 11

Commands
# Create environment
conda create -n sdxl_env python=3.10 -y
conda activate sdxl_env

# Install dependencies
pip install -r requirements_exact.txt

# Generate images
python generate_images.py

# Evaluate speed
python evaluate_speed.py

# Compute FID
python evaluate_fid.py

# Generate final report
python generate_report.py

Key Findings

SDXL Turbo is significantly faster and more memory-efficient.

SDXL Base produces sharper textures and higher realism but is unsuitable for real-time inference on limited hardware.

Quantitative metrics (FID, time) confirm the expected trade-off between speed and quality.

Acknowledgments

Stability AI for SDXL models

Hugging Face Diffusers and PyTorch for open-source frameworks
