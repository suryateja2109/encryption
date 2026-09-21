# A Robust Chaotic Encryption Scheme for Facial Images and Video Streams
## Reproduction, Enhancement, and Hardware Feasibility of the Ding et al. Framework

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-30%2F30%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Framework](https://img.shields.io/badge/IEEE%20TCSVT-July%202025-orange.svg)](https://ieeexplore.ieee.org/document/10892015)

This repository provides an end-to-end Python implementation, empirical validation, algorithmic enhancement, and hardware feasibility study of the chaotic facial-image encryption framework originally proposed by Ding et al. (*IEEE Transactions on Circuits and Systems for Video Technology*, Vol. 35, No. 7, pp. 6421–6434, July 2025).

---

## 🌟 Key Features & Innovations

1. **Bit-Exact Lossless Inversion**:
   - Solved continuous-to-discrete quantization errors in Semi-Tensor Product (STP) modular matrix diffusion using exact integer quotient matrix tracking ($\max |F_{\text{dec}} - F_{\text{orig}}| = 0$).
2. **3D Coupled Ikeda Map with Bounded Amplitude (3D-CIMBA)**:
   - Hyperchaotic continuous attractor with controllable Lyapunov exponents ($LE_1 \approx g$), wide continuous bifurcation across $a, b \in [0, 10]$, and Sample Entropy ($1.348$) outperforming classic 2D Ikeda ($0.512$).
3. **Chaotic-Adaptive PSO (APSO)**:
   - Dynamic non-linear inertia weight $w(t) \in [0.9 \to 0.4]$, time-varying acceleration coefficients $c_1(t), c_2(t)$, and chaotic perturbation.
   - Achieves higher global fitness ($15.3785$ vs $15.3583$) and a **$10.35\%$ runtime reduction** ($39.46$s vs $44.01$s).
4. **Group-of-Pictures (GOP) Video Stream Encryption**:
   - Extends the static image cipher to live video streams using OpenCV MIL visual tracking and continuous chaotic stream offset progression.
   - Delivers **$3.05$ FPS ($327.8$ ms/frame)** throughput — a **$13.7\times$ speedup** over per-frame full optimization.
5. **Multi-Face Selective Privacy**:
   - Supports multi-subject frames with independent cryptographic sub-keys derived per face: $K^{(i)} = \operatorname{KDF}(K_{\text{master}}, \text{FaceID}^{(i)}, \text{ROI}^{(i)})$.
6. **Embedded & FPGA Feasibility Study**:
   - Evaluated 24-bit fixed-point (Q8.16) arithmetic, proving zero entropy collapse ($H = 7.9129$).
   - Synthesis mapping to **Xilinx Artix-7 XC7A100T** FPGA confirms low resource footprint ($10.0\%$ DSPs, $6.67\%$ BRAM, $22.87\%$ LUTs) with a formal **"GO" verdict**.
7. **Extended Cryptanalysis Suite**:
   - Strict Avalanche Criterion (SAC): $49.97\% \approx 50.0\%$.
   - Chi-Square Uniformity Test: $\chi^2 = 239.37$, $p$-value $= 0.7508 \gg 0.05$ (Accept $H_0$).
   - Machine Learning Attack Resistance: Random Forest attack classifier accuracy held to $55.6\%$ (chance baseline: $50.0\%$).
8. **Interactive Streamlit Web Dashboard**:
   - Interactive GUI (`gui/app.py`) for static image encryption, real-time video playback, and live attack simulation (Gaussian noise, salt & pepper, cropping, and $10^{-15}$ key perturbation).

---

## 📊 Comparative Performance Benchmark

| # | Metric | Ideal Theoretical | Base Paper (Ding et al. 2025) | Phase 1 Baseline (Reproduction) | Phase 2–5 Enhanced System | Advantage / Status |
|---|---|---|---|---|---|---|
| 1 | **Cipher ROI Shannon Entropy ($H$)** | 8.0000 | *Not in source text* | 7.9834 | **7.9835** | Near theoretical ceiling |
| 2 | **Histogram Variance ($V$)** | 0.0 | *Not in source text* | 131.25 | **128.27** | Improved (>97% drop from plain) |
| 3 | **Horizontal Adjacent Correlation** | 0.0000 | *Not in source text* | +0.0093 | **+0.0017** | **$81.7\%$ reduction** towards ideal |
| 4 | **Vertical Adjacent Correlation** | 0.0000 | *Not in source text* | -0.0041 | **-0.0032** | **$22.0\%$ reduction** towards ideal |
| 5 | **Diagonal Adjacent Correlation** | 0.0000 | *Not in source text* | -0.0214 | **-0.0018** | **$91.6\%$ reduction** towards ideal |
| 6 | **Key Space Size** | $\ge 2^{256}$ | $10^{128} \approx 2^{425.2}$ | $10^{128} \approx 2^{425.2}$ | **$10^{128} \approx 2^{425.2}$** | Exceeds NIST standard (Immune to brute force) |
| 7 | **Key Sensitivity ($\Delta a = 10^{-15}$ SSIM)** | 0.0000 | *Noise output* | 0.0174 | **0.0174** | High sensitivity |
| 8 | **NPCR (Facial ROI %)** | 99.6094% | *Not in source text* | 99.6130% | **99.6130%** | **PASS** (Exceeds critical 99.5530%) |
| 9 | **UACI (Facial ROI %)** | 33.4635% | *Not in source text* | 33.4800% | **33.4800%** | **PASS** (Centered in 95% CI) |
| 10 | **Optimizer Fitness $F(p)$** | Maximized | *Not in source text* | 15.3583 | **15.3785** | Higher global fitness |
| 11 | **Optimizer Runtime (35 iter)** | Minimized | *Not in source text* | 44.01 s | **39.46 s** | **$10.35\%$ faster convergence** |
| 12 | **Video Throughput (Tracked)** | $\ge 24$ FPS | None (Static only) | N/A | **3.05 FPS (327.8 ms)** | **$13.7\times$ speedup** via GOP+MIL |
| 13 | **Clean Channel Max Pixel Error ($\Delta$)** | 0 | 0 | 0 (Bit-exact) | **0 (Bit-exact multi-frame)** | Mathematically lossless |
| 14 | **Strict Avalanche Criterion (SAC %)** | 50.00% | *Not tested* | N/A | **49.97%** | Near-ideal bit avalanche |
| 15 | **Chi-Square Uniformity $p$-value** | > 0.05 | *Not tested* | N/A | **0.7508** | Uniform noise ($H_0$ accepted) |
| 16 | **ML Attack Classifier Accuracy** | 50.00% | *Not tested* | N/A | **55.6%** | Resists AI cryptanalysis |
| 17 | **Multi-Face Independent Encryption** | Bit-exact | Single face only | Single face | **Supported ($\Delta = 0$)** | Independent sub-key privacy |
| 18 | **FPGA Feasibility Assessment** | Low footprint | Future work note | N/A | **GO (DSP: 10%, LUT: 23%)** | Feasible on low-cost FPGA |

---

## 📁 Repository Structure

```
encryption/
├── docs/                      # Technical documentation & final project report
│   ├── final_report.md        # Comprehensive 400+ line technical report
│   └── data_manifest.md       # LFW dataset audit & verification manifest
├── gui/                       # Streamlit interactive application
│   └── app.py                 # Multi-mode web dashboard
├── results/                   # Empirical benchmark data and figures
│   ├── figures/               # Generated bifurcation plots, decryptions, demo GIF
│   ├── tables/                # JSON benchmark exports (Phases 1–5)
│   ├── comparison_table.csv   # Consolidated master metric comparison table
│   └── baseline_report.md     # Phase 1 reproduction report
├── scripts/                   # Automated benchmarking and analysis scripts
│   ├── benchmark_phase2_optimizers.py
│   ├── benchmark_phase3_video.py
│   ├── benchmark_phase4_lightweight.py
│   ├── benchmark_phase5_improvements.py
│   ├── run_phase1_baseline.py
│   └── generate_comparison_table.py
├── src/                       # Production source code
│   ├── chaotic_map/           # 3D-CIMBA generator, Lyapunov exponents, sample entropy
│   ├── cryptanalysis/         # Shannon entropy, correlation, NPCR/UACI, SAC, ML attack
│   ├── encryption/            # Cyclic shifting, STP diffusion, lossless decryptor, multi-face
│   ├── face_processing/       # DeepFace database, Haar/MTCNN detectors, feature matcher
│   ├── lightweight_analysis/  # Fixed-point simulation (Q8.16, Q12.20) & FPGA profiler
│   └── optimization/          # Baseline PSO and Enhanced Chaotic-Adaptive PSO (APSO)
├── tests/                     # 27 automated test suites across all components
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit AMD64)
- Git

### 2. Clone Repository
```bash
git clone https://github.com/suryateja2109/<repo-name>.git
cd <repo-name>
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running Automated Tests

Verify the entire pipeline using `pytest` (100% test pass rate across 30 suites):

```bash
pytest tests/ -v
```

Test suites include:
- `tests/test_chaotic_map.py`: 3D-CIMBA dynamics, Lyapunov exponents, bifurcation ranges.
- `tests/test_encryption.py`: Cyclic shift permutation and STP matrix diffusion.
- `tests/test_encryption_lossless.py`: Bit-exact zero-error recovery validation ($\Delta = 0$).
- `tests/test_optimization.py`: PSO and APSO convergence against parameter bounds.
- `tests/test_cryptanalysis.py`: Entropy, correlation, NPCR, and UACI verification.
- `tests/test_face_processing.py`: DeepFace enrollment, detection, and database verification.

---

## 🖥️ Launching the Interactive Web GUI

Run the Streamlit dashboard to test live image/video encryption and attack simulations:

```bash
streamlit run gui/app.py
```

### Dashboard Modes:
1. **Static Facial Encryption**: Upload custom facial images or select from the gallery, choose detector (OpenCV vs MTCNN) and optimizer (Baseline PSO vs APSO), and inspect intermediate permutation/diffusion states.
2. **Video Stream Encryption**: Encrypt MP4/AVI clips using GOP tracking and inspect real-time frame throughput.
3. **Cryptanalytic Attack Simulator**: Test cipher resilience under Gaussian noise, Salt & Pepper noise, cropping occlusion, and $10^{-15}$ key perturbation.

---

## 📚 References & Citation

If you use this implementation in your research, please cite the foundational reference publication:

```bibtex
@article{ding2025deepface,
  title={Deepface-Based Chaotic Image Encryption Using Key Optimization and Semi-Tensor Product Theory},
  author={Ding, Dawei and Xie, Dong and Zhang, Hongwei and Yang, Zongli and Liu, Chu'an},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={35},
  number={7},
  pages={6421--6434},
  year={2025},
  publisher={IEEE}
}
```

---
*Developed as part of the RGMCET ECE Batch 14 Capstone Project.*
