# Streamlit Deployment Guide

## 1. Local Deployment (Active)

The Streamlit dashboard is currently running locally as a background service:

- **Local Access URL**: [http://localhost:8501](http://localhost:8501)
- **Health Check Endpoint**: [http://localhost:8501/_stcore/health](http://localhost:8501/_stcore/health) (Status: `200 ok`)
- **Main Entrypoint**: [`streamlit_app.py`](../streamlit_app.py) or [`gui/app.py`](../gui/app.py)

To start the server manually at any time:
```powershell
python -m streamlit run streamlit_app.py
```

---

## 2. One-Click Cloud Deployment (Streamlit Community Cloud)

This repository is pre-configured for deployment on **Streamlit Community Cloud** (`share.streamlit.io`):

### Deployment Assets Included
1. **[`streamlit_app.py`](../streamlit_app.py)**: Auto-detected root entry point for Streamlit Cloud.
2. **[`packages.txt`](../packages.txt)**: Specifies required Linux system packages for headless OpenCV and DeepFace (`libgl1`, `libglib2.0-0`, `libgomp1`).
3. **[`.streamlit/config.toml`](../.streamlit/config.toml)**: Dark mode theme, optimized upload limits (200MB), and headless server settings.
4. **[`requirements.txt`](../requirements.txt)**: Complete pinned Python dependencies.
5. **Self-contained Fallbacks**: [`gui/app.py`](../gui/app.py) dynamically resolves sample face images from `test case/used_faces/` so the app works even when the multi-gigabyte LFW dataset archive is excluded via `.gitignore`.

### Steps to Deploy Online (Free):

1. **Push your changes to GitHub**:
   ```bash
   git add streamlit_app.py packages.txt .streamlit/ gui/app.py docs/streamlit_deployment.md
   git commit -m "feat: configure Streamlit deployment and root entrypoint"
   git push origin main
   ```

2. **Connect to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
   - Click **"New app"**.

3. **Configure App Settings**:
   - **Repository**: `suryateja2109/encryption`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom subdomain (e.g., `chaotic-facial-encryption.streamlit.app`)

4. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit Cloud will install packages from `packages.txt` and `requirements.txt` and publish your live web dashboard.

---

## 3. Features Available in the Streamlit Dashboard

- **Static Facial Image Encryption**:
  - Select from pre-loaded LFW dataset samples.
  - Upload custom facial images (`.jpg`, `.jpeg`, `.png`).
  - 📸 **Live Webcam Capture**: Click a live photo from your webcam to encrypt your own face.
  - Toggle between **OpenCV** and **MTCNN** face detectors.
  - Choose between **Chaotic-Adaptive PSO (APSO)**, **Baseline PSO**, or **Default Keys**.
- **Bit-Exact Lossless Invertibility**:
  - Live verification that maximum pixel error $\Delta = 0$, PSNR = $\infty$, and SSIM = 1.000000.
- **Full Cryptanalysis Battery**:
  - Information Entropy ($H \approx 7.98$) across R, G, B channels.
  - Pearson Adjacent Correlation ($\rho \approx 0$) in Horizontal, Vertical, Diagonal directions.
  - NPCR ($> 99.60\%$) and UACI ($\approx 33.48\%$) Differential Security tests.
  - Chi-square ($\chi^2$) Uniformity test ($p > 0.05$) and local Shannon entropy.
  - NIST SP 800-22 live randomness battery.
  - Attack Robustness Simulator (Gaussian noise, Salt & Pepper, Cropping).
- **Video Stream Tracking & Encryption**:
  - View GOP-based real-time tracking throughput (3.05 FPS / 13.7× speedup) and animated stream visualization.
