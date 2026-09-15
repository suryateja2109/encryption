"""
Interactive Streamlit Dashboard for Robust Chaotic Facial Image & Video Encryption.
Reference: Ding et al. (IEEE TCSVT 2025) and RGMCET ECE Batch 14 Project.
"""

import os
import sys
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from src.chaotic_map.cimba3d import CIMBAMap
from src.optimization.baseline_pso import BaselinePSO
from src.optimization.enhanced_optimizer import ChaoticAdaptivePSO
from src.face_processing.detector import detect_face_roi, detect_multiple_faces
from src.encryption.cipher_pipeline import encrypt_full_image
from src.encryption.decryptor import decrypt_full_image
from src.encryption.multi_face_cipher import encrypt_multi_face_image, decrypt_multi_face_image
from src.cryptanalysis.metrics import compute_image_entropy, compute_histogram_variance, compute_ssim_psnr
from src.cryptanalysis.correlation import evaluate_image_correlations

st.set_page_config(
    page_title="Chaotic Face & Video Encryption (3D-CIMBA, PSO, STP)",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 Robust Chaotic Facial Image & Video Encryption")
st.caption("Based on Ding et al., IEEE TCSVT 2025 & RGMCET Dept. of ECE, Batch 14")

# Sidebar Configuration
st.sidebar.header("⚙️ System Configuration")
mode = st.sidebar.selectbox("Operation Mode", ["Single Face Selective", "Multi-Face Selective", "Video Demo / Benchmark"])
detector_backend = st.sidebar.selectbox("Face Detector Backend", ["opencv", "mtcnn"])
optimizer_choice = st.sidebar.selectbox("Key Optimizer", ["Chaotic-Adaptive PSO (Phase 2)", "Baseline PSO (Phase 1)", "Default 3D-CIMBA Keys"])
roundnum = st.sidebar.slider("Cyclic Shift Rounds (Alg. 2)", min_value=100, max_value=6000, value=1000, step=100)

if mode in ["Single Face Selective", "Multi-Face Selective"]:
    st.subheader("1. Input Image Selection")
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload Face Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        sample_path = "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
        use_sample = st.checkbox("Use LFW Dataset Sample", value=True if uploaded_file is None else False)

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        input_image = np.array(pil_img)
    elif use_sample and os.path.exists(sample_path):
        input_image = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
    else:
        # Fallback synthetic face
        input_image = np.full((250, 250, 3), 180, dtype=np.uint8)
        cv2.circle(input_image, (125, 125), 60, (220, 180, 150), -1)
        cv2.circle(input_image, (105, 110), 8, (50, 50, 50), -1)
        cv2.circle(input_image, (145, 110), 8, (50, 50, 50), -1)

    with col2:
        st.image(input_image, caption="Plain Input Image", use_container_width=True)

    st.markdown("---")
    st.subheader("2. Selective Encryption Execution")

    if st.button("🚀 Run Chaotic Face Encryption Pipeline", type="primary"):
        with st.spinner("Executing Face Detection, Key Optimization, Cyclic Shifting & STP Diffusion..."):
            t0 = time.perf_counter()

            # 1. Detection
            if mode == "Single Face Selective":
                roi, bbox = detect_face_roi(input_image, backend=detector_backend)
                st.info(f"Detected Facial ROI: Bounding Box (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")

                # 2. Key Optimization
                if optimizer_choice == "Chaotic-Adaptive PSO (Phase 2)":
                    crop = cv2.resize(roi, (64, 64))
                    opt = ChaoticAdaptivePSO(n_particles=15, n_iterations=10, seed=42)
                    cimba_key, best_fit, _ = opt.optimize(crop, eval_rounds=50)
                elif optimizer_choice == "Baseline PSO (Phase 1)":
                    crop = cv2.resize(roi, (64, 64))
                    opt = BaselinePSO(n_particles=15, n_iterations=10, seed=42)
                    cimba_key, best_fit, _ = opt.optimize(crop, eval_rounds=50)
                else:
                    cimba_key = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])

                # 3. Encryption & Decryption
                cipher_img, meta = encrypt_full_image(input_image, bbox=bbox, cimba_params=cimba_key, roundnum=roundnum)
                decrypted_img = decrypt_full_image(cipher_img, meta)

                # Lossless check
                x, y, w, h = bbox
                plain_roi = input_image[y : y + h, x : x + w]
                c_roi = cipher_img[y : y + h, x : x + w]
                dec_roi = decrypted_img[y : y + h, x : x + w]
                bit_diff = np.max(np.abs(plain_roi.astype(int) - dec_roi.astype(int)))
            else:
                cimba_key = np.array([10.0, 10.0, 0.4, 6.0, 10.0, 0.1, 0.2, 0.3])
                cipher_img, meta_list = encrypt_multi_face_image(input_image, cimba_key, roundnum=roundnum)
                decrypted_img = decrypt_multi_face_image(cipher_img, meta_list)
                bit_diff = np.max(np.abs(input_image.astype(int) - decrypted_img.astype(int)))
                c_roi = cipher_img

            elapsed = time.perf_counter() - t0

        st.success(f"Pipeline Finished in {elapsed:.2f} seconds! Bit-Exact Lossless Invertibility: {'✅ PASSED (Error = 0)' if bit_diff == 0 else '❌ FAILED'}")

        # Display Images Side by Side
        im_col1, im_col2, im_col3 = st.columns(3)
        with im_col1:
            st.image(input_image, caption="Plain Image", use_container_width=True)
        with im_col2:
            st.image(cipher_img, caption="Cipher Image (Selective STP)", use_container_width=True)
        with im_col3:
            st.image(decrypted_img, caption="Lossless Decrypted Image", use_container_width=True)

        st.markdown("---")
        st.subheader("3. Real-Time Cryptographic Metrics")

        ent = compute_image_entropy(c_roi)
        corrs = evaluate_image_correlations(c_roi, n_samples=2000)
        hist_var = compute_histogram_variance(c_roi)
        ssim_val, psnr_val = compute_ssim_psnr(input_image, decrypted_img)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cipher ROI Entropy", f"{ent['mean']:.4f}", "Ideal: 8.0000")
        m2.metric("Pixel Correlation", f"{corrs['horizontal']['mean']:+.4f}", "Ideal: 0.0000")
        m3.metric("Histogram Variance", f"{hist_var:.2f}", "Plain: > 4500")
        m4.metric("Decrypted SSIM", f"{ssim_val:.4f}", "Lossless: 1.0000")

elif mode == "Video Demo / Benchmark":
    st.subheader("Video Selective Encryption Demo & Tracking")
    st.info("Demonstrates face tracking, GOP key reuse (15 frames), and per-frame STP diffusion throughput.")

    video_json_path = "results/tables/phase3_video_benchmark.json"
    gif_path = "results/figures/video_encryption_demo.gif"
    montage_path = "results/figures/video_frames_montage.png"

    if os.path.exists(video_json_path):
        with open(video_json_path, "r") as f:
            v_data = json.load(f)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Total Frames", v_data["total_frames"])
        v2.metric("Intermediate FPS", f"{v_data['intermediate_fps']:.2f} FPS")
        v3.metric("Mean Cipher Entropy", f"{v_data['mean_cipher_entropy']:.4f}")
        v4.metric("Lossless Across All", "✅ TRUE" if v_data["all_bit_exact"] else "❌ FALSE")

    col_a, col_b = st.columns(2)
    if os.path.exists(gif_path):
        with col_a:
            st.image(gif_path, caption="Selective Video Encryption Stream (Animated GIF)", use_container_width=True)
    if os.path.exists(montage_path):
        with col_b:
            st.image(montage_path, caption="Keyframe vs. Tracked Frames Montage", use_container_width=True)
