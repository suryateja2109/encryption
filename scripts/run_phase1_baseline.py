"""
Script to execute Phase 1 reproduction benchmark against the LFW dataset
and collect verified empirical results for results/baseline_report.md.
"""

import os
import sys
import glob
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.chaotic_map.cimba3d import CIMBAMap
from src.optimization.baseline_pso import BaselinePSO
from src.face_processing.detector import detect_face_roi
from src.face_processing.database import FaceDatabase
from src.encryption.cipher_pipeline import encrypt_full_image, encrypt_face_roi
from src.encryption.decryptor import decrypt_full_image
from src.cryptanalysis.metrics import compute_image_entropy, compute_histogram_variance, compute_ssim_psnr
from src.cryptanalysis.correlation import evaluate_image_correlations, sample_adjacent_pixel_pairs
from src.cryptanalysis.differential import evaluate_differential_security, compute_npcr_uaci, compute_critical_thresholds
from src.cryptanalysis.robustness import benchmark_robustness_suite
from src.cryptanalysis.randomness import (
    compute_key_space,
    evaluate_key_sensitivity,
    benchmark_encryption_speed,
    run_nist_statistical_tests,
)

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("--- Starting Phase 1 Baseline Reproduction Benchmark ---")

# 1. Select 5 test images from LFW
# Aaron_Eckhart_0001, Abdullah_Gul_0013, Al_Pacino_0001, Alan_Greenspan_0001, Albert_Costa_0002
test_images_info = [
    ("Aaron_Eckhart", "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"),
    ("Abdullah_Gul", "archive/lfw-deepfunneled/lfw-deepfunneled/Abdullah_Gul/Abdullah_Gul_0013.jpg"),
    ("Al_Pacino", "archive/lfw-deepfunneled/lfw-deepfunneled/Al_Pacino/Al_Pacino_0001.jpg"),
    ("Alan_Greenspan", "archive/lfw-deepfunneled/lfw-deepfunneled/Alan_Greenspan/Alan_Greenspan_0001.jpg"),
    ("Albert_Costa", "archive/lfw-deepfunneled/lfw-deepfunneled/Albert_Costa/Albert_Costa_0002.jpg"),
]

# 2. Build Facial Database with gallery of 10 individuals
print("Step 1: Building facial database...")
face_db = FaceDatabase(model_name="Facenet", detector_backend="opencv", threshold=0.5)
gallery_names = [
    "Aaron_Eckhart", "Abdullah_Gul", "Al_Pacino", "Alan_Greenspan", "Albert_Costa",
    "Alejandro_Toledo", "Ali_Naimi", "Allison_Janney", "Alvaro_Uribe", "Amanda_Bynes"
]
for g_name in gallery_names:
    jpgs = glob.glob(f"archive/lfw-deepfunneled/lfw-deepfunneled/{g_name}/*.jpg")
    if jpgs:
        bgr = cv2.imread(jpgs[0])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_db.enroll_image(g_name, rgb)

# Test 5 probe images: 4 in gallery (different image), 1 not in gallery
# Mismatched probe: Arnold_Schwarzenegger
probes = [
    ("Image 1 (Aaron_Eckhart_0001)", "archive/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg", "Aaron_Eckhart"),
    ("Image 2 (Abdullah_Gul_0014)", "archive/lfw-deepfunneled/lfw-deepfunneled/Abdullah_Gul/Abdullah_Gul_0014.jpg", "Abdullah_Gul"),
    ("Image 3 (Al_Pacino_0002)", "archive/lfw-deepfunneled/lfw-deepfunneled/Al_Pacino/Al_Pacino_0002.jpg", "Al_Pacino"),
    ("Image 4 (Alan_Greenspan_0005)", "archive/lfw-deepfunneled/lfw-deepfunneled/Alan_Greenspan/Alan_Greenspan_0005.jpg", "Alan_Greenspan"),
    ("Image 5 (Arnold_Schwarzenegger)", "archive/lfw-deepfunneled/lfw-deepfunneled/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0001.jpg", "Not In Database"),
]

table2_results = []
print("\nStep 2: Face Recognition and Matching (Table II)...")
for label, path, expected in probes:
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = face_db.query_probe(rgb)
    status = "Successful" if res["is_match"] else "Failed"
    table2_results.append({
        "Image": label,
        "Expected": expected,
        "Matched_Identity": res["matched_name"] or "None",
        "Euclidean_Distance": float(res["min_distance"]),
        "Threshold": 0.5,
        "Comparison_Result": status,
    })
    print(f"  {label}: dist={res['min_distance']:.4f} -> {status} (matched: {res['matched_name']})")

# 3. Encrypt Image 1 and run full Cryptanalysis Suite
print("\nStep 3: Running Key Optimization & Encryption on Image 1...")
img1_path = test_images_info[0][1]
img1_bgr = cv2.imread(img1_path)
img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

# Detect face
roi, bbox = detect_face_roi(img1_rgb, backend="opencv")
print(f"  Detected facial bbox: {bbox}, ROI shape: {roi.shape}")

# Random 64x64 crop for PSO optimization
crop_y = np.random.randint(0, img1_rgb.shape[0] - 64)
crop_x = np.random.randint(0, img1_rgb.shape[1] - 64)
crop_64 = img1_rgb[crop_y : crop_y + 64, crop_x : crop_x + 64]

pso = BaselinePSO(n_particles=20, n_iterations=20, seed=42)
opt_key, best_fitness, pso_history = pso.optimize(crop_64, eval_rounds=100, verbose=True)
print(f"  Optimized 3D-CIMBA key: {np.round(opt_key, 4)}, fitness: {best_fitness:.4f}")

# Encrypt facial ROI
cipher_img, meta = encrypt_full_image(img1_rgb, bbox=bbox, cimba_params=opt_key, roundnum=6000)
decrypted_img = decrypt_full_image(cipher_img, meta)

# Bit-exact check on ROI
dec_roi = decrypted_img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
bit_exact_err = np.max(np.abs(roi.astype(int) - dec_roi.astype(int)))
print(f"  Bit-exact decryption check on clean channel: max error = {bit_exact_err}")

# Save Visual Figures (Fig. 6 in paper)
fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
axs[0].imshow(img1_rgb)
axs[0].set_title("(a) Plain Image", fontsize=11, fontweight="bold")
axs[0].axis("off")
axs[1].imshow(cipher_img)
axs[1].set_title("(b) Cipher Image (Selective)", fontsize=11, fontweight="bold")
axs[1].axis("off")
axs[2].imshow(decrypted_img)
axs[2].set_title("(c) Decrypted Image", fontsize=11, fontweight="bold")
axs[2].axis("off")
plt.tight_layout()
plt.savefig("results/figures/plain_cipher_decrypted_faces.png", dpi=200)
plt.close(fig)

# 4. Histograms (Fig. 8 in paper)
print("\nStep 4: Generating Histograms...")
fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=150)
colors = ["red", "green", "blue"]
channel_names = ["R", "G", "B"]
for c in range(3):
    axs[0, c].hist(roi[:, :, c].flatten(), bins=256, range=(0, 256), color=colors[c], alpha=0.7)
    axs[0, c].set_title(f"Plain ROI ({channel_names[c]} Channel)")
    axs[0, c].set_xlim([0, 256])
    axs[0, c].grid(True, linestyle="--", alpha=0.3)

    cipher_roi = cipher_img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    axs[1, c].hist(cipher_roi[:, :, c].flatten(), bins=256, range=(0, 256), color=colors[c], alpha=0.7)
    axs[1, c].set_title(f"Cipher ROI ({channel_names[c]} Channel)")
    axs[1, c].set_xlim([0, 256])
    axs[1, c].grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/histograms_plain_vs_cipher.png", dpi=200)
plt.close(fig)

# 5. Entropy Analysis (Table III in paper)
print("\nStep 5: Computing Entropy & Hist Variance (Table III)...")
plain_entropy = compute_image_entropy(roi)
cipher_entropy = compute_image_entropy(cipher_roi)
plain_var = compute_histogram_variance(roi)
cipher_var = compute_histogram_variance(cipher_roi)
print(f"  Plain ROI Entropy: {plain_entropy['mean']:.4f} (Var: {plain_var:.2f})")
print(f"  Cipher ROI Entropy: {cipher_entropy['mean']:.4f} (Var: {cipher_var:.2f})")

# 6. Correlation Analysis (Table IV & V in paper)
print("\nStep 6: Computing Adjacent Pixel Correlation (Table IV & V)...")
plain_corrs = evaluate_image_correlations(roi, n_samples=3000)
cipher_corrs = evaluate_image_correlations(cipher_roi, n_samples=3000)

for d in ["horizontal", "vertical", "diagonal"]:
    print(f"  Direction {d:10s} | Plain: {plain_corrs[d]['mean']:+.4f} | Cipher: {cipher_corrs[d]['mean']:+.4f}")

# Correlation scatter plots (Fig. 9 in paper)
fig, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=150)
for idx, d in enumerate(["horizontal", "vertical", "diagonal"]):
    xp, yp = sample_adjacent_pixel_pairs(roi[:, :, 0], direction=d, n_samples=1000)
    xc, yc = sample_adjacent_pixel_pairs(cipher_roi[:, :, 0], direction=d, n_samples=1000)

    axs[0, idx].scatter(xp, yp, s=1.0, c="blue", alpha=0.4)
    axs[0, idx].set_title(f"Plain {d.capitalize()} (R)")
    axs[0, idx].set_xlim([0, 255]); axs[0, idx].set_ylim([0, 255])
    axs[0, idx].grid(True, linestyle="--", alpha=0.3)

    axs[1, idx].scatter(xc, yc, s=1.0, c="red", alpha=0.4)
    axs[1, idx].set_title(f"Cipher {d.capitalize()} (R)")
    axs[1, idx].set_xlim([0, 255]); axs[1, idx].set_ylim([0, 255])
    axs[1, idx].grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/correlation_scatter_plots.png", dpi=200)
plt.close(fig)

# 7. Key Space & Key Sensitivity (Section V-D)
print("\nStep 7: Key Space & Key Sensitivity (Section V-D)...")
key_space_info = compute_key_space()
key_sens_info = evaluate_key_sensitivity(img1_rgb, bbox=bbox, base_params=opt_key, delta=1e-15)

# Also compute ROI-specific sensitivity
dec_roi_a = key_sens_info["param_a_perturbed"]["decrypted"][bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
dec_roi_x1 = key_sens_info["state_x1_perturbed"]["decrypted"][bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
roi_ssim_a, _ = compute_ssim_psnr(roi, dec_roi_a)
roi_ssim_x1, _ = compute_ssim_psnr(roi, dec_roi_x1)

print(f"  Key Space: {key_space_info['formula']} (~{key_space_info['equivalent_bits']:.1f} bits)")
print(f"  Key Sensitivity param 'a' + 1e-15 (Full Image SSIM: {key_sens_info['param_a_perturbed']['SSIM']:.4f}, Facial ROI SSIM: {roi_ssim_a:.4f})")
print(f"  Key Sensitivity state 'x1' + 1e-15 (Full Image SSIM: {key_sens_info['state_x1_perturbed']['SSIM']:.4f}, Facial ROI SSIM: {roi_ssim_x1:.4f})")

# Key sensitivity visual plot (Fig. 10)
fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
axs[0].imshow(decrypted_img)
axs[0].set_title("Correct Key (Lossless Decryption)")
axs[0].axis("off")
axs[1].imshow(key_sens_info["param_a_perturbed"]["decrypted"])
axs[1].set_title(r"Wrong Key ($a + 10^{-15}$)")
axs[1].axis("off")
axs[2].imshow(key_sens_info["state_x1_perturbed"]["decrypted"])
axs[2].set_title(r"Wrong Key ($x_1 + 10^{-15}$)")
axs[2].axis("off")
plt.tight_layout()
plt.savefig("results/figures/key_sensitivity_decryptions.png", dpi=200)
plt.close(fig)

# 8. Differential Attacks: NPCR & UACI (Table VII, VIII, IX in paper)
# Using plaintext-associated key derivation (Ding et al. Contribution 3 & Section IV-A)
print("\nStep 8: Differential Attack Analysis (NPCR & UACI)...")
import hashlib
h1 = hashlib.sha256(roi.tobytes()).digest()

roi_perturbed = roi.copy()
roi_perturbed[10, 10, 0] ^= 1  # 1 bit toggle
h2 = hashlib.sha256(roi_perturbed.tobytes()).digest()

opt_key_p1 = opt_key.copy()
opt_key_p2 = opt_key.copy()
opt_key_p1[5] = (opt_key[5] + int.from_bytes(h1[:4], "big") / (2**32)) % 1.0
opt_key_p2[5] = (opt_key[5] + int.from_bytes(h2[:4], "big") / (2**32)) % 1.0

c_roi1, _ = encrypt_face_roi(roi, opt_key_p1, roundnum=6000)
c_roi2, _ = encrypt_face_roi(roi_perturbed, opt_key_p2, roundnum=6000)

diff_results = evaluate_differential_security(c_roi1, c_roi2, alpha=0.05)
print(f"  Facial ROI NPCR: {diff_results['NPCR']:.4f}% (Critical threshold: {diff_results['NPCR_critical']:.4f}%) -> Passed: {diff_results['NPCR_passed']}")
print(f"  Facial ROI UACI: {diff_results['UACI']:.4f}% (Critical range: [{diff_results['UACI_lower']:.4f}%, {diff_results['UACI_upper']:.4f}%]) -> Passed: {diff_results['UACI_passed']}")
print(f"  Overall Differential Security Passed: {diff_results['Overall_passed']}")

# 9. Robustness Analysis (Fig. 11 in paper)
print("\nStep 9: Robustness Analysis (Noise & Cropping)...")
robustness_results = benchmark_robustness_suite(img1_rgb, cipher_img, meta)
fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
attack_keys = list(robustness_results.keys())
for idx, k in enumerate(attack_keys):
    r, c = idx // 3, idx % 3
    axs[r, c].imshow(robustness_results[k]["decrypted"])
    axs[r, c].set_title(f"{k}\nPSNR: {robustness_results[k]['PSNR']:.2f} dB | SSIM: {robustness_results[k]['SSIM']:.4f}", fontsize=9)
    axs[r, c].axis("off")
plt.tight_layout()
plt.savefig("results/figures/robustness_decryptions.png", dpi=200)
plt.close(fig)

for k, val in robustness_results.items():
    print(f"  {k:15s} | SSIM: {val['SSIM']:.4f} | PSNR: {val['PSNR']:.2f} dB")

# 10. Speed & Randomness Tests
print("\nStep 10: Speed Benchmark & Randomness Battery...")
speed_info = benchmark_encryption_speed(img1_rgb, bbox, opt_key, roundnum=1000, n_runs=3)
print(f"  Execution time (1000 rounds): {speed_info['avg_time_sec']*1000:.2f} ms ({speed_info['fps']:.2f} FPS)")
print(f"  Estimated clock cycles (@ 3.5 GHz): {speed_info['clock_cycles']:.2e}")

cimba = CIMBAMap(a=opt_key[0], b=opt_key[1], delta=opt_key[2], K=opt_key[3], g=opt_key[4], x1=opt_key[5], y1=opt_key[6], z1=opt_key[7])
nist_info = run_nist_statistical_tests(cimba, n_bits=50000)
print(f"  NIST Randomness tests: passed {nist_info['passed_tests']}/{nist_info['total_tests']} ({nist_info['pass_rate']*100:.1f}%)")

# Save consolidated benchmark dict to JSON for baseline report
baseline_metrics = {
    "table2_matching": table2_results,
    "plain_entropy": plain_entropy,
    "cipher_entropy": cipher_entropy,
    "plain_variance": plain_var,
    "cipher_variance": cipher_var,
    "plain_correlations": plain_corrs,
    "cipher_correlations": cipher_corrs,
    "key_space": key_space_info,
    "key_sensitivity": {
        "a_perturbed_ssim": key_sens_info["param_a_perturbed"]["SSIM"],
        "x1_perturbed_ssim": key_sens_info["state_x1_perturbed"]["SSIM"],
    },
    "differential": diff_results,
    "robustness": {k: {"SSIM": robustness_results[k]["SSIM"], "PSNR": robustness_results[k]["PSNR"]} for k in attack_keys},
    "speed": speed_info,
    "nist": {
        "total": nist_info["total_tests"],
        "passed": nist_info["passed_tests"],
        "pass_rate": nist_info["pass_rate"],
        "details": nist_info["details"],
    },
    "optimized_key": opt_key.tolist(),
}

with open("results/tables/phase1_baseline_metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)

print("\n--- Phase 1 Benchmark Complete! Metrics saved to results/tables/phase1_baseline_metrics.json ---")
