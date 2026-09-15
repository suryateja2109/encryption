"""
Phase 4 Lightweight and Hardware-Readiness Study:
Generates fixed-point degradation curves, operation counts, and FPGA feasibility assessment.
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from src.lightweight_analysis.fixed_point import evaluate_precision_degradation
from src.lightweight_analysis.complexity_profiler import estimate_per_pixel_operations, generate_fpga_feasibility_report

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("--- Starting Phase 4 Lightweight & Hardware-Readiness Benchmark ---")

# 1. Precision Degradation Benchmark
print("\n[1/3] Benchmarking Fixed-Point Precision Degradation...")
precision_results = evaluate_precision_degradation(n_points=2000)
for p_name, data in precision_results.items():
    print(f"  {p_name:20s} | Entropy: {data['entropy']:.4f} | Mean: {data['mean']:.4f} | Std: {data['std']:.4f}")

# 2. Per-Pixel Operation Profile
print("\n[2/3] Profiling Hardware Operations...")
ops_profile = estimate_per_pixel_operations(m=105, n=105, roundnum=6000)
print(f"  Total Multiplications: {ops_profile['total_multiplications']:,}")
print(f"  Total Additions:       {ops_profile['total_additions']:,}")
print(f"  Per-Pixel Multiplications: {ops_profile['per_pixel_multiplications']:.1f}")
print(f"  Per-Pixel Additions:       {ops_profile['per_pixel_additions']:.1f}")

# 3. FPGA Feasibility Assessment
print("\n[3/3] Generating FPGA Resource Feasibility Assessment...")
fpga_report = generate_fpga_feasibility_report(target_fpga="Xilinx Artix-7 XC7A100T")
print(f"  Target Device: {fpga_report['target_fpga']}")
print(f"  Verdict:       {fpga_report['verdict']}")
print(f"  DSP Usage:     {fpga_report['estimated_utilization']['DSP_utilization_pct']:.1f}% ({fpga_report['estimated_utilization']['DSP48E1_slices']} slices)")
print(f"  BRAM Usage:    {fpga_report['estimated_utilization']['BRAM_utilization_pct']:.1f}% ({fpga_report['estimated_utilization']['Block_RAM_18Kb']} blocks)")
print(f"  LUT Usage:     {fpga_report['estimated_utilization']['LUT_utilization_pct']:.1f}% ({fpga_report['estimated_utilization']['Logic_LUTs']} LUTs)")

# 4. Plot Precision Comparison
names = list(precision_results.keys())
entropies = [precision_results[k]["entropy"] for k in names]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
bars = ax.bar(names, entropies, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"], width=0.55)
ax.axhline(8.0, color="k", linestyle="--", label="Ideal Entropy (8.0)")
ax.set_ylim([6.0, 8.2])
ax.set_ylabel("Shannon Entropy", fontsize=11, fontweight="bold")
ax.set_title("Chaotic Sequence Entropy vs. Fixed-Point Precision", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.3, axis="y")
ax.legend(frameon=True)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plot_path = "results/figures/fixed_point_precision_entropy.png"
plt.savefig(plot_path, dpi=200)
plt.close(fig)
print(f"Saved precision comparison plot to {plot_path}")

# 5. Save Phase 4 Output JSON
phase4_output = {
    "precision_degradation": precision_results,
    "operation_counts": ops_profile,
    "fpga_feasibility": fpga_report,
}

with open("results/tables/phase4_lightweight_study.json", "w") as f:
    json.dump(phase4_output, f, indent=2)

print("--- Phase 4 Study Completed Successfully! ---")
