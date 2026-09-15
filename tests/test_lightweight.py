"""
Unit tests for lightweight fixed-point analysis and FPGA feasibility profiler.
"""

import pytest
import numpy as np
from src.lightweight_analysis.fixed_point import FixedPointCIMBA, evaluate_precision_degradation
from src.lightweight_analysis.complexity_profiler import estimate_per_pixel_operations, generate_fpga_feasibility_report


def test_fixed_point_cimba_boundedness():
    """Verify fixed-point CIMBA generator produces valid bounded states."""
    fp = FixedPointCIMBA(word_length=24, fraction_bits=16)
    xs, ys, zs = fp.iterate(n_steps=500, discard=100)

    assert len(xs) == 500
    assert np.all((xs >= 0.0) & (xs < 1.0))
    assert np.all((ys >= 0.0) & (ys < 1.0))
    assert np.all((zs >= 0.0) & (zs < 1.0))


def test_precision_degradation_benchmark():
    """Verify precision degradation comparison produces entropy measurements across bit widths."""
    res = evaluate_precision_degradation(n_points=600)
    assert "Double_64bit" in res
    assert "Fixed_32bit_Q12.20" in res
    assert "Fixed_24bit_Q8.16" in res
    assert "Fixed_16bit_Q8.8" in res

    # 32-bit and 24-bit should retain high entropy
    assert res["Fixed_32bit_Q12.20"]["entropy"] > 7.5
    assert res["Fixed_24bit_Q8.16"]["entropy"] > 7.0


def test_fpga_feasibility_report():
    """Verify FPGA resource estimation and Go/No-Go feasibility verdict."""
    ops = estimate_per_pixel_operations(m=64, n=64, roundnum=1000)
    assert ops["total_pixels"] == 4096
    assert ops["per_pixel_multiplications"] > 0
    assert ops["per_pixel_additions"] > 0

    report = generate_fpga_feasibility_report()
    assert report["verdict"] == "GO"
    assert report["estimated_utilization"]["DSP_utilization_pct"] < 50.0
    assert report["estimated_utilization"]["LUT_utilization_pct"] < 50.0
