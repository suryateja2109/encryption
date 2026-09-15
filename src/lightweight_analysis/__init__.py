"""
Lightweight and hardware readiness analysis package.
"""

from .fixed_point import FixedPointCIMBA, evaluate_precision_degradation
from .complexity_profiler import estimate_per_pixel_operations, generate_fpga_feasibility_report

__all__ = [
    "FixedPointCIMBA",
    "evaluate_precision_degradation",
    "estimate_per_pixel_operations",
    "generate_fpga_feasibility_report",
]
