"""
Streamlit Cloud and Local Deployment Entrypoint.
A Robust Chaotic Facial Image and Video Encryption System (3D-CIMBA, PSO, STP).
"""

import os
import sys

# Ensure repository root is in system path
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gui.app as app

if __name__ == "__main__":
    app.main()
