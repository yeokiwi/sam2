#!/usr/bin/env python3
"""Test script to verify GUI dependencies are installed."""

import sys
import os

# Set environment variable for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print("Testing imports...")

# Test basic imports
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    print("✓ tkinter imports successful")
except ImportError as e:
    print(f"✗ tkinter import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy import successful")
except ImportError as e:
    print(f"✗ numpy import failed: {e}")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    print("✓ PIL import successful")
except ImportError as e:
    print(f"✗ PIL import failed: {e}")
    sys.exit(1)

try:
    import torch
    print("✓ torch import successful")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ torch import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV import successful, version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    sys.exit(1)

# Test SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✓ SAM2 imports successful")
except ImportError as e:
    print(f"✗ SAM2 import failed: {e}")
    print("Note: SAM2 must be installed from the repository")
    sys.exit(1)

# Check model files
print("\nChecking model files...")
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

if os.path.exists(sam2_checkpoint):
    print(f"✓ Checkpoint found: {sam2_checkpoint}")
else:
    print(f"✗ Checkpoint not found: {sam2_checkpoint}")

if os.path.exists(model_cfg):
    print(f"✓ Config file found: {model_cfg}")
else:
    print(f"✗ Config file not found: {model_cfg}")

print("\nAll imports successful! GUI should be ready to run.")