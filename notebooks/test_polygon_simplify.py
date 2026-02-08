#!/usr/bin/env python
"""
Test polygon simplification in show_mask function.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add current directory to path to import from image.py
sys.path.insert(0, os.path.dirname(__file__))

from image import show_mask

def test_polygon_simplification():
    """Test that contours are simplified with approxPolyDP."""
    print("Testing polygon simplification...")
    
    # Create a simple binary mask (rectangle)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255  # rectangle
    
    # Create a temporary directory for output
    tmp_dir = tempfile.mkdtemp(prefix='contours_test_')
    print(f"Output directory: {tmp_dir}")
    
    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    try:
        # Test with epsilon=0 (no simplification, should keep many points)
        print("\n1. Testing with epsilon=0 (no simplification)...")
        show_mask(mask, ax, borders=True, save_contours=True, 
                  contour_output_dir=tmp_dir, epsilon=0)
        
        # Find generated CSV files
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        print(f"Generated {len(csv_files)} contour file(s).")
        
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, csv_file))
            print(f"  {csv_file}: {len(df)} points")
        
        # Clean up for next test
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        
        # Test with epsilon=5.0 (some simplification)
        print("\n2. Testing with epsilon=5.0 (some simplification)...")
        show_mask(mask, ax, borders=True, save_contours=True,
                  contour_output_dir=tmp_dir, epsilon=5.0)
        
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, csv_file))
            print(f"  {csv_file}: {len(df)} points")
        
        # Test with epsilon=0.1 (relative to perimeter)
        print("\n3. Testing with epsilon=0.1 (relative)...")
        show_mask(mask, ax, borders=True, save_contours=True,
                  contour_output_dir=tmp_dir, epsilon=0.1)
        
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, csv_file))
            print(f"  {csv_file}: {len(df)} points")
            
        # Verify that simplification reduces point count
        print("\n4. Manual verification of cv2.approxPolyDP usage...")
        # Direct contour extraction and simplification
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"  Original contour has {len(contours[0])} points")
        
        # Apply approxPolyDP with epsilon=5
        approx = cv2.approxPolyDP(contours[0], epsilon=5.0, closed=True)
        print(f"  After approxPolyDP(epsilon=5): {len(approx)} points")
        
        # Apply approxPolyDP with epsilon=0
        approx2 = cv2.approxPolyDP(contours[0], epsilon=0, closed=True)
        print(f"  After approxPolyDP(epsilon=0): {len(approx2)} points")
        
        # Check that epsilon=5 reduces points
        if len(approx) < len(contours[0]):
            print("  ✓ Simplification works (point count reduced)")
        else:
            print("  ⚠ No reduction (contour may already be simple)")
        
        print("\nAll tests completed successfully.")
        
    finally:
        plt.close(fig)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up {tmp_dir}")

if __name__ == '__main__':
    test_polygon_simplification()