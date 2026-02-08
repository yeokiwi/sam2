#!/usr/bin/env python
"""
Simple test for contour plotting.
"""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_show_mask():
    """Test show_mask function with enhanced contour plotting."""
    print("Testing show_mask with contour plotting...")
    
    # Create a simple mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[40:60, 20:40] = 0  # create a notch
    
    # Test different configurations
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Import show_mask
    from image import show_mask
    
    # Test 1: borders=True, plot_contours=True (default)
    print("1. Testing borders=True, plot_contours=True...")
    show_mask(mask, axes[0, 0], borders=True, plot_contours=True, epsilon=5.0)
    axes[0, 0].set_title("Borders + Contours (epsilon=5.0)")
    
    # Test 2: borders=False, plot_contours=True
    print("2. Testing borders=False, plot_contours=True...")
    show_mask(mask, axes[0, 1], borders=False, plot_contours=True, epsilon=2.0)
    axes[0, 1].set_title("Contours only (epsilon=2.0)")
    
    # Test 3: borders=True, plot_contours=False
    print("3. Testing borders=True, plot_contours=False...")
    show_mask(mask, axes[1, 0], borders=True, plot_contours=False, epsilon=5.0)
    axes[1, 0].set_title("Borders only")
    
    # Test 4: borders=False, plot_contours=False
    print("4. Testing borders=False, plot_contours=False...")
    show_mask(mask, axes[1, 1], borders=False, plot_contours=False)
    axes[1, 1].set_title("No borders/contours")
    
    plt.tight_layout()
    output_path = "test_contour_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Test plots saved to {output_path}")
    
    # Test CSV output
    print("\n5. Testing CSV output...")
    import tempfile
    import pandas as pd
    tmp_dir = tempfile.mkdtemp(prefix='csv_test_')
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    show_mask(mask, ax, borders=True, save_contours=True, 
              contour_output_dir=tmp_dir, epsilon=5.0, plot_contours=True)
    plt.close(fig)
    
    csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
    print(f"  Generated {len(csv_files)} CSV file(s)")
    for f in csv_files:
        df = pd.read_csv(os.path.join(tmp_dir, f))
        print(f"    {f}: {len(df)} points")
        print(f"      Example points:\n{df.head()}")
    
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("\n✓ All tests completed successfully!")

if __name__ == '__main__':
    test_show_mask()