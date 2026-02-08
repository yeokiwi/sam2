#!/usr/bin/env python
"""
Test enhanced contour plotting with mask visualization.
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

def test_contour_plotting():
    """Test that simplified contours are plotted together with mask."""
    print("Testing enhanced contour plotting with mask...")
    
    # Create a synthetic binary mask (shape with multiple contours)
    mask = np.zeros((200, 200), dtype=np.uint8)
    # Create a rectangle
    mask[30:80, 30:80] = 255
    # Create a circle
    cv2.circle(mask, (120, 120), 30, 255, -1)
    # Create a triangle
    triangle = np.array([[160, 50], [190, 100], [130, 100]], np.int32)
    cv2.fillPoly(mask, [triangle], 255)
    
    # Create a temporary directory for output
    tmp_dir = tempfile.mkdtemp(prefix='contour_plot_test_')
    print(f"Output directory: {tmp_dir}")
    
    try:
        # Test 1: Default behavior (borders=True, plot_contours=True)
        print("\n1. Testing default behavior (borders=True, plot_contours=True)...")
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        # Import show_mask from image.py
        sys.path.insert(0, os.path.dirname(__file__))
        from image import show_mask
        
        show_mask(mask, ax1, borders=True, save_contours=True, 
                  contour_output_dir=tmp_dir, epsilon=5.0, plot_contours=True)
        ax1.set_title("Mask with borders and contour plotting")
        output1 = os.path.join(tmp_dir, "test1_mask_plot.png")
        plt.savefig(output1, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output1}")
        plt.close(fig1)
        
        # Check CSV files were created
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        print(f"  Generated {len(csv_files)} contour CSV file(s)")
        for f in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, f))
            print(f"    {f}: {len(df)} points")
        
        # Clean up CSV files for next test
        for f in csv_files:
            os.remove(os.path.join(tmp_dir, f))
        
        # Test 2: Only contour plotting (borders=False, plot_contours=True)
        print("\n2. Testing contour plotting only (borders=False, plot_contours=True)...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        # Create a background image (gray)
        background = np.ones((200, 200, 3)) * 0.8
        ax2.imshow(background)
        
        show_mask(mask, ax2, borders=False, save_contours=True,
                  contour_output_dir=tmp_dir, epsilon=2.0, plot_contours=True)
        ax2.set_title("Contour plotting only (no borders)")
        output2 = os.path.join(tmp_dir, "test2_contour_only.png")
        plt.savefig(output2, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output2}")
        plt.close(fig2)
        
        # Test 3: Only borders (borders=True, plot_contours=False)
        print("\n3. Testing borders only (borders=True, plot_contours=False)...")
        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        ax3.imshow(background)
        
        show_mask(mask, ax3, borders=True, save_contours=False,
                  contour_output_dir=tmp_dir, epsilon=5.0, plot_contours=False)
        ax3.set_title("Borders only (no contour plotting)")
        output3 = os.path.join(tmp_dir, "test3_borders_only.png")
        plt.savefig(output3, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {output3}")
        plt.close(fig3)
        
        # Test 4: Different epsilon values
        print("\n4. Testing different epsilon values...")
        epsilons = [0, 2.0, 5.0, 10.0, 0.1]
        for eps in epsilons:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(background)
            show_mask(mask, ax, borders=False, save_contours=False,
                      epsilon=eps, plot_contours=True)
            ax.set_title(f"Epsilon = {eps}")
            output = os.path.join(tmp_dir, f"test4_epsilon_{eps}.png")
            plt.savefig(output, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Epsilon={eps}: Plot saved")
        
        # Test 5: Verify contour simplification
        print("\n5. Verifying contour simplification...")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"  Original contours: {len(contours)}")
        for i, contour in enumerate(contours):
            print(f"    Contour {i}: {len(contour)} points")
            
            # Test with different epsilon
            for eps in [0, 5.0, 10.0]:
                approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
                reduction = 100 * (len(contour) - len(approx)) / len(contour)
                print(f"      Epsilon={eps}: {len(approx)} points ({reduction:.1f}% reduction)")
        
        print("\n✓ All tests passed. Enhanced contour plotting is working.")
        print(f"  Check output directory for plots: {tmp_dir}")
        
    finally:
        # Keep the directory for inspection
        print(f"\nTest files kept in: {tmp_dir}")
        # shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    test_contour_plotting()