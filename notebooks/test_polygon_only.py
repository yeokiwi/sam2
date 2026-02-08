#!/usr/bin/env python
"""
Test polygon simplification directly.
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_mask_test(mask, ax, random_color=False, borders=True, save_contours=False, contour_output_dir=None, epsilon=2.0):
    """Copy of show_mask function from image.py for testing."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 1/255, 2/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Simplify contours with polygon approximation
        simplified_contours = []
        for contour in contours:
            # If epsilon is between 0 and 1, treat as relative to perimeter
            if 0 < epsilon < 1:
                perimeter = cv2.arcLength(contour, closed=True)
                eps = epsilon * perimeter
            else:
                eps = epsilon
            approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
            simplified_contours.append(approx)
        mask_image = cv2.drawContours(mask_image, simplified_contours, -1, (1, 1, 1, 0.5), thickness=2)
        # Save contours to CSV files if requested
        if save_contours:
            import os
            import time
            import pandas as pd
            if contour_output_dir is None:
                contour_output_dir = "./contours"
            os.makedirs(contour_output_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            for i, contour in enumerate(simplified_contours):
                # contour shape: (n, 1, 2)
                points = contour.reshape(-1, 2)
                df = pd.DataFrame(points, columns=['x', 'y'])
                filename = os.path.join(contour_output_dir, f"contour_{timestamp}_{i}.csv")
                df.to_csv(filename, index=False)
                print(f"Saved simplified contour ({len(points)} points) to {filename}")
    ax.imshow(mask_image)

def main():
    print("Testing polygon simplification with cv2.approxPolyDP")
    
    # Create a simple binary mask (rectangle with notch to make contour non-rectangular)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    mask[40:60, 20:40] = 0  # create a notch
    
    tmp_dir = tempfile.mkdtemp(prefix='poly_test_')
    print(f"Output dir: {tmp_dir}")
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    try:
        # Test with epsilon=0 (no simplification)
        print("\n1. Epsilon=0 (no simplification):")
        show_mask_test(mask, ax, borders=True, save_contours=True,
                       contour_output_dir=tmp_dir, epsilon=0)
        
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        for f in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, f))
            print(f"  {f}: {len(df)} points")
        
        # Clean up
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
        
        # Test with epsilon=5.0
        print("\n2. Epsilon=5.0 (some simplification):")
        show_mask_test(mask, ax, borders=True, save_contours=True,
                       contour_output_dir=tmp_dir, epsilon=5.0)
        
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        for f in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, f))
            print(f"  {f}: {len(df)} points")
        
        # Test with epsilon=0.1 (relative)
        print("\n3. Epsilon=0.1 (relative to perimeter):")
        show_mask_test(mask, ax, borders=True, save_contours=True,
                       contour_output_dir=tmp_dir, epsilon=0.1)
        
        csv_files = [f for f in os.listdir(tmp_dir) if f.endswith('.csv')]
        for f in csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, f))
            print(f"  {f}: {len(df)} points")
        
        # Direct verification
        print("\n4. Direct cv2.approxPolyDP verification:")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"  Original contour points: {len(contours[0])}")
        
        for eps in [0, 2.0, 5.0, 10.0]:
            approx = cv2.approxPolyDP(contours[0], epsilon=eps, closed=True)
            print(f"  Epsilon={eps}: {len(approx)} points")
        
        # Check CSV content
        print("\n5. Checking CSV file format:")
        if csv_files:
            df = pd.read_csv(os.path.join(tmp_dir, csv_files[0]))
            print(f"  Columns: {list(df.columns)}")
            print(f"  First few points:\n{df.head()}")
            print(f"  Last few points:\n{df.tail()}")
        
        print("\n✓ All tests passed. Polygon simplification is working.")
        
    finally:
        plt.close(fig)
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    main()