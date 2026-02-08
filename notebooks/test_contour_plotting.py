#!/usr/bin/env python3
"""Test script to verify contour plotting functionality."""

import os
import sys
import csv
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

def test_csv_loading():
    """Test that CSV files in contours directory can be loaded."""
    contours_dir = "contours"
    if not os.path.exists(contours_dir):
        print(f"Warning: Contours directory '{contours_dir}' not found.")
        print("Creating a test contour file...")
        os.makedirs(contours_dir, exist_ok=True)
        
        # Create a simple circular contour for testing
        center_x, center_y = 100, 100
        radius = 50
        points = []
        for angle in np.linspace(0, 2*np.pi, 36):  # 36 points around circle
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append((x, y))
        
        # Save to CSV
        test_csv = os.path.join(contours_dir, "test_contour_0.csv")
        with open(test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for x, y in points:
                writer.writerow([x, y])
        
        print(f"Created test contour file: {test_csv}")
    
    # List CSV files
    import glob
    csv_files = glob.glob(os.path.join(contours_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in '{contours_dir}':")
    for csv_file in csv_files[:5]:  # Show first 5
        print(f"  - {os.path.basename(csv_file)}")
    
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more")
    
    # Test loading a CSV file
    if csv_files:
        test_file = csv_files[0]
        print(f"\nTesting loading of '{os.path.basename(test_file)}':")
        points = []
        try:
            with open(test_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        x, y = float(row[0]), float(row[1])
                        points.append((x, y))
            
            print(f"  Successfully loaded {len(points)} points")
            if points:
                print(f"  First point: {points[0]}")
                print(f"  Last point: {points[-1]}")
                print(f"  Points shape: {len(points)} points")
        except Exception as e:
            print(f"  Error loading CSV: {e}")
    
    return True

def test_gui_methods():
    """Test that GUI methods exist and can be called."""
    try:
        # Import GUI class
        from sam2_gui import SAM2GUI
        
        print("\nTesting GUI class methods:")
        
        # Check required methods
        required_methods = [
            'plot_contours',
            'save_contours_to_csv',
            'update_canvas',
            'load_image',
            'predict_mask'
        ]
        
        for method_name in required_methods:
            if hasattr(SAM2GUI, method_name):
                print(f"  ✓ Method '{method_name}' exists")
            else:
                print(f"  ✗ Method '{method_name}' missing")
        
        # Check class attributes
        print("\nChecking class initialization attributes:")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        try:
            gui = SAM2GUI(root)
            print("  ✓ GUI instance created successfully")
            
            # Check important attributes
            attributes = ['contours', 'points', 'original_image', 'current_mask']
            for attr in attributes:
                if hasattr(gui, attr):
                    print(f"  ✓ Attribute '{attr}' exists")
                else:
                    print(f"  ✗ Attribute '{attr}' missing")
            
            # Clean up
            root.destroy()
            
        except Exception as e:
            print(f"  ✗ Error creating GUI instance: {e}")
            root.destroy()
        
    except ImportError as e:
        print(f"Error importing GUI class: {e}")
        return False
    
    return True

def main():
    print("=== Testing Contour Plotting Functionality ===\n")
    
    print("1. Testing CSV file loading...")
    test_csv_loading()
    
    print("\n2. Testing GUI methods...")
    test_gui_methods()
    
    print("\n=== Test Summary ===")
    print("The contour plotting functionality should work as follows:")
    print("1. Load an image using the 'Load Image' button")
    print("2. Add points and predict a mask (optional)")
    print("3. Save the mask (this creates contour CSV files if OpenCV is available)")
    print("4. Click 'Plot Contours' to load and display green contour lines")
    print("\nNote: OpenCV is required for contour extraction from masks.")
    print("If OpenCV is not available, you can manually create contour CSV files")
    print("in the 'contours/' directory with x,y columns.")

if __name__ == "__main__":
    main()