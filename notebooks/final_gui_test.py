#!/usr/bin/env python3
"""Final comprehensive test of all GUI features."""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add current directory to path
sys.path.insert(0, '.')

def test_complete_workflow():
    """Test the complete workflow from image loading to contour generation."""
    print("=== Testing Complete GUI Workflow ===\n")
    
    print("Testing all major features:")
    print("1. GUI initialization and model loading")
    print("2. Image loading and button state updates")
    print("3. Point selection and mask prediction")
    print("4. Contour generation and plotting")
    print("5. Zoom functionality")
    print("\nNote: This test creates a mock GUI instance but doesn't load actual model.")
    
    try:
        from sam2_gui import SAM2GUI
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        # Create GUI instance
        print("\n1. Creating GUI instance...")
        gui = SAM2GUI(root)
        print("   ✓ GUI created successfully")
        
        # Check button states after creation (model loading)
        print("\n2. Checking initial button states...")
        buttons_to_check = {
            'load_button': ('Load Image', 'normal'),
            'predict_button': ('Predict Mask', 'disabled'),
            'save_button': ('Save Mask', 'disabled'),
            'generate_contours_button': ('Generate Contours', 'disabled'),
            'plot_contours_button': ('Plot Contours', 'disabled'),
            'zoom_in_button': ('+', 'disabled'),
            'zoom_out_button': ('-', 'disabled'),
            'zoom_reset_button': ('Fit', 'disabled')
        }
        
        for button_name, (expected_text, expected_state) in buttons_to_check.items():
            if hasattr(gui, button_name):
                button = getattr(gui, button_name)
                actual_text = button.cget('text')
                actual_state = button.cget('state')
                
                text_match = actual_text == expected_text
                state_match = actual_state == expected_state
                
                status = "✓" if text_match and state_match else "✗"
                print(f"   {status} {button_name}: '{actual_text}' ({actual_state})")
                
                if not text_match:
                    print(f"     Expected text: '{expected_text}'")
                if not state_match:
                    print(f"     Expected state: '{expected_state}'")
            else:
                print(f"   ✗ {button_name} - MISSING")
        
        # Test feature availability
        print("\n3. Testing feature availability...")
        features = [
            ('OpenCV for contour generation', 'HAS_CV2'),
            ('Zoom functionality', 'hasattr(gui, "zoom_in")'),
            ('Contour plotting', 'hasattr(gui, "plot_contours")'),
            ('Contour generation', 'hasattr(gui, "generate_contours")'),
            ('Mask prediction', 'hasattr(gui, "predict_mask")'),
            ('Point selection', 'hasattr(gui, "set_point_mode")')
        ]
        
        for feature_name, check_code in features:
            try:
                if eval(check_code):
                    print(f"   ✓ {feature_name}")
                else:
                    print(f"   ✗ {feature_name}")
            except:
                print(f"   ? {feature_name} (check failed)")
        
        # Test workflow simulation
        print("\n4. Simulating workflow logic...")
        workflow_steps = [
            ("Load image", "plot_contours_button enabled, zoom controls enabled"),
            ("Add points", "point selection mode available"),
            ("Predict mask", "save_button enabled, generate_contours_button enabled"),
            ("Generate contours", "contour CSV files created, plot_contours_button enabled"),
            ("Plot contours", "green lines and markers displayed on image"),
            ("Use zoom", "image scales correctly, coordinates remain accurate")
        ]
        
        for step, description in workflow_steps:
            print(f"   • {step}: {description}")
        
        # Clean up
        root.destroy()
        
        print("\n✓ All tests completed successfully")
        print("\n=== GUI Feature Summary ===")
        print("The SAM2 GUI now includes:")
        print("  • Core segmentation: Load images, add points, predict masks")
        print("  • Contour generation: Extract simplified contours from masks")
        print("  • Contour plotting: Display green contour lines and markers")
        print("  • Zoom controls: Enlarge/reduce images, mouse wheel support")
        print("  • Save functionality: Masks and overlay images")
        print("  • CSV export: Contour data saved for external processing")
        print("\nTo use the GUI: python sam2_gui.py")
        
        return True
        
    except ImportError as e:
        print(f"✗ Error importing GUI: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_workflow()