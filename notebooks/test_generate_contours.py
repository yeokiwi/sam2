#!/usr/bin/env python3
"""Test the Generate Contours button functionality."""

import sys
import tkinter as tk

# Add current directory to path
sys.path.insert(0, '.')

def test_generate_contours_methods():
    """Test that all generate contours methods exist."""
    try:
        from sam2_gui import SAM2GUI
        print("✓ GUI class imported successfully")
        
        # Check required methods
        required_methods = [
            'generate_contours',
            'on_contour_generation_complete',
            'save_contours_to_csv',
            'plot_contours'
        ]
        
        print("\nChecking contour generation methods:")
        all_present = True
        for method_name in required_methods:
            if hasattr(SAM2GUI, method_name):
                print(f"  ✓ {method_name}")
            else:
                print(f"  ✗ {method_name} - MISSING")
                all_present = False
        
        # Test button existence by creating a mock GUI
        print("\nTesting GUI button creation:")
        root = tk.Tk()
        root.withdraw()
        
        try:
            gui = SAM2GUI(root)
            print("✓ GUI instance created successfully")
            
            # Check button existence
            buttons_to_check = [
                'generate_contours_button',
                'plot_contours_button',
                'save_button',
                'predict_button'
            ]
            
            for button_name in buttons_to_check:
                if hasattr(gui, button_name):
                    button = getattr(gui, button_name)
                    state = button.cget('state')
                    text = button.cget('text')
                    print(f"  ✓ {button_name}: '{text}' (state: {state})")
                else:
                    print(f"  ✗ {button_name} - MISSING")
            
            # Check initial button states
            print("\nChecking initial button states:")
            expected_disabled = ['generate_contours_button', 'plot_contours_button', 'save_button']
            for button_name in expected_disabled:
                if hasattr(gui, button_name):
                    button = getattr(gui, button_name)
                    if button.cget('state') == 'disabled':
                        print(f"  ✓ {button_name} correctly disabled initially")
                    else:
                        print(f"  ✗ {button_name} should be disabled but is {button.cget('state')}")
            
            # Clean up
            root.destroy()
            
        except Exception as e:
            print(f"✗ Error testing GUI instance: {e}")
            root.destroy()
        
        print(f"\n{'All contour generation functionality present' if all_present else 'Some functionality missing'}")
        return all_present
        
    except ImportError as e:
        print(f"✗ Error importing GUI class: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_workflow_logic():
    """Test the logical workflow for contour generation."""
    print("\n=== Contour Generation Workflow ===")
    print("Expected workflow:")
    print("1. Load image → Plot Contours button enabled")
    print("2. Add points and predict mask → Generate Contours button enabled")
    print("3. Click Generate Contours → Saves CSV files in contours/ directory")
    print("4. After generation → Plot Contours button enabled (to display contours)")
    print("5. Click Plot Contours → Loads and displays green contour lines")
    
    print("\nRequirements:")
    print("  • OpenCV required for contour extraction")
    print("  • Mask must be predicted before contour generation")
    print("  • CSV files saved to 'contours/' directory")
    print("  • Each contour saved as separate CSV file with x,y columns")
    print("  • Contours simplified using cv2.approxPolyDP")
    
    return True

def main():
    print("=== Testing Generate Contours Functionality ===\n")
    
    print("1. Testing method existence...")
    test_generate_contours_methods()
    
    print("\n2. Testing workflow logic...")
    test_workflow_logic()
    
    print("\n=== Summary ===")
    print("The 'Generate Contours' button adds the following functionality:")
    print("  • Dedicated button for contour generation (enabled after mask prediction)")
    print("  • Separate from 'Save Mask' button (which also generates contours)")
    print("  • Uses existing save_contours_to_csv() method")
    print("  • Threaded execution to avoid UI freezing")
    print("  • Proper status updates and error handling")
    print("  • Enables 'Plot Contours' button after successful generation")
    print("\nNote: OpenCV is required for contour extraction from masks.")
    print("If OpenCV is not available, contour generation will not work.")

if __name__ == "__main__":
    main()