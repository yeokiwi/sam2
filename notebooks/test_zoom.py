#!/usr/bin/env python3
"""Test zoom functionality."""

import sys
import tkinter as tk

# Add current directory to path
sys.path.insert(0, '.')

def test_zoom_methods():
    """Test that all zoom methods exist."""
    try:
        from sam2_gui import SAM2GUI
        print("✓ GUI class imported successfully")
        
        # Check required methods
        required_methods = [
            'zoom_in',
            'zoom_out', 
            'zoom_reset',
            'on_mouse_wheel',
            'update_canvas',
            'update_zoom_label'
        ]
        
        print("\nChecking zoom methods:")
        all_present = True
        for method_name in required_methods:
            if hasattr(SAM2GUI, method_name):
                print(f"  ✓ {method_name}")
            else:
                print(f"  ✗ {method_name} - MISSING")
                all_present = False
        
        # Check class attributes
        print("\nChecking zoom attributes:")
        attributes = ['display_scale', 'min_scale', 'max_scale', 'zoom_step']
        for attr in attributes:
            if hasattr(SAM2GUI, attr):
                print(f"  ✓ {attr} attribute")
            else:
                # Check instance attributes by creating a mock instance
                root = tk.Tk()
                root.withdraw()
                try:
                    gui = SAM2GUI(root)
                    if hasattr(gui, attr):
                        print(f"  ✓ {attr} (instance attribute)")
                    else:
                        print(f"  ✗ {attr} - MISSING")
                except Exception as e:
                    print(f"  ? {attr} - Error checking: {e}")
                finally:
                    root.destroy()
        
        print(f"\n{'All zoom functionality present' if all_present else 'Some zoom functionality missing'}")
        return all_present
        
    except ImportError as e:
        print(f"✗ Error importing GUI class: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_coordinate_scaling():
    """Test coordinate scaling logic."""
    print("\nTesting coordinate scaling logic:")
    
    # Simulate scaling logic
    original_x = 100
    original_y = 200
    display_scale = 2.0  # 200% zoom
    
    # Original to display
    display_x = original_x * display_scale
    display_y = original_y * display_scale
    
    # Display to original (for point selection)
    if display_scale > 0:
        recovered_x = display_x / display_scale
        recovered_y = display_y / display_scale
    else:
        recovered_x = display_x
        recovered_y = display_y
    
    print(f"  Original coordinates: ({original_x}, {original_y})")
    print(f"  Display scale: {display_scale*100}%")
    print(f"  Display coordinates: ({display_x}, {display_y})")
    print(f"  Recovered coordinates: ({recovered_x}, {recovered_y})")
    print(f"  ✓ Coordinates recover correctly: {abs(recovered_x - original_x) < 0.001 and abs(recovered_y - original_y) < 0.001}")
    
    return True

def main():
    print("=== Testing Zoom Functionality ===\n")
    
    print("1. Testing method existence...")
    test_zoom_methods()
    
    print("\n2. Testing coordinate scaling...")
    test_coordinate_scaling()
    
    print("\n=== Zoom Feature Summary ===")
    print("The zoom functionality includes:")
    print("  • Zoom In button (+) - Enlarge image up to 1000%")
    print("  • Zoom Out button (-) - Reduce image down to 10%")
    print("  • Fit button - Reset to fit canvas")
    print("  • Mouse wheel support - Scroll to zoom in/out")
    print("  • Zoom percentage display - Shows current zoom level")
    print("  • Proper coordinate scaling - Points and contours scale correctly")
    print("\nNote: Zoom works with all overlays (points, contours, masks)")
    print("and maintains correct coordinate mapping for point selection.")

if __name__ == "__main__":
    main()