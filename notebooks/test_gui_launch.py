#!/usr/bin/env python3
"""Test that the GUI can be launched (without SAM2 model)."""

import sys
import os

# Set environment variable for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Mock SAM2 if not available
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    HAS_SAM2 = True
    print("SAM2 imports successful")
except ImportError:
    HAS_SAM2 = False
    print("SAM2 not found - creating mock classes")
    # Create mock classes
    class MockSAM2Predictor:
        def set_image(self, image):
            print("Mock: set_image called")
        def predict(self, **kwargs):
            print("Mock: predict called")
            import numpy as np
            return [np.zeros((100, 100), dtype=bool)], [0.5], [None]
    
    class MockBuildSAM2:
        pass
    
    # Create mock module
    import types
    sam2_module = types.ModuleType('sam2')
    sam2_module.build_sam = MockBuildSAM2
    sam2_module.sam2_image_predictor = types.ModuleType('sam2_image_predictor')
    sam2_module.sam2_image_predictor.SAM2ImagePredictor = MockSAM2Predictor
    sys.modules['sam2'] = sam2_module
    sys.modules['sam2.build_sam'] = sam2_module
    sys.modules['sam2.sam2_image_predictor'] = sam2_module.sam2_image_predictor
    
    # Now import the GUI
    print("Mock SAM2 created")

# Import GUI
sys.path.insert(0, '.')
from sam2_gui import SAM2GUI

import tkinter as tk

def test_gui():
    """Create and destroy GUI quickly."""
    root = tk.Tk()
    root.withdraw()  # Hide window
    app = SAM2GUI(root)
    
    # Check some basic attributes
    print(f"GUI created successfully")
    print(f"  Canvas: {app.canvas}")
    print(f"  Load button: {app.load_button}")
    print(f"  Model loaded: {app.model_loaded}")
    
    # Schedule destruction
    root.after(100, root.destroy)
    root.mainloop()
    print("GUI test completed")

if __name__ == "__main__":
    test_gui()