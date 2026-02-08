# SAM2 Interactive GUI for Image Segmentation

A graphical user interface built with Python and Tkinter for interactive image segmentation using the SAM2 (Segment Anything Model 2) model.

## Features

- **Load Images**: Load any standard image format (JPG, PNG, BMP, TIFF)
- **Interactive Point Selection**: 
  - Positive points (green): Indicate foreground objects
  - Negative points (red): Indicate background regions
- **Real-time Mask Prediction**: Predict segmentation masks based on selected points
- **Mask Visualization**: Overlay masks on original images with transparency
- **Save Results**: Save predicted masks and overlay images
- **User-friendly Interface**: Intuitive controls with status feedback

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch (CPU or CUDA)
- Tkinter (usually included with Python)
- NumPy
- Pillow (PIL)

### Optional Dependencies
- OpenCV (for better point visualization)
- SAM2 model and dependencies (hydra, omegaconf)

### Installation

1. **Install Python dependencies**:
   ```bash
   pip install numpy pillow torch
   ```

2. **Install SAM2** (if not already installed):
   ```bash
   cd d:\sam2
   pip install -e .
   ```

3. **Download SAM2 model checkpoint** (if not already present):
   - Download `sam2.1_hiera_large.pt` from the [SAM2 repository](https://github.com/facebookresearch/sam2)
   - Place it in `d:\sam2\checkpoints\`

## Usage

### Running the GUI

1. Navigate to the notebooks directory:
   ```bash
   cd d:\sam2\notebooks
   ```

2. Run the GUI application:
   ```bash
   python sam2_gui.py
   ```

### User Interface Guide

1. **Load an Image**:
   - Click the "Load Image" button
   - Select an image file from the dialog
   - Images from the `images/` folder are available for testing

2. **Select Points**:
   - Click "Add Positive Point" to enter positive point mode (green)
   - Click "Add Negative Point" to enter negative point mode (red)
   - Click on the image to place points
   - Selected points appear in the list below

3. **Predict Mask**:
   - Click "Predict Mask" to generate segmentation
   - The model will predict masks based on your points
   - The best mask will be displayed as a red overlay

4. **Save Results**:
   - Click "Save Mask" to save the predicted mask
   - Choose whether to save the overlay image as well

5. **Clear Points**:
   - Click "Clear Points" to remove all points and masks
   - Start over with new points

### Test Images

Sample images are available in the `images/` folder:
- `truck.jpg` - Example from SAM2 documentation
- `groceries.jpg` - Grocery store scene
- `cars.jpg` - Multiple vehicles
- `test.jpg` - High-resolution test image

## Troubleshooting

### Common Issues

1. **OpenCV Import Error**:
   - The GUI will automatically fall back to PIL drawing
   - Points will still be visible (circles with X markers)

2. **SAM2 Model Not Found**:
   - Ensure SAM2 is installed: `pip install -e .` from the sam2 directory
   - Check that the model checkpoint exists at `../checkpoints/sam2.1_hiera_large.pt`

3. **Hydra Import Error**:
   - Install hydra: `pip install hydra-core`

4. **CUDA Not Available**:
   - The GUI will automatically use CPU or MPS (Apple Silicon)
   - Performance may be slower but functionality remains

### Testing

Run the test scripts to verify dependencies:

```bash
# Test basic imports
python test_gui_imports.py

# Test GUI launch (without model)
python test_gui_launch.py
```

## File Structure

- `sam2_gui.py` - Main GUI application
- `test_gui_imports.py` - Dependency test script
- `test_gui_launch.py` - GUI launch test
- `README_GUI.md` - This documentation
- `images/` - Sample images for testing

## Implementation Details

The GUI uses:
- **Tkinter** for the windowing framework
- **PIL/Pillow** for image loading and display
- **Threading** for non-blocking model loading and prediction
- **NumPy** for array operations
- **PyTorch** for model inference

The interface includes:
- Scrollable canvas for large images
- Status bar for feedback
- Point list for tracking selections
- Responsive button states

## Limitations

- Large images may be resized for display (original resolution maintained for prediction)
- Model loading can take time (especially on CPU)
- Requires SAM2 model files (~900MB)

## Future Enhancements

Potential improvements:
- Box selection mode
- Multiple mask selection
- Mask refinement with additional points
- Batch processing
- Export options (JSON, PNG, CSV)

## License

This GUI is built on top of the SAM2 model from Meta Research. Refer to the SAM2 repository for license details.