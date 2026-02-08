# Polygon Simplification in SAM2 Image Visualization

## Summary of Changes

The `show_mask` function in `image.py` has been enhanced to use `cv2.approxPolyDP` for contour simplification, reducing the number of polygon edges while preserving shape.

## Key Features

1. **Polygon Simplification**: Contours are now simplified using OpenCV's `cv2.approxPolyDP` algorithm
2. **Flexible Epsilon Parameter**: 
   - Absolute epsilon values (e.g., `epsilon=5.0`)
   - Relative epsilon values (0-1) as percentage of perimeter (e.g., `epsilon=0.1`)
3. **Backward Compatibility**: All existing parameters remain unchanged
4. **CSV Output**: Simplified polygons are saved to CSV files with `x,y` coordinates
5. **Visual Display**: Simplified contours are displayed with borders on the mask overlay

## Implementation Details

### Modified Functions

1. **`show_mask(mask, ax, ..., epsilon=2.0)`**:
   - Added `epsilon` parameter (default: 2.0)
   - When `0 < epsilon < 1`: treated as relative to contour perimeter
   - Otherwise: treated as absolute pixel distance threshold
   - Simplified contours are drawn and saved to CSV

2. **`show_masks(image, masks, scores, ..., epsilon=2.0)`**:
   - Added `epsilon` parameter to propagate to `show_mask`
   - Default value: 2.0 pixels

### Example Usage

```python
# Default simplification (epsilon=2.0)
show_mask(mask, ax, borders=True, save_contours=True)

# Stronger simplification
show_mask(mask, ax, borders=True, save_contours=True, epsilon=10.0)

# Relative simplification (10% of perimeter)
show_mask(mask, ax, borders=True, save_contours=True, epsilon=0.1)

# In the main pipeline
show_masks(image, masks, scores, save_contours=True, contour_output_dir="contours", epsilon=5.0)
```

## CSV Output Format

CSV files are saved with naming pattern: `contour_{timestamp}_{index}.csv`
Each file contains:
- `x`: X-coordinate of polygon vertex
- `y`: Y-coordinate of polygon vertex

Example:
```csv
x,y
20,20
20,39
40,40
40,59
20,60
20,79
79,79
79,20
```

## Testing

Run the test script to verify functionality:
```bash
python test_polygon_only.py
```

Expected output shows point count reduction:
- Original contour: 274 points
- Epsilon=0: 10 points (minimal)
- Epsilon=5.0: 8 points
- Epsilon=0.1: 4 points (significant simplification)

## Performance Benefits

- Reduced file size for CSV exports
- Smoother polygon rendering
- More efficient downstream processing
- Maintains essential shape characteristics

## Notes

- The implementation is backward compatible; existing code continues to work
- The `epsilon` parameter controls simplification strength
- CSV files are saved only when `save_contours=True`
- Visual borders show the simplified polygon outlines