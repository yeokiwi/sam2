import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def main():
    # Paths
    image_path = "images/test.jpg"
    contours_dir = "contours"
    
    # Load image
    image = Image.open(image_path)
    image = np.array(image)
    
    # Find all contour CSV files
    csv_files = glob.glob(os.path.join(contours_dir, "contour_*.csv"))
    if not csv_files:
        print(f"No contour CSV files found in {contours_dir}")
        return
    
    print(f"Found {len(csv_files)} contour files.")
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Colors for different contours
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_files)))
    
    for idx, csv_file in enumerate(csv_files):
        # Read contour points
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        x = df['x'].values
        y = df['y'].values
        
        # Close the contour by appending first point at the end
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        
        # Plot contour lines
        ax.plot(x_closed, y_closed, color=colors[idx], linewidth=2, label=os.path.basename(csv_file))
        
        # Compute center (mean of points)
        center_x = np.mean(x)
        center_y = np.mean(y)
        
        # Annotate with filename (without extension)
        #filename = os.path.splitext(os.path.basename(csv_file))[0]
        #ax.text(center_x, center_y, filename, fontsize=9, color='white',
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
        #        horizontalalignment='center', verticalalignment='center')
    
    ax.set_title("Contours from CSV files")
    ax.axis('off')
    # Optionally add legend (might be crowded)
    # ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    
    # Save figure
    output_path = "contours_display.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Show plot (blocking)
    plt.show()

if __name__ == "__main__":
    main()
