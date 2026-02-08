import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True, save_contours=False, contour_output_dir=None, epsilon=2.0, plot_contours=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 1/255, 2/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    simplified_contours = []
    if borders or plot_contours:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Simplify contours with polygon approximation
        for contour in contours:
            # If epsilon is between 0 and 1, treat as relative to perimeter
            if 0 < epsilon < 1:
                perimeter = cv2.arcLength(contour, closed=True)
                eps = epsilon * perimeter
            else:
                eps = epsilon
            approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
            simplified_contours.append(approx)
        
        if borders:
            mask_image = cv2.drawContours(mask_image, simplified_contours, -1, (0, 1, 0, 1), thickness=4)
        
        # Plot simplified contours as matplotlib lines for better visibility
        if plot_contours:
            for contour in simplified_contours:
                if len(contour) > 0:
                    # Reshape contour points
                    points = contour.reshape(-1, 2)
                    # Close the polygon by appending first point
                    if len(points) > 0:
                        closed_points = np.vstack([points, points[0]])
                        # Plot as red line with markers at vertices
                        ax.plot(closed_points[:, 0], closed_points[:, 1], 'r-', linewidth=3, alpha=0.8)
                        # Plot vertices as blue dots
                        ax.scatter(points[:, 0], points[:, 1], c='blue', s=30, alpha=0.8, edgecolors='white', linewidths=1)
        
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

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_contours=False, contour_output_dir=None, epsilon=2.0, plot_contours=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = 1 - mask
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders, save_contours=save_contours, contour_output_dir=contour_output_dir, epsilon=epsilon, plot_contours=plot_contours)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        break   # show only the first mask for brevity
        

image = Image.open('images/test.jpg')
image = np.array(image.convert("RGB"))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

# Interactive point selection
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title("Click to select multiple points (press Enter when done, timeout 30 seconds)")
plt.axis('on')
print("Click to add points (positive). Press Enter when done (you have 30 seconds)...")
try:
    clicks = plt.ginput(n=-1, timeout=30)  # wait up to 30 seconds, unlimited clicks
    if clicks:
        input_point = np.array(clicks)
        print(f"Selected {len(input_point)} points:")
        for i, pt in enumerate(input_point):
            print(f"  Point {i+1}: {pt}")
    else:
        # timeout or no click
        input_point = np.array([[500, 375]])
        print("No clicks detected, using default point (500, 375)")
except Exception as e:
    print(f"Error during point selection: {e}")
    input_point = np.array([[500, 375]])
plt.close()

input_label = np.ones(len(input_point), dtype=int)  # all positive points

# Display selected point
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

masks.shape  # (number_of_masks) x H x W

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, save_contours=True, contour_output_dir="contours", epsilon=10.0)
