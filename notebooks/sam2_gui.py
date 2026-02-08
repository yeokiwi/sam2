#!/usr/bin/env python3
"""
SAM2 Interactive GUI for Image Segmentation
A graphical interface to load images, select points, and predict masks using SAM2.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import torch
import threading
import csv
import time

# Try to import OpenCV for better marker drawing, but fall back to PIL
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Note: OpenCV not found. Using PIL for point drawing.")

# Set environment variable for MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check for SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("SAM2 not found. Please install it following the instructions at:")
    print("https://github.com/facebookresearch/sam2#installation")
    sys.exit(1)


class SAM2GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Interactive Segmentation")
        self.root.geometry("1200x800")
        
        # Variables
        self.image_path = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image for display
        self.tk_image = None        # ImageTk.PhotoImage
        self.points = []            # List of (x, y, label) where label: 1=positive, 0=negative
        self.masks = []             # List of mask arrays
        self.current_mask = None    # Currently displayed mask
        self.contours = []          # List of contour arrays [(x1,y1), (x2,y2), ...]
        self.current_contour_dir = None  # Directory for current image's contours
        
        # Display scaling and zoom
        self.display_scale = 1.0    # Scale factor for displayed image
        self.min_scale = 0.1        # Minimum zoom scale (10% of original)
        self.max_scale = 10.0       # Maximum zoom scale (1000% of original)
        self.zoom_step = 1.2        # Zoom factor per step
        self.display_offset_x = 0   # X offset for centered image
        self.display_offset_y = 0   # Y offset for centered image
        self.pan_start_x = 0        # For panning/dragging
        self.pan_start_y = 0
        self.is_panning = False
        
        # Model and predictor
        self.device = self.get_device()
        self.predictor = None
        self.model_loaded = False
        
        # GUI elements
        self.setup_ui()
        
        # Try to load model in background
        self.load_model_async()
    
    def get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device
    
    def load_model_async(self):
        """Load SAM2 model in a background thread."""
        def load():
            try:
                # Model paths (adjust as needed)
                sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
                model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
                
                if not os.path.exists(sam2_checkpoint):
                    messagebox.showerror("Model Not Found", 
                                        f"Checkpoint not found at:\n{sam2_checkpoint}\n"
                                        "Please download it from the SAM2 repository.")
                    return
                
                # Build model
                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
                self.predictor = SAM2ImagePredictor(sam2_model)
                self.model_loaded = True
                
                # Update UI
                self.root.after(0, lambda: self.status_label.config(
                    text="Model loaded successfully. Load an image to start.", 
                    foreground="green"))
                self.root.after(0, lambda: self.predict_button.config(state="normal"))
                
                # Configure CUDA settings
                if self.device.type == "cuda":
                    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                
            except Exception as e:
                error_msg = f"Failed to load model:\n{str(e)}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Model Load Error", error_msg))
                self.root.after(0, lambda: self.status_label.config(
                    text="Model load failed. Check console for details.", foreground="red"))
        
        # Show loading status
        self.status_label.config(text="Loading SAM2 model...", foreground="blue")
        self.predict_button.config(state="disabled")
        
        # Start thread
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def setup_ui(self):
        """Setup the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Buttons
        self.load_button = ttk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5)
        
        self.add_pos_button = ttk.Button(button_frame, text="Add Positive Point", 
                                         command=lambda: self.set_point_mode(1),
                                         state="disabled")
        self.add_pos_button.grid(row=0, column=1, padx=5)
        
        self.add_neg_button = ttk.Button(button_frame, text="Add Negative Point",
                                         command=lambda: self.set_point_mode(0),
                                         state="disabled")
        self.add_neg_button.grid(row=0, column=2, padx=5)
        
        self.predict_button = ttk.Button(button_frame, text="Predict Mask",
                                         command=self.predict_mask,
                                         state="disabled")
        self.predict_button.grid(row=0, column=3, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Clear Points",
                                       command=self.clear_points,
                                       state="disabled")
        self.clear_button.grid(row=0, column=4, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Mask",
                                      command=self.save_mask,
                                      state="disabled")
        self.save_button.grid(row=0, column=5, padx=5)
        
        self.plot_contours_button = ttk.Button(button_frame, text="Plot Contours",
                                               command=self.plot_contours,
                                               state="disabled")
        self.plot_contours_button.grid(row=0, column=6, padx=5)
        
        self.generate_contours_button = ttk.Button(button_frame, text="Generate Contours",
                                                   command=self.generate_contours,
                                                   state="disabled")
        self.generate_contours_button.grid(row=0, column=7, padx=5)
        
        # Zoom controls
        zoom_frame = ttk.Frame(button_frame)
        zoom_frame.grid(row=0, column=8, padx=20)
        
        self.zoom_out_button = ttk.Button(zoom_frame, text="-", width=3,
                                          command=self.zoom_out,
                                          state="disabled")
        self.zoom_out_button.grid(row=0, column=0, padx=2)
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=8)
        self.zoom_label.grid(row=0, column=1, padx=2)
        
        self.zoom_in_button = ttk.Button(zoom_frame, text="+", width=3,
                                         command=self.zoom_in,
                                         state="disabled")
        self.zoom_in_button.grid(row=0, column=2, padx=2)
        
        self.zoom_reset_button = ttk.Button(zoom_frame, text="Fit", width=4,
                                            command=self.zoom_reset,
                                            state="disabled")
        self.zoom_reset_button.grid(row=0, column=3, padx=2)
        
        # Point mode indicator
        self.point_mode = None
        self.mode_label = ttk.Label(button_frame, text="Mode: None")
        self.mode_label.grid(row=0, column=9, padx=20)
        
        # Image canvas
        canvas_frame = ttk.Frame(main_frame, relief="sunken", borderwidth=2)
        canvas_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Points listbox
        list_frame = ttk.LabelFrame(main_frame, text="Selected Points", padding="5")
        list_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.points_listbox = tk.Listbox(list_frame, height=5, width=40)
        self.points_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Loading model...", relief="sunken", anchor="w")
        self.status_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        status_frame.columnconfigure(0, weight=1)
    
    def set_point_mode(self, mode):
        """Set the point selection mode (1=positive, 0=negative)."""
        self.point_mode = mode
        color = "green" if mode == 1 else "red"
        label = "Positive" if mode == 1 else "Negative"
        self.mode_label.config(text=f"Mode: {label} Point", foreground=color)
    
    def load_image(self):
        """Load an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load and convert image
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert("RGB")
            
            # Reset points and masks
            self.points = []
            self.masks = []
            self.current_mask = None
            self.contours = []
            
            # Create unique directory for this image's contours
            image_name = os.path.splitext(os.path.basename(file_path))[0]
            # Remove any characters that might cause issues in directory names
            safe_image_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            if not safe_image_name:
                safe_image_name = "image_" + str(int(time.time() * 1000))
            
            self.current_contour_dir = os.path.join("contours", safe_image_name)
            os.makedirs(self.current_contour_dir, exist_ok=True)
            
            # Update display
            self.update_canvas()
            
            # Enable buttons
            self.add_pos_button.config(state="normal")
            self.add_neg_button.config(state="normal")
            self.clear_button.config(state="normal")
            self.save_button.config(state="disabled")
            self.plot_contours_button.config(state="normal")
            self.zoom_in_button.config(state="normal")
            self.zoom_out_button.config(state="normal")
            self.zoom_reset_button.config(state="normal")
            
            # Update status
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}", foreground="black")
            
            # Clear points listbox
            self.points_listbox.delete(0, tk.END)
            
            # Set image for predictor if model is loaded
            if self.model_loaded and self.predictor:
                image_np = np.array(self.original_image)
                self.predictor.set_image(image_np)
                self.status_label.config(text=f"Image set for prediction. Add points and click Predict Mask.", foreground="black")
            
            print(f"Contours for this image will be saved to: {self.current_contour_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    
    def canvas_click(self, event):
        """Handle canvas click for point selection."""
        if self.original_image is None or self.point_mode is None:
            return
        
        # Get canvas coordinates (already account for scrolling)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Adjust for image offset (centered display)
        display_x = canvas_x - self.display_offset_x
        display_y = canvas_y - self.display_offset_y
        
        # Check if click is within displayed image bounds
        if self.display_image:
            img_width, img_height = self.display_image.size
            
            if 0 <= display_x < img_width and 0 <= display_y < img_height:
                # Convert display coordinates back to original image coordinates
                if self.display_scale > 0:
                    original_x = display_x / self.display_scale
                    original_y = display_y / self.display_scale
                else:
                    original_x = display_x
                    original_y = display_y
                
                # Add point in original image coordinates
                self.points.append((original_x, original_y, self.point_mode))
                
                # Update listbox with original coordinates
                label_text = "Positive" if self.point_mode == 1 else "Negative"
                self.points_listbox.insert(tk.END, f"{label_text}: ({original_x:.1f}, {original_y:.1f})")
                
                # Update canvas
                self.update_canvas()
    
    def clear_points(self):
        """Clear all selected points."""
        self.points = []
        self.points_listbox.delete(0, tk.END)
        self.current_mask = None
        self.update_canvas()
    
    def predict_mask(self):
        """Predict mask based on selected points."""
        if not self.model_loaded or self.predictor is None:
            messagebox.showerror("Error", "Model not loaded yet.")
            return
        
        if not self.points:
            messagebox.showwarning("No Points", "Please add at least one point.")
            return
        
        # Extract coordinates and labels
        point_coords = np.array([[x, y] for x, y, _ in self.points])
        point_labels = np.array([label for _, _, label in self.points])
        
        # Ensure predictor has the image set
        if self.original_image is not None:
            image_np = np.array(self.original_image)
            self.predictor.set_image(image_np)
        
        # Show progress
        self.status_label.config(text="Predicting mask...", foreground="blue")
        self.root.update()
        
        # Run prediction in thread to avoid freezing UI
        def run_prediction():
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                
                # Sort by score
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                
                # Store all masks
                self.masks = masks
                
                # Use the best mask
                self.current_mask = masks[0]
                self.current_mask = 1 - self.current_mask
                # Update UI
                self.root.after(0, lambda: self.on_prediction_complete(scores[0]))
                
            except Exception as e:
                error_msg = f"Prediction failed:\n{str(e)}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Prediction Error", error_msg))
                self.root.after(0, lambda: self.status_label.config(
                    text="Prediction failed. Check console for details.", foreground="red"))
        
        thread = threading.Thread(target=run_prediction, daemon=True)
        thread.start()
    
    def on_prediction_complete(self, score):
        """Update UI after prediction completes."""
        self.status_label.config(text=f"Mask predicted successfully. Score: {score:.3f}", foreground="green")
        self.save_button.config(state="normal")
        self.generate_contours_button.config(state="normal")
        self.update_canvas()
    
    def save_mask(self):
        """Save the current mask to file and extract contours to CSV."""
        if self.current_mask is None:
            messagebox.showwarning("No Mask", "No mask to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Save mask as binary image
            mask_img = Image.fromarray((self.current_mask * 255).astype(np.uint8))
            mask_img.save(file_path)
            
            # Also save overlay if requested
            if messagebox.askyesno("Save Overlay", "Save overlay image with mask as well?"):
                overlay_path = file_path.replace(".png", "_overlay.png")
                if self.display_image:
                    self.display_image.save(overlay_path)
            
            # Extract and save contours if OpenCV is available
            if HAS_CV2:
                self.save_contours_to_csv()
            
            self.status_label.config(text=f"Mask saved to {os.path.basename(file_path)}", foreground="black")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save mask:\n{str(e)}")
    
    def save_contours_to_csv(self, contour_dir=None):
        """Extract contours from current mask and save to CSV files in specified directory."""
        if self.current_mask is None:
            return
        
        # Use provided directory or current image's contour directory
        if contour_dir is None:
            if self.current_contour_dir is None:
                # Fall back to generic contours directory
                contour_dir = "contours"
            else:
                contour_dir = self.current_contour_dir
        
        # Ensure contours directory exists
        os.makedirs(contour_dir, exist_ok=True)
        
        # Convert mask to uint8 for OpenCV
        mask_uint8 = (self.current_mask * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No contours found in mask.")
            return
        
        # Generate unique filename base using timestamp
        timestamp = int(time.time() * 1000)
        base_filename = f"contour_{timestamp}"
        
        # Save each contour to a separate CSV file
        saved_files = []
        for i, contour in enumerate(contours):
            # Simplify contour to reduce points (optional)
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Extract x, y coordinates
            points = approx.squeeze()
            if len(points.shape) == 1:
                # Single point contour (rare)
                continue
            
            # Create DataFrame
            import pandas as pd
            df = pd.DataFrame(points, columns=['x', 'y'])
            
            # Save to CSV
            contour_filename = f"{base_filename}_{i}.csv"
            contour_path = os.path.join(contour_dir, contour_filename)
            df.to_csv(contour_path, index=False)
            saved_files.append(contour_filename)
            
            print(f"Saved contour {i} with {len(points)} points to {contour_path}")
        
        if saved_files:
            # Update status
            dir_name = os.path.basename(contour_dir)
            status_text = f"Contours saved to {dir_name}/ ({len(saved_files)} files)"
            self.root.after(0, lambda: self.status_label.config(text=status_text, foreground="black"))
            print(f"Saved {len(saved_files)} contour files to {contour_dir}/")
        else:
            print("No valid contours to save.")
    
    def clean_contour_directory(self, contour_dir=None):
        """Remove all contour CSV files from the specified directory."""
        if contour_dir is None:
            if self.current_contour_dir is None:
                return 0
            contour_dir = self.current_contour_dir
        
        if not os.path.exists(contour_dir):
            return 0
        
        import glob
        csv_files = glob.glob(os.path.join(contour_dir, "contour_*.csv"))
        deleted_count = 0
        
        for csv_file in csv_files:
            try:
                os.remove(csv_file)
                print(f"Removed existing contour file: {os.path.basename(csv_file)}")
                deleted_count += 1
            except Exception as e:
                print(f"Error removing contour file {csv_file}: {e}")
        
        if deleted_count > 0:
            print(f"Cleaned {deleted_count} existing contour files from {contour_dir}")
        
        return deleted_count
    
    def generate_contours(self):
        """Generate simplified contours from the current mask and save to CSV files."""
        if self.current_mask is None:
            messagebox.showwarning("No Mask", "Please predict a mask first.")
            return
        
        if not HAS_CV2:
            messagebox.showerror("OpenCV Required", 
                                "OpenCV is required for contour generation.\n"
                                "Please install OpenCV: pip install opencv-python")
            return
        
        # Show progress
        self.status_label.config(text="Generating simplified contours...", foreground="blue")
        self.root.update()
        
        # Run contour generation in thread to avoid freezing UI
        def run_contour_generation():
            try:
                # First, clean up any existing contour files in the current image's directory
                deleted_count = self.clean_contour_directory()
                if deleted_count > 0:
                    print(f"Removed {deleted_count} previous contour files before generating new ones")
                
                # Call the existing contour saving method
                self.save_contours_to_csv()
                
                # Update UI on success
                self.root.after(0, lambda: self.on_contour_generation_complete())
                
            except Exception as e:
                error_msg = f"Contour generation failed:\n{str(e)}"
                print(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Contour Generation Error", error_msg))
                self.root.after(0, lambda: self.status_label.config(
                    text="Contour generation failed. Check console for details.", foreground="red"))
        
        thread = threading.Thread(target=run_contour_generation, daemon=True)
        thread.start()
    
    def on_contour_generation_complete(self):
        """Update UI after contour generation completes."""
        self.status_label.config(text="Simplified contours generated and saved to contours/ directory.", foreground="green")
        # Enable plot contours button since we now have contours
        self.plot_contours_button.config(state="normal")
    
    def plot_contours(self):
        """Load and plot simplified contours from CSV files in the current image's contour directory."""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        # Check if current image has a contour directory
        if self.current_contour_dir is None or not os.path.exists(self.current_contour_dir):
            messagebox.showwarning("No Contours", 
                                  f"No contour directory found for current image.\n"
                                  f"Please generate contours first using 'Generate Contours' button.")
            return
        
        # Look for contour CSV files in the current image's contour directory
        import glob
        csv_files = glob.glob(os.path.join(self.current_contour_dir, "contour_*.csv"))
        if not csv_files:
            messagebox.showwarning("No Contours", 
                                  f"No contour CSV files found in:\n{self.current_contour_dir}\n"
                                  f"Please generate contours first using 'Generate Contours' button.")
            return
        
        print(f"Found {len(csv_files)} contour files in {self.current_contour_dir}")
        
        # Load and parse contour data
        self.contours = []
        for csv_file in csv_files:
            try:
                points = []
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) >= 2:
                            x, y = float(row[0]), float(row[1])
                            points.append((x, y))
                if points:
                    self.contours.append(points)
                    print(f"Loaded contour from {os.path.basename(csv_file)} with {len(points)} points")
            except Exception as e:
                print(f"Error loading contour from {csv_file}: {e}")
        
        if not self.contours:
            messagebox.showwarning("No Valid Contours", 
                                  "No valid contours could be loaded from CSV files.\n"
                                  "The contour files may be empty or corrupted.")
            return
        
        # Update status and redraw canvas
        dir_name = os.path.basename(self.current_contour_dir)
        self.status_label.config(text=f"Loaded {len(self.contours)} contours from {dir_name}. Displaying green lines and markers.", 
                                foreground="green")
        self.update_canvas()
    
    def zoom_in(self):
        """Zoom in on the image."""
        if self.original_image is None:
            return
        
        # Calculate new scale
        new_scale = self.display_scale * self.zoom_step
        
        # Limit maximum zoom
        if new_scale > self.max_scale:
            new_scale = self.max_scale
            self.status_label.config(text=f"Maximum zoom reached ({int(new_scale*100)}%)", foreground="orange")
        else:
            self.status_label.config(text=f"Zooming in ({int(new_scale*100)}%)", foreground="blue")
        
        # Update scale and redraw
        self.display_scale = new_scale
        self.update_canvas()
        self.update_zoom_label()
    
    def zoom_out(self):
        """Zoom out from the image."""
        if self.original_image is None:
            return
        
        # Calculate new scale
        new_scale = self.display_scale / self.zoom_step
        
        # Limit minimum zoom
        if new_scale < self.min_scale:
            new_scale = self.min_scale
            self.status_label.config(text=f"Minimum zoom reached ({int(new_scale*100)}%)", foreground="orange")
        else:
            self.status_label.config(text=f"Zooming out ({int(new_scale*100)}%)", foreground="blue")
        
        # Update scale and redraw
        self.display_scale = new_scale
        self.update_canvas()
        self.update_zoom_label()
    
    def zoom_reset(self):
        """Reset zoom to fit image to canvas."""
        if self.original_image is None:
            return
        
        # Reset to fit-to-canvas scale
        self.display_scale = 1.0  # Will be recalculated in update_canvas to fit
        
        # Update status
        self.status_label.config(text="Zoom reset to fit canvas", foreground="green")
        
        # Update display
        self.update_canvas()
        self.update_zoom_label()
    
    def update_zoom_label(self):
        """Update the zoom percentage label."""
        if self.original_image is None:
            return
        
        percentage = int(self.display_scale * 100)
        self.zoom_label.config(text=f"{percentage}%")
        
        # Also update status with current zoom info
        if self.display_scale >= 1.0:
            zoom_text = f"Zoom: {percentage}% (enlarged)"
        else:
            zoom_text = f"Zoom: {percentage}% (reduced)"
        
        # Only update if we're not in the middle of another operation
        current_status = self.status_label.cget("text")
        if "Zoom:" not in current_status and "Zooming" not in current_status:
            self.status_label.config(text=f"{current_status} | {zoom_text}")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming."""
        if self.original_image is None:
            return
        
        # Determine zoom direction
        # Windows: event.delta positive for up, negative for down
        # Linux: event.num == 4 for up, 5 for down
        zoom_in = False
        if event.num == 4:  # Linux scroll up
            zoom_in = True
        elif event.num == 5:  # Linux scroll down
            zoom_in = False
        elif hasattr(event, 'delta'):
            # Windows
            zoom_in = (event.delta > 0)
        else:
            return
        
        # Apply zoom
        if zoom_in:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def update_canvas(self):
        """Update the canvas with the current image and overlays."""
        if self.original_image is None:
            return
        
        # Get screen/canvas dimensions for scaling
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Ensure minimum canvas dimensions
        if canvas_width < 100 or canvas_height < 100:
            canvas_width = 800
            canvas_height = 600
            # Update canvas config to ensure it has a size
            self.canvas.config(width=canvas_width, height=canvas_height)
        
        # Target half screen resolution
        target_width = max(100, canvas_width)
        target_height = max(100, canvas_height)
        
        # Original image size
        orig_width, orig_height = self.original_image.size
        
        # If we're in zoom reset mode or initial load, calculate fit-to-canvas scale
        if self.display_scale == 1.0:  # Reset signal
            # Calculate scale to fit within canvas while maintaining aspect ratio
            width_scale = target_width / orig_width
            height_scale = target_height / orig_height
            self.display_scale = min(width_scale, height_scale, 1.0)  # Don't scale up beyond original
        
        # Ensure scale is within zoom bounds
        if self.display_scale < self.min_scale:
            self.display_scale = self.min_scale
        elif self.display_scale > self.max_scale:
            self.display_scale = self.max_scale
        
        # Ensure scale is positive and reasonable
        if self.display_scale <= 0:
            self.display_scale = min(1.0, target_width / orig_width, target_height / orig_height)
        
        # Calculate new size with minimum dimensions
        new_width = max(1, int(orig_width * self.display_scale))
        new_height = max(1, int(orig_height * self.display_scale))
        
        # Calculate centering offsets
        self.display_offset_x = max(0, (canvas_width - new_width) // 2)
        self.display_offset_y = max(0, (canvas_height - new_height) // 2)
        
        # Create a copy for display and resize
        display_img = self.original_image.copy()
        if self.display_scale != 1.0:
            display_img = display_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Draw points if any (points are in original image coordinates, need to scale)
        if self.points:
            if HAS_CV2:
                # Use OpenCV for better marker drawing
                img_np = np.array(display_img)
                
                for x, y, label in self.points:
                    # Scale point coordinates to display size
                    display_x = int(x * self.display_scale)
                    display_y = int(y * self.display_scale)
                    color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for positive, Red for negative
                    # Draw a star-like shape
                    cv2.drawMarker(img_np, (display_x, display_y), color, markerType=cv2.MARKER_STAR, 
                                  markerSize=4, thickness=4, line_type=cv2.LINE_AA)
                
                # Convert back to PIL
                display_img = Image.fromarray(img_np)
            else:
                # Use PIL for drawing points (fallback)
                draw = ImageDraw.Draw(display_img)
                for x, y, label in self.points:
                    # Scale point coordinates to display size
                    display_x = x * self.display_scale
                    display_y = y * self.display_scale
                    color = "green" if label == 1 else "red"
                    # Draw a circle with an 'x' inside
                    radius = 10
                    # Draw outer circle
                    draw.ellipse([display_x - radius, display_y - radius, 
                                 display_x + radius, display_y + radius], 
                                 outline=color, width=3)
                    # Draw 'x' inside
                    draw.line([display_x - radius + 3, display_y - radius + 3, 
                              display_x + radius - 3, display_y + radius - 3], 
                              fill=color, width=2)
                    draw.line([display_x - radius + 3, display_y + radius - 3, 
                              display_x + radius - 3, display_y - radius + 3], 
                              fill=color, width=2)
        
        # Draw contours if any (contours are in original image coordinates, need to scale)
        if self.contours:
            draw = ImageDraw.Draw(display_img)
            for contour_points in self.contours:
                if len(contour_points) < 2:
                    continue
                
                # Scale contour points to display size
                display_points = []
                for x, y in contour_points:
                    display_x = x * self.display_scale
                    display_y = y * self.display_scale
                    display_points.append((display_x, display_y))
                
                # Close the contour by connecting last point to first
                if len(display_points) > 2:
                    display_points.append(display_points[0])
                
                # Draw green lines connecting contour points
                for i in range(len(display_points) - 1):
                    x1, y1 = display_points[i]
                    x2, y2 = display_points[i + 1]
                    draw.line([x1, y1, x2, y2], fill="green", width=2)
                
                # Draw green markers at each contour point
                for x, y in display_points[:-1]:  # Skip the duplicated closing point
                    radius = 3
                    draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                                 fill="green", outline="green")
        
        # Draw mask if available
        if self.current_mask is not None:
            # Create mask overlay
            mask_np = self.current_mask.astype(np.uint8) * 255
            mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = mask_np  # Red channel
            
            # Convert mask to PIL and resize to display size
            mask_img = Image.fromarray(mask_rgb).convert("RGBA")
            if self.display_scale != 1.0:
                mask_img = mask_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            
            # Create transparency
            alpha = np.zeros((mask_img.height, mask_img.width), dtype=np.uint8)
            alpha_np = np.array(mask_img.convert("L"))
            alpha[alpha_np > 0] = 128  # 50% opacity
            alpha_img = Image.fromarray(alpha, mode='L')
            mask_img.putalpha(alpha_img)
            
            # Composite with original image
            display_img = display_img.convert("RGBA")
            display_img = Image.alpha_composite(display_img, mask_img)
        
        # Convert to PhotoImage
        self.display_image = display_img
        self.tk_image = ImageTk.PhotoImage(display_img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, 
                                anchor="nw", image=self.tk_image)
        
        # Update scroll region to match canvas size (not image size)
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # Update zoom label
        self.update_zoom_label()
    

def main():
    """Main entry point."""
    root = tk.Tk()
    app = SAM2GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()