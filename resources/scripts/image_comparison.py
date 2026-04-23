import tkinter as tk
from tkinter import filedialog, messagebox
import os
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image, ImageTk
import numpy as np
import flip_evaluator as flip
import lpips

# Global variables
reference_files = []
photo_images = []  # To keep references to PhotoImage objects

def format_size(bytes_size):
    """Convert a file size in bytes to a human-readable string."""
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def load_image(path):
    """
    Load an image from the given path using PIL.
    Returns an RGB numpy array.
    """
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def select_source():
    filename = filedialog.askopenfilename(
        title="Select Source Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.dds")]
    )
    if filename:
        source_path.set(filename)

def select_references():
    global reference_files
    files = filedialog.askopenfilenames(
        title="Select Reference Images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.dds")]
    )
    if files:
        reference_files = list(files)
        ref_label.config(text="\n".join([os.path.basename(f) for f in reference_files]))
    else:
        ref_label.config(text="No reference images selected")

def calculate():
    global photo_images
    # Clear previous results from the results frame and clear photo references.
    for widget in result_frame.winfo_children():
        widget.destroy()
    photo_images = []

    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    src_path = source_path.get()
    if not src_path:
        messagebox.showerror("Error", "Please select a source image.")
        return

    if not reference_files:
        messagebox.showerror("Error", "Please select at least one reference image.")
        return

    # Load the source image
    source_img = load_image(src_path)
    if source_img is None:
        messagebox.showerror("Error", f"Unable to load source image:\n{src_path}")
        return

    # Get file size for source image
    try:
        source_size = os.path.getsize(src_path)
        source_size_str = format_size(source_size)
    except Exception:
        source_size_str = "Unknown"

    # Display source image info as a header
    header_text = f"Source Image: {os.path.basename(src_path)} (File Size: {source_size_str})"
    header_label = tk.Label(result_frame, text=header_text, font=("Arial", 10, "bold"))
    header_label.pack(pady=5)

    # Prepare the source tensor for LPIPS (normalize to [-1, 1])
    source_tensor = torch.from_numpy(source_img).permute(2, 0, 1).unsqueeze(0).float()
    source_tensor = (source_tensor / 127.5) - 1.0

    # Determine a valid win_size for SSIM based on the source image dimensions
    min_dim = min(source_img.shape[:2])
    default_win_size = 7
    if default_win_size > min_dim:
        win_size = min_dim if (min_dim % 2 == 1) else min_dim - 1
        if win_size < 3:
            win_size = 3  # SSIM requires win_size >= 3
    else:
        win_size = default_win_size
        row_frame = tk.Frame(result_frame, borderwidth=1, relief="groove")
        row_frame.pack(fill="x", padx=5, pady=5)

        # Create and pack the 16x16 center region, then resize it for display
        if source_img is not None:
            h, w, _ = source_img.shape
            half_crop = 8  # Half of 16
            center_y = h // 2
            center_x = w // 2
            # Crop a 16x16 region from the center (ensuring indices are within bounds)
            cropped_region = source_img[
                max(center_y - half_crop, 0):min(center_y + half_crop, h),
                max(center_x - half_crop, 0):min(center_x + half_crop, w),
                :
            ]
            pil_crop = Image.fromarray(cropped_region)
            # Resize the 16x16 region to a larger display size (e.g., 64x64)
            display_size = (128, 128)
            resized_crop = pil_crop.resize(display_size, Image.NEAREST)
            photo = ImageTk.PhotoImage(resized_crop)
            photo_images.append(photo)  # keep a reference
            img_label = tk.Label(row_frame, image=photo)
            img_label.pack(side="left", padx=5, pady=5)
            stats_label = tk.Label(row_frame, text="source image", justify="left", anchor="w", font=("Arial", 10))
            stats_label.pack(side="left", padx=10)
    # Process each reference image
    for ref in reference_files:
        # Get file size for reference image
        try:
            ref_size = os.path.getsize(ref)
            ref_size_str = format_size(ref_size)
        except Exception:
            ref_size_str = "Unknown"

        ref_img = load_image(ref)
        if ref_img is None:
            stats_text = f"{os.path.basename(ref)}: Unable to load image. (File Size: {ref_size_str})"
        elif source_img.shape != ref_img.shape:
            stats_text = f"{os.path.basename(ref)}: Dimension mismatch with source image. (File Size: {ref_size_str})"
        else:
            # Calculate PSNR and SSIM
            psnr_value = peak_signal_noise_ratio(source_img, ref_img, data_range=255)
            ssim_value = structural_similarity(
                source_img, ref_img,
                channel_axis=2,
                data_range=255,
                win_size=win_size
            )
            # Prepare FLIP metric (images in [0, 1])
            source_img_norm = source_img.astype(np.float32) / 255.0
            ref_img_norm = ref_img.astype(np.float32) / 255.0
            flipErrorMap, meanFLIPError, parameters = flip.evaluate(source_img_norm, ref_img_norm, "LDR")

            # Prepare LPIPS distance using the Alex network
            ref_tensor = torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0).float()
            ref_tensor = (ref_tensor / 127.5) - 1.0
            d = loss_fn_alex(source_tensor, ref_tensor)

            stats_text = (
                f"{os.path.basename(ref)} (File Size: {ref_size_str}):\n"
                f"  PSNR = {psnr_value:.2f} dB\n"
                f"  SSIM = {ssim_value:.4f}\n"
                f"  FLIP = {meanFLIPError:.4f}\n"
                f"  LPIPS = {d.item():.4f}"
            )

        # Create a frame for each reference image's result
        row_frame = tk.Frame(result_frame, borderwidth=1, relief="groove")
        row_frame.pack(fill="x", padx=5, pady=5)

        # Create and pack the 16x16 center region, then resize it for display
        if ref_img is not None:
            h, w, _ = ref_img.shape
            half_crop = 8  # Half of 16
            center_y = h // 2
            center_x = w // 2
            # Crop a 16x16 region from the center (ensuring indices are within bounds)
            cropped_region = ref_img[
                max(center_y - half_crop, 0):min(center_y + half_crop, h),
                max(center_x - half_crop, 0):min(center_x + half_crop, w),
                :
            ]
            pil_crop = Image.fromarray(cropped_region)
            # Resize the 16x16 region to a larger display size (e.g., 64x64)
            display_size = (128, 128)
            resized_crop = pil_crop.resize(display_size, Image.NEAREST)
            photo = ImageTk.PhotoImage(resized_crop)
            photo_images.append(photo)  # keep a reference
            img_label = tk.Label(row_frame, image=photo)
            img_label.pack(side="left", padx=5, pady=5)
        else:
            img_label = tk.Label(row_frame, text="[No Image]", width=15, height=7, bg="gray")
            img_label.pack(side="left", padx=5, pady=5)

        # Create and pack the label with the stats
        stats_label = tk.Label(row_frame, text=stats_text, justify="left", anchor="w", font=("Arial", 10))
        stats_label.pack(side="left", padx=10)

# Create the main window
root = tk.Tk()
root.title("Image Quality Comparison Calculator")

# Variable for storing the source image path
source_path = tk.StringVar()

# Main frame
frame = tk.Frame(root, padx=10, pady=10)
frame.grid(row=0, column=0)

# Source image selection
source_button = tk.Button(frame, text="Select Source Image", command=select_source)
source_button.grid(row=0, column=0, sticky="w", padx=5, pady=5)
source_label = tk.Label(frame, textvariable=source_path, wraplength=400, justify="left")
source_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)

# Reference images selection
ref_button = tk.Button(frame, text="Select Reference Images", command=select_references)
ref_button.grid(row=1, column=0, sticky="w", padx=5, pady=5)
ref_label = tk.Label(frame, text="No reference images selected", wraplength=400, justify="left")
ref_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)

# Calculate button
calc_button = tk.Button(frame, text="Calculate", command=calculate)
calc_button.grid(row=2, column=0, columnspan=2, pady=10)

# Results frame (to show the upscaled 16x16 cutout with stats)
result_frame = tk.Frame(frame, borderwidth=2, relief="sunken")
result_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
