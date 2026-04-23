#import tkinter as tk
#from tkinter import filedialog, messagebox
import os
#import sys
import argparse
#import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path

#import flip_evaluator as flip
#import lpips

# python image_comparison_cmd.py -r "E:\temp\jpeg_test\wiese\original.jpg" -i "E:\temp\jpeg_test\wiese\jpeg_70.jpg"

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

parser = argparse.ArgumentParser(description="Example script with -o option")

#parser.add_argument('-o', '--output', type=str, help='Path to output file')
parser.add_argument('-r', '--reference', type=str, help='reference image')
#parser.add_argument('-i', '--image', type=str, help='test image')
parser.add_argument('-i', nargs='+', help='test images')

# Parse the arguments
args = parser.parse_args()

# Access the argument
print("Reference:   ", args.reference)
print("Test Images: ", args.i)

img_reference = load_image(args.reference)
#img_test      = load_image(args.i[0])

testImages = []
for path in args.i:
    image = load_image(path)
    testImages.append(image)


for i in range(len(testImages)):
	image = testImages[i]
	filename = Path(args.i[i]).name
	psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
	print(f"{filename:20} PSNR: {psnr_value:.2f}")

# Determine a valid win_size for SSIM based on the source image dimensions
# min_dim = min(img_test.shape[:2])
# default_win_size = 7
# win_size = default_win_size
# if default_win_size > min_dim:
# 	win_size = min_dim if (min_dim % 2 == 1) else min_dim - 1
# 	if win_size < 3:
# 		win_size = 3  # SSIM requires win_size >= 3

# ssim_value = structural_similarity(
# 	img_reference, img_test,
# 	channel_axis=2,
# 	data_range=255,
# 	win_size=win_size
# )


# print("SSIM: ", round(ssim_value, 2))