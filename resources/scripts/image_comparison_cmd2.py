#import tkinter as tk
#from tkinter import filedialog, messagebox
import os
import shutil
#import sys
import argparse
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pillow_jxl
import pillow_avif
# import pillow_heif
from PIL import Image, ImageTk, features
import numpy as np
import flip_evaluator as flip
import lpips
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import namedtuple

#from pillow_heif import register_heif_opener
#register_heif_opener()

# dense
bcFormats = ["bc1", "bc7"]
astcFormats = ["astc_ldr_4x4", "astc_ldr_6x6", "astc_ldr_8x8", "astc_ldr_10x10", "astc_ldr_12x12"]
qualityLevels = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
# qualityLevels = [40, 60, 80]
# qualityLevels = list(range(40, 96))
jpegQualityLevels = qualityLevels
jpegXLQualityLevels = qualityLevels


# bcFormats = ["bc1"]
# astcFormats = ["astc_ldr_4x4", "astc_ldr_6x6", "astc_ldr_8x8", "astc_ldr_10x10", "astc_ldr_12x12"]
# qualityLevels = [40, 60,90]
# jpegQualityLevels = qualityLevels
# jpegXLQualityLevels = qualityLevels

# sparse
# bcFormats = []
# astcFormats = []
# jpegQualityLevels = [20, 40, 60]
# jpegXLQualityLevels = jpegQualityLevels
# qualityLevels = jpegQualityLevels


nvcompressCuality = "-production" # -fast, -production, -highest

ncompressPath = "C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools/nvcompress.exe"
nvdecompressPath = "C:/Program Files/NVIDIA Corporation/NVIDIA Texture Tools/nvdecompress.exe"
cjpegliPath = "/home/mschuetz/dev/jpegli/build/tools/cjpegli"

loss_fn_alex = lpips.LPIPS(net='alex')

# check what jpeg encoder PIL uses
#print("PIL JPEG support:")
#print(features.check("jpg"))
#print(features.version("jpg"))
print("PIL modules: ")
for item in features.get_supported_modules():
	version = features.version_module(item)
	print(f"- {item}, {version}")
print("PIL codecs: ")
for item in features.get_supported_codecs():
	version = features.version_codec(item)
	print(f"- {item}, {version}")
print("PIL features: ")
for item in features.get_supported_features():
	version = features.version_feature(item)
	print(f"- {item}, {version}")

#print(f"PIL version: {PIL.__version__}")

def to_wsl_path(win_path):
	out = subprocess.check_output(["wsl", "wslpath", "-a", "-u", win_path], text=True)
	return out.strip()


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
parser.add_argument('-o', '--output', type=str, help='output directory')
parser.add_argument('-i', nargs='+', help='list of images')

# Parse the arguments
args = parser.parse_args()

outputDirectory = args.output

# Convert the source image into various compressed image formats, e.g. various jpeg levels, astc, bc, etc.
def convertImages(paths):

	for sourcePath in paths:

		filename = Path(sourcePath).name
		os.makedirs(f"{outputDirectory}/{filename}", exist_ok=True)
		shutil.copy(sourcePath, f"{outputDirectory}/{filename}/{filename}")

		# convert to various compressed formats
		# .\nvcompress.exe -color -nomips -production -bc1 <source> <target>
		for format in (astcFormats + bcFormats):
			print(f"compressing to format: {format}")
			result = subprocess.run(
				[
					ncompressPath,
					"-color", "-nomips", nvcompressCuality,
					f"-{format}",
					sourcePath,
					f"{outputDirectory}/{filename}/{format}.dds"
				], 
				capture_output=True, text=True
			)
			print("STDOUT:", result.stdout)
			print("STDERR:", result.stderr)

		# now convert compressed formats to png to 
		# retain all the compression formats but making the images accessible
		#.\nvdecompress.exe -format png <source> <target>
		for format in (astcFormats + bcFormats):
			print(f"transcoding {format} to png")
			result = subprocess.run(
				[
					nvdecompressPath,
					"-format", "png",
					f"{outputDirectory}/{filename}/{format}.dds",
					f"{outputDirectory}/{filename}/{format}.png"
				], 
				capture_output=True, text=True
			)
			print("STDOUT:", result.stdout)
			print("STDERR:", result.stderr)


		# create JPEGs in various quality levels
		image = Image.open(sourcePath)
		for quality in jpegQualityLevels:
			image.convert("RGB").save(f"{outputDirectory}/{filename}/jpegturbo_{quality}.jpg", "JPEG", quality=quality, optimize=True, subsampling="4:2:0")

		# create AVIF
		for quality in qualityLevels:
			image.convert("RGB").save(f"{outputDirectory}/{filename}/AVIF_{quality}.avif", "AVIF", quality=quality)

		# # create HEIF
		# for quality in qualityLevels:
		# 	image.convert("RGB").save(f"{outputDirectory}/{filename}/HEIC_{quality}.heic", "HEIF", quality=quality)

		# create JPEGs using jpegli
		for quality in jpegQualityLevels:
			wslOutDir = to_wsl_path(outputDirectory)

			result = subprocess.run(
				[
					"wsl", cjpegliPath,
					to_wsl_path(sourcePath),
					f"{wslOutDir}/{filename}/jpegli_{quality}.jpg",
					"-q", f"{quality}",
					"--chroma_subsampling=420",
					"-p", "0"
				], 
				capture_output=True, text=True
			)
			print("STDOUT:", result.stdout)
			print("STDERR:", result.stderr)

		# create JPEGs using jpegli with XYB
		# for quality in jpegQualityLevels:
		# 	wslOutDir = to_wsl_path(outputDirectory)

		# 	result = subprocess.run(
		# 		[
		# 			"wsl", cjpegliPath,
		# 			to_wsl_path(sourcePath),
		# 			f"{wslOutDir}/{filename}/jpegli_{quality}_xyb.jpg",
		# 			"-q", f"{quality}",
		# 			"--chroma_subsampling=420",
		# 			"-p", "0",
		# 			"--xyb"
		# 		], 
		# 		capture_output=True, text=True
		# 	)
		# 	print("STDOUT:", result.stdout)
		# 	print("STDERR:", result.stderr)

		# create JPEG XL's
		for quality in jpegXLQualityLevels:
			image.save(f"{outputDirectory}/{filename}/jpegXL_{quality}.jxl", quality=quality, lossless_jpeg=False, effort=9)

def runFlip(img_reference, img_test):
	img_reference_norm = img_reference.astype(np.float32) / 255.0
	img_test_norm = img_test.astype(np.float32) / 255.0
	flipErrorMap, meanFLIPError, parameters = flip.evaluate(img_reference_norm, img_test_norm, "LDR")

	return meanFLIPError

def runLpips(img_reference, img_test):
	ref_tensor = torch.from_numpy(img_reference).permute(2, 0, 1).unsqueeze(0).float()
	ref_tensor = (ref_tensor / 127.5) - 1.0

	test_tensor = torch.from_numpy(img_test).permute(2, 0, 1).unsqueeze(0).float()
	test_tensor = (test_tensor / 127.5) - 1.0

	d = loss_fn_alex(ref_tensor, test_tensor)

	return d.item()

def compareImages(paths):

	Record = namedtuple("Record", ["algorithm", "label", "bpp", "psnr", "ssim", "flip", "lpips"])

	for sourcePath in paths:
		filename = Path(sourcePath).name
		img_reference = load_image(sourcePath)
		records = []

		# ASTC
		for format in astcFormats:
			filesize = os.path.getsize(f"{outputDirectory}/{filename}/{format}.dds")
			image = load_image(f"{outputDirectory}/{filename}/{format}.png")
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			label = format.replace("astc_ldr_", "")
			records.append(Record("ASTC", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# BC
		for format in bcFormats:
			filesize = os.path.getsize(f"{outputDirectory}/{filename}/{format}.dds")
			image = load_image(f"{outputDirectory}/{filename}/{format}.png")
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			records.append(Record("BC", format, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# JPEG 
		for quality in jpegQualityLevels:
			compressedPath = f"{outputDirectory}/{filename}/jpegturbo_{quality}.jpg"
			filesize = os.path.getsize(compressedPath)
			image = load_image(compressedPath)
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			format = f"JPEG {quality}%"
			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			label = f"{quality}"
			if quality != min(jpegQualityLevels) and quality != max(jpegQualityLevels):
				label = ""
			records.append(Record("JPEG (PIL)", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# JPEGLI
		for quality in jpegQualityLevels:
			compressedPath = f"{outputDirectory}/{filename}/jpegli_{quality}.jpg"
			filesize = os.path.getsize(compressedPath)
			image = load_image(compressedPath)
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			format = f"JPEGLI {quality}%"
			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			label = f"{quality}"
			if quality != min(jpegQualityLevels) and quality != max(jpegQualityLevels):
				label = ""
			records.append(Record("JPEGLI", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# JPEG XL
		for quality in jpegXLQualityLevels:
			compressedPath = f"{outputDirectory}/{filename}/jpegXL_{quality}.jxl"
			filesize = os.path.getsize(compressedPath)
			image = load_image(compressedPath)
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			format = f"JPEG XL {quality}%"
			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			label = f"{quality}"
			if quality != min(jpegQualityLevels) and quality != max(jpegQualityLevels):
				label = ""
			records.append(Record("JPEG XL", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# AVIF
		for quality in qualityLevels:
			compressedPath = f"{outputDirectory}/{filename}/AVIF_{quality}.avif"
			filesize = os.path.getsize(compressedPath)
			image = load_image(compressedPath)
			width, height, channels = image.shape

			bytesPerPoint = filesize / (width * height)
			bpp = bytesPerPoint * 8
			psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
			ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
			flip_value = runFlip(img_reference, image)
			lpips_value = runLpips(img_reference, image)

			format = f"AVIF {quality}%"
			print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

			label = f"{quality}"
			if quality != min(jpegQualityLevels) and quality != max(jpegQualityLevels):
				label = ""
			records.append(Record("AVIF", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))

		# # HEIC
		# for quality in qualityLevels:
		# 	compressedPath = f"{outputDirectory}/{filename}/PIL_HEIC_{quality}.heic"
		# 	filesize = os.path.getsize(compressedPath)
		# 	image = load_image(compressedPath)
		# 	width, height, channels = image.shape

		# 	bytesPerPoint = filesize / (width * height)
		# 	bpp = bytesPerPoint * 8
		# 	psnr_value = peak_signal_noise_ratio(img_reference, image, data_range=255)
		# 	ssim_value = structural_similarity(img_reference, image, data_range=255, channel_axis=2)
		# 	flip_value = runFlip(img_reference, image)
		# 	lpips_value = runLpips(img_reference, image)

		# 	format = f"HEIC {quality}%"
		# 	print(f"{filename:20} format: {format:14}   bpp: {bpp:.2f}   PSNR↑: {psnr_value:.2f}    SSIM↑: {ssim_value:.4f}    FLIP↓: {flip_value:.4f}    LPIPS↓: {lpips_value:.4f}")

		# 	label = f"{quality}"
		# 	if quality != min(jpegQualityLevels) and quality != max(jpegQualityLevels):
		# 		label = ""
		# 	records.append(Record("HEIC", label, bpp, psnr_value, ssim_value, flip_value, lpips_value))
		
		#####################################
		### FIGURE 1 - PSNR
		#####################################

		# first plot of a reference image clears all previous plots
		plt.close("all")
		plt.clf()

		plt.figure(1)
		plt.xlim( 0, 4.5)

		algorithms = list(dict.fromkeys([t.algorithm for t in records]))
		print(algorithms)

		for algorithm in algorithms:
			entries = [t for t in records if t.algorithm == algorithm]
			bpps = [t.bpp for t in entries]
			psnrs = [t.psnr for t in entries]
			labels = [t.label for t in entries]

			plt.plot(bpps, psnrs, marker='o', linestyle='-', label=algorithm, clip_on=True)

			for i in range(len(bpps)):
				plt.text(bpps[i] + 0.0, psnrs[i] + 0.0, labels[i], fontsize=8, clip_on=True)
		
		# Axis labels and title
		plt.xlabel("bits per pixel")
		# plt.ylabel("PSNR↑")
		# plt.title(f"Compression - {filename}")

		plt.grid(True)
		plt.legend()
		
		plt.savefig(f"{outputDirectory}/{filename}/plot_psnr.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
		# plt.show()

		#####################################
		### FIGURE 2 - SSIM
		#####################################
		
		plt.figure(2)
		plt.xlim( 0, 4.5)

		algorithms = list(dict.fromkeys([t.algorithm for t in records]))
		print(algorithms)

		for algorithm in algorithms:
			entries = [t for t in records if t.algorithm == algorithm]
			bpps = [t.bpp for t in entries]
			ssims = [t.ssim for t in entries]
			labels = [t.label for t in entries]

			plt.plot(bpps, ssims, marker='o', linestyle='-', label=algorithm, clip_on=True)

			for i in range(len(bpps)):
				plt.text(bpps[i] + 0.0, ssims[i] + 0.0, labels[i], fontsize=8, clip_on=True)
		
		# Axis labels and title
		plt.xlabel("bits per pixel")
		# plt.ylabel("SSIM↑")
		# plt.title(f"Compression - {filename}")

		plt.grid(True)

		plt.legend()
		
		plt.savefig(f"{outputDirectory}/{filename}/plot_ssim.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
		# plt.show()

		#####################################
		### FIGURE 3 - FLIP
		#####################################
		
		plt.figure(3)
		plt.xlim( 0, 4.5)

		algorithms = list(dict.fromkeys([t.algorithm for t in records]))
		print(algorithms)

		for algorithm in algorithms:
			entries = [t for t in records if t.algorithm == algorithm]
			bpps = [t.bpp for t in entries]
			flips = [t.flip for t in entries]
			labels = [t.label for t in entries]

			plt.plot(bpps, flips, marker='o', linestyle='-', label=algorithm, clip_on=True)

			for i in range(len(bpps)):
				plt.text(bpps[i] + 0.0, flips[i] + 0.0, labels[i], fontsize=8, clip_on=True)
		
		# Axis labels and title
		plt.xlabel("bits per pixel")
		# plt.ylabel("FLIP↓")
		# plt.title(f"Compression - {filename}")

		plt.grid(True)

		plt.legend()
		
		plt.savefig(f"{outputDirectory}/{filename}/plot_flip.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

		#####################################
		### FIGURE 4 - LPIPS
		#####################################
		
		plt.figure(4)
		plt.xlim( 0, 4.5)

		algorithms = list(dict.fromkeys([t.algorithm for t in records]))
		print(algorithms)

		for algorithm in algorithms:
			entries = [t for t in records if t.algorithm == algorithm]
			bpps = [t.bpp for t in entries]
			lpipss = [t.lpips for t in entries]
			labels = [t.label for t in entries]

			plt.plot(bpps, lpipss, marker='o', linestyle='-', label=algorithm, clip_on=True)

			for i in range(len(bpps)):
				plt.text(bpps[i] + 0.0, lpipss[i] + 0.0, labels[i], fontsize=8, clip_on=True)
		
		# Axis labels and title
		plt.xlabel("bits per pixel")
		# plt.ylabel("LPIPS↓")
		# plt.title(f"Compression - {filename}")

		plt.grid(True)

		plt.legend()
		
		plt.savefig(f"{outputDirectory}/{filename}/plot_lpips.png", dpi=300, bbox_inches="tight", pad_inches=0.05)



		# plt.show()
		





# convertImages(args.i)
compareImages(args.i)

# python image_comparison_cmd2.py -o E:\temp\jpeg_test\out -i "E:\temp\jpeg_test\PavingStones126A_4K-PNG_Color.png" 

 




