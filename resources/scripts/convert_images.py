#import tkinter as tk
#from tkinter import filedialog, messagebox
import os
import shutil
#import sys
import argparse
# import torch
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pillow_jxl
import pillow_avif
# import pillow_heif
from PIL import Image, ImageTk, features
# import numpy as np
# import flip_evaluator as flip
# import lpips
from pathlib import Path
import subprocess
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from collections import namedtuple

#from pillow_heif import register_heif_opener
#register_heif_opener()

# dense
bcFormats = ["bc1", "bc7"]
astcFormats = ["astc_ldr_4x4", "astc_ldr_6x6", "astc_ldr_8x8", "astc_ldr_10x10", "astc_ldr_12x12"]
qualityLevels = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
# qualityLevels = [40, 60, 80]
# qualityLevels = list(range(40, 96))
jpegQualityLevels = qualityLevels
jpegXLQualityLevels = qualityLevels

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
ffmpegPath = "D:/software/ffmpeg-2025-10-05-git-6231fa7fb7-full_build/bin/ffmpeg.exe"

# loss_fn_alex = lpips.LPIPS(net='alex')

# check what jpeg encoder PIL uses
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
			# image.convert("RGB").save(f"{outputDirectory}/{filename}/AVIF_{quality}.avif", "AVIF", quality=quality)
			# .\ffmpeg.exe -i "D:\dev\workspaces\jpeg\tmp\asphalt_04_diff_4k.png" -c:v libaom-av1 -still-picture 1 -pix_fmt yuv444p -crf 50 ffmpeg_crf50_asphalt_04_diff_4k.avif
		
			crf = ((100 - quality) * 63 + 50) / 100
			print(f"compressing to format AVIF. Quality: {quality}, ffmpeg crf: {crf}")
			result = subprocess.run(
				[
					ffmpegPath,
					"-i", sourcePath,
					"-c:v", "libaom-av1",
					"-still-picture", "1",
					"-pix_fmt", "yuv444p", 
					"-crf", f"{crf}", 
					"-y",
					f"{outputDirectory}/{filename}/AVIF_{quality}.avif"
				], 
				capture_output=True, text=True
			)
			print("STDOUT:", result.stdout)
			print("STDERR:", result.stderr)

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


convertImages(args.i)

# python convert_images.py -o E:/temp/jpeg_test/out -i "E:/temp/jpeg_test/textures/bridge_1024_768.png" 


# python convert_images.py -o E:/temp/jpeg_test/converted -i `
# "E:/temp/jpeg_test/textures/bridge_1024_768.png" `
# "E:/temp/jpeg_test/textures/coast_sand.png" `
# "E:/temp/jpeg_test/textures/Ground082S_4K-PNG_Color.png" `
# "E:/temp/jpeg_test/textures/PavingStones138_4K-PNG_Color.png" `
# "E:/temp/jpeg_test/textures/wall_1024_768.png"

 




