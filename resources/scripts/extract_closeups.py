#import tkinter as tk
#from tkinter import filedialog, messagebox
import os
import shutil
#import sys
import argparse
#import torch
#from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pillow_jxl
import pillow_avif
# import pillow_heif
from PIL import Image, ImageTk, features
import numpy as np
#import flip_evaluator as flip
#import lpips
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import namedtuple

#from pillow_heif import register_heif_opener
#register_heif_opener()

#closeup
closeup = {
	"x": 152,
	"y": 512,
	"width": 43,
	"height": 33
}

parser = argparse.ArgumentParser(description="Example script with -o option")
parser.add_argument('-o', type=str, help='')
parser.add_argument('-i', type=str, help='')

args = parser.parse_args()

files = os.listdir(args.i)
files_astc = [f for f in files if f.startswith("astc") and f.endswith("dds")]
files_astc_sizes = [os.path.getsize(f"{args.i}/{f}") for f in files_astc]

print(files_astc)
print(files_astc_sizes)

# algorithms = ["bc", "jpegli", "jpegturbo", "jpegXL", "AVIF"]
algorithms = ["jpegli"]

def findClosest(value, listOfValues):

	closest = 0.000001
	
	for refVal in listOfValues:
		diff = abs(value - refVal)

		if (diff < abs(value - closest)):
			closest = refVal 

	return {"value": closest, "index": listOfValues.index(closest)}



for algorithm in algorithms:

	afiles = [f for f in files if f.startswith(algorithm)]
	afiles_sizes = [os.path.getsize(f"{args.i}/{f}") for f in afiles]

	for i in range(len(afiles)):
		afile = afiles[i]
		afile_size = afiles_sizes[i]

		closest = findClosest(afile_size, files_astc_sizes)
		#print(f"closest to {afile_size}: {closest["value"]}")

		ratio = afile_size / closest["value"]
		#print(f"ratio: {ratio:.2f}")

		if (1 - abs(ratio)) > 0.1: continue
			#print("acceptable")

		closestAstc = files_astc[closest["index"]]
		print(f"{afile} with {afile_size / 1000} kb is closest to {closestAstc} with {closest["value"] / 1000} kb")



	print(afiles_sizes)


