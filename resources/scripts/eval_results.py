import os
import shutil
import argparse
import pillow_jxl
import pillow_avif
from PIL import Image, ImageTk, features
import numpy as np
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import namedtuple
import csv


#csvPath = "E:/temp/jpeg_test/eval/statue.png/data.csv"
csvPaths = [
	# "E:/temp/jpeg_test/eval/statue.png/data.csv",
	# "E:/temp/jpeg_test/eval/bricks.png/data.csv",
	# "E:/temp/jpeg_test/eval/bridge.png/data.csv",
	# "E:/temp/jpeg_test/eval/wall.png/data.csv",
	# "E:/temp/jpeg_test/eval/snow.png/data.csv",

	# "E:/temp/jpeg_test/eval/anita_mui.png/data.csv",
	"E:/temp/jpeg_test/eval/bricks.png/data.csv",
	"E:/temp/jpeg_test/eval/bridge.png/data.csv",
	"E:/temp/jpeg_test/eval/bridge_1024_768.png/data.csv",
	"E:/temp/jpeg_test/eval/coast_sand.png/data.csv",
	"E:/temp/jpeg_test/eval/coast_sand_01_diff_4k.png/data.csv",
	"E:/temp/jpeg_test/eval/coral_gravel_diff_4k.png/data.csv",
	"E:/temp/jpeg_test/eval/fabric.png/data.csv",
	"E:/temp/jpeg_test/eval/flowers.png/data.csv",
	"E:/temp/jpeg_test/eval/graffiti.png/data.csv",
	"E:/temp/jpeg_test/eval/Ground082S_4K-PNG_Color.png/data.csv",
	"E:/temp/jpeg_test/eval/hill.png/data.csv",
	"E:/temp/jpeg_test/eval/house.png/data.csv",
	"E:/temp/jpeg_test/eval/house2.png/data.csv",
	"E:/temp/jpeg_test/eval/kyumizudera.png/data.csv",
	"E:/temp/jpeg_test/eval/Onyx010_4K-PNG_Color.png/data.csv",
	"E:/temp/jpeg_test/eval/PavingStones126A_4K-PNG_Color.png/data.csv",
	"E:/temp/jpeg_test/eval/PavingStones138_4K-PNG_Color.png/data.csv",
	"E:/temp/jpeg_test/eval/plastic.png/data.csv",
	"E:/temp/jpeg_test/eval/river.png/data.csv",
	"E:/temp/jpeg_test/eval/snow.png/data.csv",
	"E:/temp/jpeg_test/eval/statue.png/data.csv",
	"E:/temp/jpeg_test/eval/wall.png/data.csv",
	"E:/temp/jpeg_test/eval/wall_1024_768.png/data.csv",
]



# first plot of a reference image clears all previous plots
plt.close("all")
plt.clf()

# LIST OF ALL ALGORITHMS
allAlgorithms = []
for csvPath in csvPaths:

	records = []

	with open(csvPath, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter=";")
		for row in reader:
			records.append(row)

	algorithms = [r["algorithm"] for r in records]
	allAlgorithms.extend(algorithms)

allAlgorithms = list(set(allAlgorithms))
allAlgorithms.sort()
print("Algorithms: ")
print(allAlgorithms)

# PLOT CSVs
for csvPath in csvPaths:

	records = []

	with open(csvPath, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter=";")
		for row in reader:
			records.append(row)

	algorithms = [r["algorithm"] for r in records]
	algorithms = list(set(algorithms))
	algorithms.sort()

	for algorithm in ["JPEG XL", "JPEGLI", "AVIF"]:
		entries = [t for t in records if t["algorithm"] == algorithm]
		bpps = [float(t["bpp"]) for t in entries]
		psnrs = [float(t["flip"]) for t in entries]
		bbps_min = min(bpps)
		bbps_max = max(bpps)
		psnr_min = min(psnrs)
		psnr_max = max(psnrs)




	for algorithm in algorithms:
		entries = [t for t in records if t["algorithm"] == algorithm]
		bpps = [float(t["bpp"]) for t in entries]
		psnrs = [float(t["flip"]) for t in entries]

		# bpps = [(v - bbps_min) / (bbps_max - bbps_min) for v in bpps]
		# psnrs = [(v - psnr_min) / (psnr_max - psnr_min) for v in psnrs]
		
		colorIndex = allAlgorithms.index(algorithm)
		print(f"colorIndex: {colorIndex}")
		color = plt.get_cmap("tab10").colors[colorIndex]
		plt.plot(bpps, psnrs, linestyle='-', label=algorithm, clip_on=True, color=color, alpha=0.5)
		# plt.plot(bpps, psnrs, marker='o', linestyle='-', label=algorithm, clip_on=True, color=color)
		#plt.xlim( 0, 4.5)

	# Axis labels and title
	
	# plt.ylabel("PSNR↑")
	# plt.title(f"Compression - {filename}")

	

	# plt.savefig(f"{args.output}/{filename}/plot_psnr.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
	

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))   # remove duplicates
plt.legend(by_label.values(), by_label.keys())


plt.xlabel("bits per pixel")
plt.grid(True)
#plt.legend()
plt.xlim( 0, 2.5)
plt.show()