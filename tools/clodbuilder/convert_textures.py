#!/usr/bin/env python3
"""
Converts the textures written by clodbuilder into texture.dds, the BC7-compressed texture with mip levels that CuRast reads.
BC7 encoding is done by AMD Compressonator's command line tool (compressonatorcli).

With multiple textures (the tiles of an atlas, see "texture" in metadata.json), each tile is encoded on its own,
and the atlas is assembled from their BC7 blocks. Each level of the atlas consists of the same level of all tiles,
so mip levels don't bleed between tiles.

Usage: convert_textures.py <clodbuilderOutputDir> [path/to/compressonatorcli]
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import time

DDS_HEADER_SIZE = 148  # "DDS ", the 124 byte header and the 20 byte DX10 header


def image_size(path):
	"""Width and height of a png or jpeg file"""
	with open(path, "rb") as file:
		data = file.read(1 << 20)

	if data[:8] == b"\x89PNG\r\n\x1a\n":
		return struct.unpack(">II", data[16:24])

	# jpeg: find the start of frame marker
	offset = 2
	while offset + 9 < len(data):
		marker = data[offset + 1]
		length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
		if marker in (0xC0, 0xC1, 0xC2):
			height, width = struct.unpack(">HH", data[offset + 5:offset + 9])
			return width, height
		offset += 2 + length

	sys.exit(f"ERROR: can not determine the size of {path}")


def num_mip_levels(width, height, whole_blocks):
	"""Levels end at 4x4, because Compressonator V4.5.52 encodes smaller levels incorrectly.
	For atlas tiles, every level must also consist of whole 4x4 blocks."""
	levels = 1
	while True:
		w, h = width >> levels, height >> levels
		if w < 4 or h < 4:
			break
		if whole_blocks and (w % 4 != 0 or h % 4 != 0):
			break
		levels += 1

	return levels


def encode(compressonator, source, target, levels):
	if os.path.exists(target):
		os.remove(target)

	command = [compressonator, "-fd", "BC7", "-miplevels", str(levels), "-NumThreads", str(os.cpu_count()), "-noprogress", source, target]

	# The Linux release's compressonatorcli is a wrapper script that starts with an empty line instead of #!/bin/bash,
	# so it can't be executed directly.
	with open(compressonator, "rb") as file:
		if file.read(4) != b"\x7fELF":
			command = ["bash"] + command

	subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

	if not os.path.exists(target):
		sys.exit(f"ERROR: compressonatorcli did not write {target}")


def read_dds(path):
	with open(path, "rb") as file:
		data = file.read()

	height, width = struct.unpack_from("<II", data, 12)
	levels = max(struct.unpack_from("<I", data, 28)[0], 1)
	dxgi_format = struct.unpack_from("<I", data, 128)[0]

	if data[:4] != b"DDS " or data[84:88] != b"DX10" or dxgi_format not in (98, 99):
		sys.exit(f"ERROR: {path} is not a BC7-compressed dds file")

	return {"data": data, "width": width, "height": height, "levels": levels}


def assemble_atlas(tiles, columns, rows, target):
	"""Writes a dds file with the tiles side by side, row by row. Empty cells are filled with zero blocks."""
	first = tiles[0]
	width, height, levels = first["width"], first["height"], first["levels"]

	for tile in tiles:
		if (tile["width"], tile["height"], tile["levels"]) != (width, height, levels):
			sys.exit("ERROR: all textures of the atlas must have the same size and number of levels")

	header = bytearray(first["data"][:DDS_HEADER_SIZE])
	struct.pack_into("<I", header, 12, height * rows)
	struct.pack_into("<I", header, 16, width * columns)
	struct.pack_into("<I", header, 20, (width * columns // 4) * (height * rows // 4) * 16)  # linear size of the first level

	with open(target, "wb") as file:
		file.write(header)

		level_offset = DDS_HEADER_SIZE
		for level in range(levels):
			blocks_per_row = (width >> level) // 4
			block_rows = (height >> level) // 4
			row_bytes = 16 * blocks_per_row

			for row in range(rows):
				for block_row in range(block_rows):
					start = level_offset + block_row * row_bytes
					for column in range(columns):
						index = row * columns + column
						if index < len(tiles):
							file.write(tiles[index]["data"][start:start + row_bytes])
						else:
							file.write(bytes(row_bytes))

			level_offset += row_bytes * block_rows

	return width * columns, height * rows, levels


def main():
	if len(sys.argv) < 2:
		sys.exit(__doc__)

	directory = sys.argv[1]
	compressonator = sys.argv[2] if len(sys.argv) > 2 else shutil.which("compressonatorcli")
	if compressonator is None:
		fallback = os.path.expanduser("~/.local/bin/compressonatorcli")
		compressonator = fallback if os.path.exists(fallback) else None
	if compressonator is None:
		sys.exit("ERROR: compressonatorcli not found. Pass its path as the second argument.")

	with open(os.path.join(directory, "metadata.json")) as file:
		metadata = json.load(file)

	if "texture" not in metadata:
		sys.exit("The mesh has no texture.")

	# older clodbuilder versions wrote a single "file"
	texture = metadata["texture"]
	files = texture.get("files", [texture.get("file")])
	columns = texture.get("columns", 1)
	rows = texture.get("rows", 1)
	target = os.path.join(directory, "texture.dds")
	start = time.time()

	if len(files) == 1:
		source = os.path.join(directory, files[0])
		width, height = image_size(source)
		levels = num_mip_levels(width, height, False)
		print(f"encoding {files[0]} ({width}x{height}, {levels} levels)")
		encode(compressonator, source, target, levels)
	else:
		tiles = []
		for name in files:
			source = os.path.join(directory, name)
			tile_target = os.path.splitext(source)[0] + ".dds"
			width, height = image_size(source)
			if width % 4 != 0 or height % 4 != 0:
				sys.exit(f"ERROR: {name} is {width}x{height}, atlas tiles must consist of whole 4x4 blocks")

			levels = num_mip_levels(width, height, True)
			print(f"encoding {name} ({width}x{height}, {levels} levels, {time.time() - start:.0f}s)")
			encode(compressonator, source, tile_target, levels)
			tiles.append(read_dds(tile_target))

		print(f"assembling the {columns}x{rows} atlas")
		width, height, levels = assemble_atlas(tiles, columns, rows, target)

		for name in files:
			os.remove(os.path.join(directory, os.path.splitext(name)[0] + ".dds"))

	result = read_dds(target)
	print(f"wrote {target}: {result['width']}x{result['height']}, {result['levels']} levels, "
		f"{len(result['data']) / 1e6:.0f} MB ({time.time() - start:.0f}s)")


if __name__ == "__main__":
	main()
