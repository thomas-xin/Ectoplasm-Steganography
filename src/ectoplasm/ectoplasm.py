import io
import logging
import os
from math import sqrt, floor
from PIL import Image
import numpy as np
import reedsolo

rsc = reedsolo.RSCodec(1)
CHANNELS = [1, 2, 4]

def encode_b128(b):
	paddings = 0
	while b[-1] == 0:
		paddings += 1
		b = b[:-1]
	import bitarray
	b1 = bitarray.bitarray(endian="big")
	b1.frombytes(b)
	b2 = bitarray.bitarray(endian="big")
	while len(b1):
		temp, b1 = b1[:7], b1[7:]
		temp.insert(0, 0)
		b2.extend(temp)
	return b2.tobytes() + b"\x00" * paddings

def decode_b128(b):
	assert b.isascii()
	import bitarray
	b1 = bitarray.bitarray(endian="big")
	b1.frombytes(b)
	b2 = bitarray.bitarray(endian="big")
	while len(b1):
		temp, b1 = b1[1:8], b1[8:]
		if len(temp) < 7:
			break
		b2.extend(temp)
	return b2.tobytes().removesuffix(b"\x00")

# text = b"Hello World!\n" * 100
# assert decode_b128(encode_b128(text)) == text
# assert encode_b128(decode_b128(text)).removesuffix(b"\x00") == text

def quantise_into(a, clip=None, in_place=True, checker=1, dtype=np.uint8):
	if issubclass(a.dtype.type, np.integer):
		return a
	if checker:
		inds = np.indices(a.shape, dtype=np.uint8).sum(axis=0, dtype=np.uint8)
		inds &= checker
		inds = inds.view(bool)
	z = np.random.random_sample(a.shape)
	if checker:
		z %= 0.5
	else:
		z %= 1 - 2 ** -12
	a = np.add(a, z, out=a if in_place else None)
	if checker:
		a[inds] += 0.5 - 2 ** -12
	if clip:
		a = np.clip(a, *clip, out=a)
	if issubclass(getattr(dtype, "type", dtype), np.floating):
		np.floor(a, out=a)
	return np.asanyarray(a, dtype=dtype)

def encode_nrzi(text):
	rs2 = reedsolo.RSCodec(min(len(text) + 1 >> 1, 85))
	data = rs2.encode(text)
	arr = np.frombuffer(data, dtype=np.uint8)
	bits = np.empty(len(data) * 8, dtype=np.uint8)
	for i in range(8):
		bits[i::8] = arr & (1 << i) != 0
	cum = np.cumsum(bits, out=bits)
	cum &= 1
	nrzi = cum.view(bool)
	return nrzi

def decode_nrzi(nrzi):
	diff = np.diff(nrzi.view(np.int8))
	bits = np.concatenate(([nrzi[0]], diff.view(bool))).astype(np.uint8)
	arr = np.zeros(len(bits) // 8, dtype=np.uint8)
	for i in range(8):
		arr += bits[i::8] << i
	data = arr.data
	rs2 = reedsolo.RSCodec(min(round(len(data) / 3), 85))
	text = rs2.decode(data)
	return text[0]

def encode_qr(text, version=3):
	import base45
	import pyqrcode
	text = base45.b45encode(text).upper().decode("ascii")
	size = len(text)
	if version == 3:
		assert size <= 77, size
		err = "L"
		w = 32
	elif version == 11:
		assert size <= 259, size
		err = "Q"
		w = 64
	elif version == 27:
		assert size <= 910, size
		err = "H"
		w = 128
	else:
		raise NotImplementedError(version)
	img = pyqrcode.create(text, error=err, version=version, mode="alphanumeric", encoding="ascii")
	import io
	b = io.BytesIO()
	img.png(b, scale=1, module_color=(0,) * 3, background=(255,) * 4, quiet_zone=2)
	b.seek(0)
	box = (0, 0, w, w)
	imo = Image.open(b)
	import random
	if random.randint(0, 1):
		imo = imo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
	imo = imo.rotate(random.randint(0, 3) * 90)
	return imo.crop(box)

def encode_barcode(text):
	import barcode
	num = int.from_bytes(text, "big")
	writer = barcode.Code128(str(num), writer=barcode.writer.ImageWriter("png"))
	import io
	b = io.BytesIO()
	writer.write(b, {"font_size": 3, "text_distance": 3, "module_width": 1, "module_height": 24, "quiet_zone": 8, "dpi": 25.40000001})
	b.seek(0)
	im = Image.open(b)
	return im.crop((0, 1, im.width, 25))

reader1 = None
reader2 = None
def decode_qr(images, channels=1):
	global reader1, reader2
	if isinstance(images, Image.Image):
		images = [images]
	if not reader1:
		import pyzbar.pyzbar
		reader1 = pyzbar.pyzbar
	import base45
	seen = []
	for image in images:
		if image.mode not in ("L", "RGB"):
			image = image.convert("L")
		for scale_factor in (0.25, 1 / 3, 0.5):
			im = image.resize((round(image.width * scale_factor), round(image.height * scale_factor)), resample=Image.Resampling.LANCZOS)
			a = (np.asanyarray(im, dtype=np.uint8) > 127).view(np.uint8)
			a *= 255
			im = Image.fromarray(a, "L").resize((im.width * 3, im.height * 3), resample=Image.Resampling.NEAREST)
			dets = reader1.decode(im, symbols=[reader1.ZBarSymbol.QRCODE])
			if dets:
				a2 = np.array(image, dtype=np.uint8)
				polys = []
				for data in dets:
					decoded = data.data
					try:
						value = base45.b45decode(decoded)
					except ValueError:
						continue
					poly = np.array([[point.x / scale_factor / 3, point.y / scale_factor / 3] for point in data.polygon], dtype=np.int32)
					polys.append(poly)
					while len(value) >= 51:
						yield value[:51]
						value = value[51:]
				import cv2
				cv2.fillPoly(a2, pts=polys, color=(255, 255, 255))
				image = Image.fromarray(a2)
		if max(image.width, image.height) > 1024:
			small = min(image.width, image.height, 1024)
			image = image.resize((small, small), resample=Image.Resampling.LANCZOS)
		seen.append(image)
	if not reader2:
		from qreader import QReader
		reader2 = QReader(model_size="s", min_confidence=0.25, reencode_to=None)
	for im in tuple(seen):
		if im.width == im.height == 1024:
			quads = (
				im.crop((0, 0, 512, 512)).rotate(180),
				im.crop((512, 512, 1024, 1024)),
				im.crop((0, 512, 512, 1024)).rotate(90),
				im.crop((512, 0, 1024, 512)).rotate(-90),
			)
			seen.extend(quads)
	found = set()
	retries = []
	for i, image in enumerate(seen):
		image_np = np.asanyarray(image, dtype=np.uint8)
		for data in reader2.detect(image_np):
			if i not in found and image.width == image.height == 1024:
				quads = (
					image.crop((0, 0, 512, 512)).rotate(180),
					image.crop((512, 512, 1024, 1024)),
					image.crop((0, 512, 512, 1024)).rotate(90),
					image.crop((512, 0, 1024, 512)).rotate(-90),
				)
				retries.extend(quads)
			found.add(i)
			if not data:
				yield None
				continue
			try:
				decoded = reader2.decode(image_np, data)
			except Exception:
				import traceback
				traceback.print_exc()
				yield None
				continue
			if not decoded:
				yield None
				continue
			try:
				value = base45.b45decode(decoded)
			except ValueError:
				continue
			while len(value) >= 51:
				yield value[:51]
				value = value[51:]
	for i, image in enumerate(retries):
		image_np = np.asanyarray(image, dtype=np.uint8)
		for data in reader2.detect(image_np):
			if not data:
				yield None
				continue
			try:
				decoded = reader2.decode(image_np, data)
			except Exception:
				import traceback
				traceback.print_exc()
				yield None
				continue
			if not decoded:
				yield None
				continue
			try:
				value = base45.b45decode(decoded)
			except ValueError:
				continue
			while len(value) >= 51:
				yield value[:51]
				value = value[51:]

def encode_message(message):
	if isinstance(message, str):
		message = message.encode("utf-8")
	if isinstance(message, memoryview):
		message = bytes(message)
	elif not isinstance(message, bytes):
		raise TypeError("`message` should be an instance of `str`, `bytes`, or `memoryview`.")
	# import zlib
	# import lzma
	import brotli
	import paq
	encodes = {}
	encodes["raw"] = b"\x91" + message
	if message.isascii():
		encodes["b128"] = b"\x92" + decode_b128(message)
	# encodes["zlib"] = zlib.compress(message, level=9)
	# encodes["lzma"] = lzma.compress(message, format=lzma.FORMAT_ALONE, check=-1, preset=9, filters=None)
	encodes["brotli"] = brotli.compress(message, quality=11, mode=brotli.MODE_TEXT)
	encodes["paq"] = paq.compress(message)
	order = sorted(encodes, key=lambda k: len(encodes[k]))
	# print("Compressed sizes:\n" + "\n".join(f"{k}:{len(encodes[k])}" for k in order))
	return encodes[order[0]]

def decode_message(message):
	if isinstance(message, str):
		message = message.encode("utf-8")
	if isinstance(message, memoryview):
		message = bytes(message)
	elif not isinstance(message, bytes):
		raise TypeError("`message` should be an instance of `str`, `bytes`, or `memoryview`.")
	if not message:
		raise ValueError("Input is empty.")
	if message.startswith(b"\x00c"):
		import paq
		return paq.decompress(message)
	if message[:1] in b"\x0b\x1b\x8b":
		import brotli
		return brotli.decompress(message)
	# if message[:1] == b"]":
	# 	import lzma
	# 	return lzma.decompress(message)
	# if message[:1] == b"x":
	# 	import zlib
	# 	return zlib.decompress(message)
	if message[:1] == b"\x92":
		return encode_b128(message[1:]).removesuffix(b"\x00")
	if message[:1] == b"\x91":
		return message[1:]
	raise ValueError("Unrecognised encoding " + str(message[:12]))
# assert decode_message(encode_message(m := b"Hello World!\n" * 100)) == m

def diagonals(rows, cols):
	indices = np.empty((2, rows * cols), dtype=np.uint32)
	i = 0
	for d in range(rows + cols - 1):
		row_start = max(0, d - cols + 1)
		row_end = min(rows - 1, d)
		col_indices = np.arange(row_start, row_end + 1)
		row_indices = d - col_indices
		size = len(row_indices)
		indices[1][i:i + size] = row_indices
		indices[0][i:i + size] = col_indices
		i += size
	return indices.T
# assert np.all(diagonals(4, 3) == [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0], [1, 2], [2, 1], [3, 0], [2, 2], [3, 1], [3, 2]])

def tesselate(tesselation, grid, tile, seed=0):
	W, H = grid
	w, h = tile
	rows = H // h
	cols = W // w
	if tesselation == "plain":
		coords = diagonals(rows, cols)
		coords *= np.uint32((w, h))
		for x, y in coords:
			yield (x, y, w, h)
		return
	if tesselation == "pythagorean":
		cells = np.ones((rows, cols), dtype=bool)
		init = -(seed & 3)
		row = 0
		col = init
		offs = 0
		while True:
			y = -col + (offs * 2) + offs // 2
			x = row * 2 + (offs & 1)
			if x + 1 >= rows or y < 0:
				offs += 1
				if offs * 2 >= cols * 1.5:
					break
				row = 0
				col = init
				continue
			if y < rows - 1:
				cells[y:y + 2].T[x:x + 2] = 0
				yield (x * w, y * h, w * 2, h * 2)
			row += 1
			col += 1
		small = np.nonzero(cells)
		for y, x in zip(*small):
			yield (x * w, y * h, w, h)
		return
	raise NotImplementedError(tesselation)

def batch_images(it, maxsize, tesselation="plain", tile=4, qr_scale=2, seed=0):
	def retrieve(qr_scale=2):
		try:
			qr_info = qr_map[qr_scale]
		except KeyError:
			raise NotImplementedError(qr_scale)
		if qr_scale == 2:
			text = next(it)
		elif qr_scale == 4:
			text = next(it) + next(it) + next(it)
		else:
			raise NotImplementedError(qr_scale)
		qr_version = qr_info["version"]
		return encode_qr(text, version=qr_version)
	im = retrieve(qr_scale=qr_scale)
	scaled = im.width * tile
	grid_img = Image.new("1", (maxsize, maxsize))
	rtes = tesselation if tesselation != "diagonal" else "plain"
	for x, y, w, h in tesselate(rtes, (maxsize, maxsize), (scaled, scaled), seed=seed):
		qs = 2
		# print(qs, w, h, scaled, scaled)
		im2 = retrieve(qs).resize((w, h), resample=Image.Resampling.NEAREST)
		grid_img.paste(im2, (x, y))
	if tesselation == "diagonal":
		x, y = round(maxsize / 2), round(maxsize / 2)
		grid_img = grid_img.resize((round(maxsize * sqrt(2)), round(maxsize * sqrt(2))), resample=Image.Resampling.LANCZOS).rotate(45, resample=Image.Resampling.BICUBIC, expand=True).crop((x, y, x + maxsize, y + maxsize))
	return grid_img

qr_map = {
	2: dict(
		size=32,
		n=42,
		n2=46,
		version=3,
	),
	4: dict(
		size=64,
		n=124,
		n2=124,
		version=11,
	),
	8: dict(
		size=128,
		n=596,
		n2=0,
		version=27,
	),
}

def encode_fountain(data, qr_scale=2, maxsize=512, channels=1, redundancy=6):
	buffer = 2
	try:
		qr_info = qr_map[qr_scale]
	except KeyError:
		raise NotImplementedError(qr_scale)
	qr_size = qr_info["size"]
	n = qr_info["n"]
	mcount = (maxsize // buffer // qr_size) ** 2 * channels
	# print("Maximum chunks:", mcount)
	count = len(data) // n + 1
	# print("Required chunks:", count)
	if len(data) > 65535 or count > mcount:
		raise OverflowError("Input exceeds maximum representable size.")
	used = (maxsize // (qr_size * 2) * 2) ** 2 * channels
	# print("Used chunks:", used)
	def iter_fountain():
		from raptorq import Encoder
		encoder = Encoder.with_defaults(data, n)
		packets = encoder.get_encoded_packets(used + 1)
		for i, packet in enumerate(packets):
			head = len(data).to_bytes(2, "little") + qr_scale.to_bytes(2, "little")
			header = rsc.encode(head)
			text = header + packet
			yield text
	return iter_fountain()

def decode_fountain(fountain):
	decoder = None
	target_size, target_n = None, None
	from raptorq import Decoder
	seen = set()
	i = 0
	for i, data in enumerate(decode_qr(fountain, channels=len(CHANNELS)), 1):
		if not data:
			continue
		header, packet = data[:5], data[5:]
		head = rsc.decode(header)[0]
		size, qr_scale = int.from_bytes(head[:2], "little"), int.from_bytes(head[2:], "little")
		qr_info = qr_map[qr_scale]
		n = qr_info["n"]
		n2 = qr_info["n2"]
		if not decoder:
			target_size = size
			target_n = n
			decoder = Decoder.with_defaults(size, n)
		elif size != target_size or n != target_n:
			raise ValueError("Data size mismatch!")
		if packet in seen:
			continue
		seen.add(packet)
		for j in range(0, len(packet), n2):
			p = packet[j:j + n2]
			assert len(p) == n2, (len(p), n2)
			result = decoder.decode(p)
			if result:
				logging.info(f"Decode successful after {len(seen)}/{i} readable chunks")
				return decode_message(result)
	if not i:
		raise FileNotFoundError("Decode unsuccessful: No valid chunks found")
	raise FileNotFoundError(f"Decode unsuccessful after {len(seen)}/{i} readable chunks")

def forceload(image):
	if isinstance(image, np.ndarray):
		image = Image.fromarray(image, mode="RGB" if image.shape[-1] == 3 else "L")
	elif isinstance(image, (str, io.IOBase)):
		image = Image.open(image)
	elif isinstance(image, (bytes, memoryview)):
		image = Image.open(io.BytesIO(image))
	elif not isinstance(image, Image.Image):
		raise TypeError("`image` should be an instance of `PIL.Image.Image`, `np.ndarray`, `bytes`, `memoryview`, `io.IOBase` or `os.PathLike`.")
	return image

def encode_image(image, message, redundancy=6, dither_wrap=4, strength=1, compress=True, debug=None):
	image = forceload(image)
	if debug and not os.path.exists(debug):
		os.mkdir(debug)
	if debug:
		for fn in os.listdir(debug):
			os.remove(debug + "/" + fn)
		image.save(f"{debug}/initial.png")
	size = image.size
	maxsize = min(2048, min(image.size))
	A = None
	if image.mode != "RGB":
		if "A" in image.mode:
			A = image.getchannel("A")
		image = image.convert("RGB")
	# image = batch_barcodes(fountain, image, strength=strength)
	rgb = np.array(image, dtype=np.float32)
	rgb *= 1 / 255
	if compress:
		data = encode_message(message)
	else:
		data = message
	# print("Encoded message:", data)

	def encode_part(imt, depth=256, start=0):
		for i, arr in enumerate(imt, start):
			if i not in CHANNELS:
				continue
			fountain = encode_fountain(data, qr_scale=2, maxsize=maxsize, channels=1, redundancy=redundancy)
			mult = floor(sqrt(size[0] * size[1] / 1048576))
			if i == 1:
				if size[0] >= 2048:
					tesselation = "pythagorean"
					tile = 1 * mult
				else:
					tesselation = "plain"
					tile = 4 * mult
			elif i == 2:
				tesselation = "diagonal"
				tile = 4 * mult
			else:
				tesselation = "pythagorean"
				tile = 2 * mult
			steg = batch_images(fountain, maxsize, tesselation, tile, seed=i)
			if steg.size != size:
				steg = steg.resize(size, resample=Image.Resampling.NEAREST)
			stegmap = np.asanyarray(steg, dtype=np.uint8).T
			if debug:
				im2 = Image.fromarray(stegmap * 255, "L")
				im2.save(f"{debug}/mask{i}.png")
			if i & 1:
				omap = np.logical_not(stegmap)
			else:
				omap = stegmap.view(bool)
			arr *= depth
			arr[omap] -= 0.5
			arr = np.round(arr, out=arr)
			arr[omap] += 0.5
			arr *= 1 / depth
			np.clip(arr, 0, 1, out=arr)
			if debug:
				a = arr.T * 255
				# print(a.shape)
				a = quantise_into(a, clip=(0, 255))
				im = Image.fromarray(a, mode="L")
				im.save(f"{debug}/{i}.png")
	import cv2

	rand = np.random.random_sample(rgb.shape)
	rand -= 0.5
	rand *= strength / 72
	rgb += rand

	yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb, dst=rgb)
	encode_part(yuv.T, depth=40 / strength, start=0)
	rgb = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB, dst=rgb)

	np.clip(rgb, 0, 1, out=rgb)

	hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS, dst=rgb)
	encode_part(hls.T, depth=64 / strength, start=3)
	rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB, dst=rgb)

	rgb *= 255
	rgb = quantise_into(rgb, clip=(0, 255), checker=dither_wrap)
	im = Image.fromarray(rgb, mode="RGB")
	if A and np.min(A) < 254:
		im.putalpha(A)

	if debug:
		im.save(f"{debug}/fountain.png")
	return im

def decode_image(image, strength=1, debug=None):
	image = forceload(image)
	if debug and not os.path.exists(debug):
		os.mkdir(debug)
	if debug:
		for fn in os.listdir(debug):
			os.remove(debug + "/" + fn)
		image.save(f"{debug}/fountain.png")
	import cv2

	def chunking(im):
		width, height = im.size
		rgb = np.array(im, dtype=np.float32)
		rgb *= 1 / 255
		# rgb = cv2.GaussianBlur(rgb, (3, 3), 0, dst=rgb)
		# rgb = cv2.resize(rgb, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
		ims = []
		def decode_part(imt, depth=256, start=0):
			for i, arr in enumerate(imt, start):
				if i not in CHANNELS:
					continue
				if debug:
					im2 = Image.fromarray((arr.T * 255).astype(np.uint8), "L")
					im2.save(f"{debug}/map-{i}.png")

				arr *= depth
				arr %= 1
				arr -= 0.5
				np.abs(arr, out=arr)
				if not i & 1:
					np.subtract(0.5, arr, out=arr)
				a = arr.T * 2 * 255
				# a = cv2.GaussianBlur(a, (3, 3), 0, dst=a)
				square = int(np.sqrt(width * height))
				if i == 4:
					small = min(2048, max(square // 2, width, height))
				else:
					small = min(1024, max(square // 4, width // 2, height // 2))
				a = cv2.resize(a, (small, small), interpolation=cv2.INTER_AREA)
				im = Image.fromarray(a.astype(np.uint8), mode="L")
				if debug:
					im.save(f"{debug}/{i}.png")
				ims.append(im)
				yield im

		yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
		yield from decode_part(yuv.T, depth=40 / strength, start=0)

		hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
		yield from decode_part(hls.T, depth=64 / strength, start=3)

	return decode_fountain(chunking(image))