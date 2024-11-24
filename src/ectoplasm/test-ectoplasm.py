import io
import logging
import os
import random
import orjson
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps
import ectoplasm

strength = 12

def predicate(func, fountain, message, debug=None):
	decoded = ectoplasm.decode_image(func(fountain), strength=strength, debug=debug)
	if decoded != message:
		logging.warning((decoded + b" != " + message).decode("utf-8", "replace"))
	return decoded == message

# Directly decode (as control)
def layer_direct(fountain):
	logging.info("Direct test:")
	return fountain
# Flip randomly (left-right, up-down, transpose etc)
def layer_flip(fountain):
	logging.info("Flip test:")
	return fountain.transpose(random.randint(0, 6))
# Invert all colours in the image
def layer_invert(fountain):
	logging.info("Invert test:")
	return ImageOps.invert(fountain)
# Scroll image randomly left/right/up/down, up to 50% of the image's original size
def layer_translate(fountain):
	logging.info("Translation test:")
	return ImageChops.offset(fountain, round((random.random() - 0.5) * fountain.width), round((random.random() - 0.5) * fountain.height))
# Blank a random section of the image, up to 50% of the image's original size
def layer_blank(fountain):
	logging.info("Blank test:")
	f = fountain.convert("RGBA")
	a = random.random() * 0.25 + 0.5
	b = random.random() * 0.25 + 0.5
	x, y = round(random.random() * (1 - a) * fountain.width), round(random.random() * (1 - b) * fountain.height)
	w, h = round(a * fountain.width), round(b * fountain.height)
	temp = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
	f.paste(temp, (x, y))
	return f
# Rotate image at any random angle
def layer_rotate(fountain):
	logging.info("Rotation test:")
	return fountain.rotate(random.random() * 360, expand=False, resample=Image.Resampling.BICUBIC)
# Skew image at any random ratio in both axes, up to 50% of the image's original size
def layer_skew(fountain):
	logging.info("Skew test:")
	x = random.random() - 0.5
	y = random.random() - 0.5
	tx = -x * fountain.width / 2
	ty = -y * fountain.height / 2
	return fountain.transform(fountain.size, Image.AFFINE, (1, x, tx, y, 1, ty), resample=Image.Resampling.BICUBIC)
# Crop image to minimum 25% of the image's original size
def layer_crop(fountain):
	logging.info("Crop test:")
	a = random.random() * 0.25 + 0.5
	b = random.random() * 0.25 + 0.5
	x, y = round(random.random() * (1 - a) * fountain.width), round(random.random() * (1 - b) * fountain.height)
	w, h = round(a * fountain.width), round(b * fountain.height)
	return fountain.crop((x, y, x + w, y + h))
# Hueshift image by a random factor
def layer_hueshift(fountain):
	logging.info("Hueshift test:")
	spl = fountain.convert("HSV").split()
	return Image.merge("HSV", (spl[0].point(lambda x: x + random.randint(1, 255) & 255), spl[1], spl[2])).convert("RGB")
# Resize image by a random factor, from 25% to 225% of the image's original size
def layer_resize(fountain):
	logging.info("Resize test:")
	return fountain.resize((round((random.random() + 0.5) * fountain.width), round((random.random() + 0.5) * fountain.height)), resample=Image.Resampling.BICUBIC)
# Performs the magik filter on the image, up to 33% of the image's original size
def quad_as_rect(quad):
	if quad[0] != quad[2]:
		return False
	if quad[1] != quad[7]:
		return False
	if quad[4] != quad[6]:
		return False
	if quad[3] != quad[5]:
		return False
	return True
def quad_to_rect(quad):
	assert(len(quad) == 8)
	assert(quad_as_rect(quad))
	return (quad[0], quad[1], quad[4], quad[3])
def rect_to_quad(rect):
	assert(len(rect) == 4)
	return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])
def shape_to_rect(shape):
	assert(len(shape) == 2)
	return (0, 0, shape[0], shape[1])
def griddify(rect, w_div, h_div):
	w = rect[2] - rect[0]
	h = rect[3] - rect[1]
	x_step = w / float(w_div)
	y_step = h / float(h_div)
	y = rect[1]
	grid_vertex_matrix = []
	for _ in range(h_div + 1):
		grid_vertex_matrix.append([])
		x = rect[0]
		for _ in range(w_div + 1):
			grid_vertex_matrix[-1].append([int(x), int(y)])
			x += x_step
		y += y_step
	grid = np.array(grid_vertex_matrix)
	return grid
def distort_grid(org_grid, max_shift):
	new_grid = np.copy(org_grid)
	x_min = np.min(new_grid[:, :, 0])
	y_min = np.min(new_grid[:, :, 1])
	x_max = np.max(new_grid[:, :, 0])
	y_max = np.max(new_grid[:, :, 1])
	new_grid += np.random.randint(-max_shift, max_shift + 1, new_grid.shape)
	new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
	new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
	new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
	new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
	return new_grid
def grid_to_mesh(src_grid, dst_grid):
	assert(src_grid.shape == dst_grid.shape)
	mesh = []
	for i in range(src_grid.shape[0] - 1):
		for j in range(src_grid.shape[1] - 1):
			src_quad = [
				src_grid[i, j, 0], src_grid[i, j, 1],
				src_grid[i + 1, j, 0], src_grid[i + 1, j, 1],
				src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
				src_grid[i, j + 1, 0], src_grid[i, j + 1, 1],
			]
			dst_quad = [
				dst_grid[i, j, 0], dst_grid[i, j, 1],
				dst_grid[i + 1, j, 0], dst_grid[i + 1, j, 1],
				dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
				dst_grid[i, j + 1, 0], dst_grid[i, j + 1, 1],
			]
			dst_rect = quad_to_rect(dst_quad)
			mesh.append([dst_rect, src_quad])
	return list(mesh)
def layer_magik(fountain):
	logging.info("Magik test:")
	dst_grid = griddify(shape_to_rect(fountain.size), 4, 4)
	src_grid = distort_grid(dst_grid, np.sqrt(fountain.width * fountain.height) // 4 // 3)
	mesh = grid_to_mesh(src_grid, dst_grid)
	return fountain.transform(fountain.size, Image.Transform.MESH, mesh, resample=Image.Resampling.NEAREST)
# Convert image to jpeg and back
def layer_jpeg(fountain):
	logging.info("JPEG test:")
	b = io.BytesIO()
	fountain.save(b, format="jpeg", quality=90)#random.randint(80, 90))
	b.seek(0)
	return Image.open(b)
# Brighten or darken image by a random factor, up to 50% above or below
def layer_brighten(fountain):
	logging.info("Brightness test:")
	return ImageEnhance.Brightness(fountain).enhance(random.random() + 0.5)

available = {
	"decode-direct": layer_direct,
	"decode-flip": layer_flip,
	"decode-invert": layer_invert,
	"decode-translate": layer_translate,
	"decode-blank": layer_blank,
	"decode-rotate": layer_rotate,
	"decode-skew": layer_skew,
	"decode-crop": layer_crop,
	"decode-hueshift": layer_hueshift,
	"decode-resize": layer_resize,
	"decode-magik": layer_magik,
	"decode-brighten": layer_brighten,
	"decode-jpeg": layer_jpeg,
}

if not os.path.exists("tests"):
	os.mkdir("tests")

test_text = "tests/texts.txt"
with open(test_text, "rb") as f:
	texts = f.read().splitlines()
test_images = "tests/images"
images = [test_images + "/" + t for t in os.listdir(test_images)]

stats = {k: [0, 0] for k in available}
statfile = "tests/stats.json"
if os.path.exists(statfile):
	with open(statfile, "rb") as f:
		stats.update(orjson.loads(f.read()))

for i in range(1000):
	im = None
	while not im:
		path = random.choice(images)
		im = Image.open(path)
		if im.width * im.height < 1048576:
			im = None
	message = b""
	while not message.strip():
		message = random.choice(texts)
	fountain = ectoplasm.encode_image(im, message, redundancy=1, dither_wrap=2, strength=strength, debug="tests/encode")
	transform = random.choice(tuple(available))
	try:
		assert predicate(available[transform], fountain, message, debug="tests/" + transform)
	except Exception as ex:
		logging.error("FAIL: " + repr(ex) + ": " + path + ", " + transform + "; " + message.decode("utf-8", "replace"))
	else:
		stats[transform][0] += 1
	stats[transform][1] += 1
	with open(statfile, "wb") as f:
		f.write(orjson.dumps(stats))