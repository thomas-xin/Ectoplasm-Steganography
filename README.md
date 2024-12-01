# Ectoplasm-Steganography

## Installation
`py -m pip install ectoplasm-steganography` - Windows<br />
`python3 -m pip install ectoplasm-steganography` - Linux

## Usage
- As a Python package (recommended due to caching):
```python
import ectoplasm
image = ectoplasm.encode_image("image.png", b"Hello World!", strength=1)
assert ectoplasm.decode_image(image, strength=1) == b"Hello World!"
```

Additional utility functions are provided for applying automatically-selected compression where applicable:
```python
message = b"This is a test, please disregard!"
compressed = ectoplasm.encode_message(message)
assert ectoplasm.decode_message(compressed) == message
```

Utility functions are also provided for saving information to image metadata. Metadata stored in this manner is automatically decoded by `decode_image`, however steganographic data is prioritised where intact.
```python
ectoplasm.save_image(image, b"Hello World!", "image~2.png")
assert ectoplasm.decode_image("image~2.png") == b"Hello World!" # This works even if the steganographic layer was not applied (ectoplasm.encode_image), however since it relies on metadata, this produces a much less robust tag. For best results it is recommended to apply both encode_image and save_image (default behaviour in standalone mode).
```

- As a standalone program:<br />
`py -m ectoplasm "image.png" "Hello World!" --strength 1 --encode "image2.png"`<br />
`py -m ectoplasm "image.png" --strength 1 --decode "image2.png"` - Windows<br /><br />
`python3 -m ectoplasm "image.png" "Hello World!" --strength 1 --encode "image2.png"`<br />
`python3 -m ectoplasm "image.png" --strength 1 --decode "image2.png"` - Linux

## Intro
Historically, steganography has existed for millenia, since well before the common era. The concept refers to the practice of concealing information within an unsuspecting object that is typically used to convey unrelated information, thus achieving security through obscurity.<br />
Steganography can involve many forms of media from digital ones such as text and audio to physical ones such as invisible ink or social context and culture, but the most common form applied in modern times is to images, as they often exist in uncompressed form but contain large amounts of data, much of which is perceptually insignificant, but can be harnessed for storage of additional information.<br />

- An example of steganography through audio.<br />
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Wikipedia_wavefile_by_Coagula_-_logarihtmic_spectrogram.svg/375px-Wikipedia_wavefile_by_Coagula_-_logarihtmic_spectrogram.svg.png" alt="Wikipedia Spectrogram" width="256"/><br />
- An example of steganography through text.<br />
<img src="https://mizabot.xyz/u/tOqfndhpGJ5B4-5Gx4I2Ox4BdIcb/0bk3nkvh9n561.webp" alt="Rickroll Essay" width="256"/>

## Problem Definition
As technology develops and the adoption of automation advances, increasing amounts of artificial-intelligence (AI) generated data has approached the point of indistinguishability amongst human works. The focus for this project will mostly be **image diffusion-based neural networks**, with the ability to generate images closely mimicking human art, which simultaneously is a step forward in replicating and understanding human capabilities, yet a leap backwards in both the valuation of human works, credibility of photography and copyright, as well as internet safety for many. It presents risks from propaganda and deception to fraud and defamation, and unlike previously developed tools (including ones powered by artificial intelligence), these tools are harnessed with minimal expertise and effort, meaning they easily achieve widespread adoption. Misuse of generative AI has caused huge controversies, fear and distrust on a global scale.<br />
- An example of images produced by the [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) open-source models, the first to achieve a level playing field with commercial ones. Not only is it difficult to immediately tell that these are not human-drawn artworks or photos, such ambiguity causes divides amongst people and communities regarding plagiarism and integrity.<br />
<img src="https://mizabot.xyz/u/7JLllAMYnkDj_cj7nkE8Ag_G5L0/image.png" height="256"><img src="https://mizabot.xyz/u/7qDiiIaBBBwMQOd-PiCftt_-PuKsAg/image.png" height="256">

### Present Status/Literature Review
As of writing this abstract (November 2024), several companies have already begun incorporating detection methods for generative AI products. For instance:
- OpenAI implemented [C2PA](https://openai.com/index/understanding-the-source-of-what-we-see-and-hear-online/) in May this year, providing a global way for all images generated by their DALL·E tools to be traced, and therefore discouraging malicious use cases.
    - All images are tagged using content verification metadata. This makes it very simple to add to any image losslessly, without altering/reducing the quality of the content.
    - However, metadata is a rather fragile form of information storage. It is not considered image data by many programs and libraries, meaning unless an exact copy of the original file is sent, it is often lost, even unintentionally in translation.
    - One example would be a user right clicking a generated image shown by the ChatGPT tool, and selecting "Copy Image" in their browser. To the user, this would appear to copy the file exactly, but it is instead typically implemented in the backend as a lossless copy of the image's pixels into a PNG file. While this preserves the image (which generally satisfies the user), the reencoded image does not preserve any of the metadata, causing the tagging to be lost.
- AI-detection tools such as [GPTZero](https://gptzero.me/) for text and [sightengine](https://sightengine.com/detect-ai-generated-images) for images exist.
    - These require no existing tagging of the content, instead opting to provide access to their own NN models trained on existing generated content, and providing a sampled approximation of the probability for AI-created content.
    - These tools have been in development for a long time since the ethical concerns of widespread AI usage were raised, and have mostly seen use in schools and other educational departments, where untagged generated content often runs rampant.
    - However, these tools also have their downsides, and said downsides ironically mostly affect their main use case. The fact that they can provide false positive cases is a very strong case against their reliability, and the tools themselves have been the subject of large controversies regarding unjustified punishments.
- Standard watermarking has been employed in several cases, where an image has human-visible markings placed either in the corners/edges, or all across the image. While traditional, it is still a generally robust way for such a tag to be preserved in the image.
    - As a bonus, the tag is visible to humans, making it quickly detectable even without the need for an external tool.
    - On the other hand, the watermarks visibly alter the content of the image, typically replacing it with visible text or icons. While this is less of an issue with less invasive watermarks placed in corners, that also makes them easy to remove both intentionally and accidentally (such as crop edits). Alternatively, covering the whole image in watermarks tends to reduce their quality irreversibly.
    - In the physical world, [printers](https://en.wikipedia.org/wiki/Printer_tracking_dots) also utilise a form of steganography, printing tiled copies of a unique pattern across all pages which can be traced back to the source printer. This was a security measure against the forging of counterfeit documents or currencies, and relies on the difficulty for a human to tamper with such fingerprinting printed onto a piece of paper.

### Solutions?
What we need is a way for generated content to be detectable, that is not accidentally lost in translation, does not cause significant visible distortions or loss of quality, and never (or at least, is infinitesimally improbable to) cause false positives. That is where the motivation for this project comes into play.

## What is Ectoplasm?
Ectoplasm is designed to be a tool, or library, used to tag images through steganography. Rather than embed a large amount of information or a second smaller image within a larger one, it trades off some amount of data capacity for robustness. It aims to provide watermarking that is both near visually transparent and has the ability to resist common manipulation methods applied by users. Unlike traditional (or more strict) steganography algorithms, it does not attempt to hide the data from automated detection, as it is not used to transmit secret information. Its goal is to carry tracking information while remaining as undisruptive as possible.

### Algorithm Overview
As of current (November 2024), this implementation of Ectoplasm encodes and decodes data through the following methods:

#### Encoding
- Convert the data from text to binary if not already in that format
- Automatically select from a list of compression algorithms (currently Base128, [Brotli](https://en.wikipedia.org/wiki/Brotli) and [PAQ9A](https://en.wikipedia.org/wiki/PAQ)) to reduce the size of the data where possible
- Split the data into equally-sized chunks using [RaptorQ](https://datatracker.ietf.org/doc/html/rfc6330), which produces an arbitrary amount of output data that does not need to be fully recovered or recovered in the correct order to be decoded. Unlike with traditional error correction algorithms, this method enables data recovery even in circumstances where total damage to the data exceeds 50%, as long as sufficient contiguous blocks of data are able to be recovered (which tends to occur with image operations such as cropping).
- Create subchunk headers containing total and chunk sizes using [Reed-Solomon](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) encoding, as RaptorQ does not natively store this information
- Convert all prepared chunks into [Base45](https://www.ietf.org/archive/id/draft-faltstrom-base45-08.html), reducing to an alphanumeric representation with minimal overhead
- Encode the data into a sequence of [QR](https://en.wikipedia.org/wiki/QR_code) codes (this produces numerous monochrome 2D barcodes that are made to be easily distinguishable amidst noise or other image data)
- Tile the codes in multiple different ways (three are currently implemented; plain (square), diagonal, and [pythagorean](https://en.wikipedia.org/wiki/Pythagorean_tiling)), producing images the same size as the source image to tag
- Choose an amount of unique, mostly linearly independent colourspaces in the source image, equal to the amount of tiled images (the three currently hardcoded are U and V channels of [YUV](https://en.wikipedia.org/wiki/Y%E2%80%B2UV), and the L channel of [HSL](https://en.wikipedia.org/wiki/HSL_and_HSV))
- Treating all image data as floating points (range [0, 1]), apply the equation `Y = (⌊X * D - I / 2⌉ + I) / D`, where `X` is the source image, `I` is the previously computed QR code tiling as a boolean mask, and `D` is an optional input parameter controlling the strength/robustness of the data (higher values for `D` allow the data to be recovered easier, but causes larger distortions in the image to achieve this). Note that `D` is not necessarily a power of 2, meaning the steganographic data does not necessarily need to align with any bitplane of the image (although that would already be an assumption given the colourspace conversions)
- The colourspace of the output is converted back into the source colourspace (RGB or RGBA). In the implementation used for this library, the final colour downsampling from 32 bit float to 8 bit integer is performed using a combination of crosshatch/checkerboard and uniform random dithering, resulting in a closer approximation to the target values over a given area. The encoding is now done, and the output image may be used.

#### Decoding
- Split the image into the relevant colour spaces (U, V, L), and apply preprocess operations (currently hardcoded to restrict the maximum size of the image for sanity and computational complexity reasons, using area downsampling if required, taking advantage of previously dithered data where applicable)
- Using the previously established value for `D`, apply the equation `I = |(Y * D) % 1 - 0.5| * 2`, extracting the original mask with minimal loss
- Using a dedicated QR code detection tool (currently hardcoded to use [ZBar](https://zbar.sourceforge.net/) and [PyQRCode](https://github.com/mnooner256/pyqrcode)) to extract and decode the data chunks from said mask
- Reverse the Reed-Solomon encoding of the data headers to obtain the total and subchunk sizes
- Pass all valid chunks to RaptorQ for decoding
- Assuming a sufficient portion of the data was recoverable, the full message can now be retrieved through reversing any previously applied compression.

### Visualisation/Example
#### Assuming an input data of the following message and cover image pair:
```
I've come to make an announcement: Shadow the Hedgehog's a bitch-ass motherfucker, he pissed on my fucking wife.
That's right, he took his hedgehog-fuckin' quilly dick out and he pissed on my fucking wife, and he said his dick was "THIS BIG," and I said "that's disgusting," so I'm making a callout post on my Twitter.com:
"Shadow the Hedgehog, you've got a small dick.
It's the size of this walnut except WAY smaller.
And guess what? Here's what my dong looks like."
PFFFFFFFFGJT.
"That's right, baby.
All points, no quills, no pillows — look at that, it looks like two balls and a bong."
He fucked my wife, so guess what, I'm gonna fuck the Earth.
That's right, this is what you get: MY SUPER LASER PISS!!
Except I'm not gonna piss on the Earth, I'm gonna go higher; I'M PISSING ON THE MOON!
How do you like that, Obama?! I PISSED ON THE MOON, YOU IDIOT!
You have twenty-three hours before the piss DRRRROPLLLETS hit the fucking Earth, now get outta my fucking sight, before I piss on you too!
```
<img src="https://mizabot.xyz/u/kMamhLc0GJ5B4-5wx-4G43OBLMcL/initial.png" alt="Cover" width="427"/>

#### The encoding process can be visualised as follows:
<img src="https://mizabot.xyz/u/qr-8m6MDGJ5B4-5wx--AIHIDpsqp/image.png" alt="Mask Tilings" width="427"/><br />
<img src="https://mizabot.xyz/u/wt6_75vDBBieQePucNyCMPhyA6SrCQ/image.png" alt="Data Embeddings" width="427"/>

#### Producing a near visually transparent steganographic image of:
<img src="https://mizabot.xyz/u/6pmzu90GGJ5B4-5w3I4G4H-H7Msr/fountain.png" alt="Steganographic" width="427"/>

#### The data is now contained in a robust manner, and various modifications may be applied to the image without loss of the embedded message, such as:
<img src="https://mizabot.xyz/u/ws2Fj4o5GJ5B4-5w3II24H4BfOqp/fountain.png" alt="Tampered" width="427"/><br />
<img src="https://mizabot.xyz/u/tMjkprgDGJ5B4-5w3IOAOHIVvIep/image.png" alt="Tampered Embeddings" width="427"/><br />
<img src="https://mizabot.xyz/u/9o2T9bocGJ5B4-5w3IOGO3-BLOM7/image.png" alt="Tampered Masks" width="427"/>

#### Due to the redundancy achieved through the error correction and fountain coding, the original message can be recovered in its entirety from the tampered image, as sufficient QR codes remain for a full decode through Reed-Solomon and RaptorQ.

## Metrics
#### Assigned a custom text and image dataset for a relatively small-scale evaluation, Ectoplasm currently achieves the following success rates at preserving the data for these image operations, using the default `strength` value of 1:
- Directly decode: `162/162` **(100%)**
- Randomly flip/transpose: `162/162` **(100%)**
- Colour inversion: `155/155` **(100%)**
- Translate image -50%~+50%: `155/155` **(100%)**
- Blank out up to 50% of image: `149/150` **(99.3%)**
- Rotate image using bicubic sampling: `130/159` **(81.8%)**
- Skew image up to 50%, bicubic sampling: `117/147` **(79.6%)**
- Crop image to minimum 25% size: `165/165` **(100%)**
- Hueshift image 0~360 degrees: `170/170` **(100%)**
- Resize image between 25% to 225% of original size, lanczos sampling: `120/165` **(72.7%)**
- Apply "magik" grid distort filter, nearest neighbour sampling (visualised above): `157/157` **(100%)**
- Adjust image brightness -50%~+50%: `96/157` **(61.1%)**
- Compress image as JPEG: `1/185` **(0.5%)**

## Limitations
As seen from the metrics, any modifications that leave a large portion of the original image intact, such as nearest-neighbour sampling, do not cause many problems in decoding. However, the algorithm struggles slightly with bicubic or lanczos sampling, which may cause some images to fail to decode, when encoded at the default strength. The resistance to lossy compression is particularly poor, requiring encoding at around 1200% strength for a longer message to be recovered, which is unreasonable for general use, as it significantly distorts the image with noise and artifacts.<br />
Ectoplasm is designed for images of size 1024x1024 to 2048x2048, and compressed data sizes of 4B to 1KiB, but may function outside those ranges at reduced levels of redundancy, or lowered computational efficiency. The absolute maximum size that may be assigned for any usage is 64KiB, beyond which integer overflow of the unsigned 2-byte headers would prevent successful encoding. In practice however, the amount of valid tile spaces is usually the limiting factor, in which case there is not yet a straightforward way to predict.
<br />
One caveat worth mentioning is that due to the current dithering methods used to better visually hide the watermarking, random uniform noise is introduced into the image. This can significantly impact compression algorithms, for example a PNG that was 1MB in size may be inflated to 4MB with steganography applied, due to the effect of noise on entropy. This may change in the future with optimisations.

## Roadmap
As there still remain significant weaknesses in the algorithm's capability, development is likely to continue, with greater focus on the resistance to resampling and compression, as these are destructive/lossy transformations also designed mainly for visual transparency. Additional improvements to the general logic of the algorithm are also under construction, such as the ability to store or autodetect the strength value to fully eliminate the requirement of information required for decoding, as well as better management of payload size vs robustness.