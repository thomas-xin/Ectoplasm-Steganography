import argparse
parser = argparse.ArgumentParser()
parser.add_argument("image_path")
parser.add_argument("message", default="")
parser.add_argument("-s", "--strength", default=1, required=False)
parser.add_argument("-c", "--compress", action="store_true", default=True, required=False)
parser.add_argument("-e", "--encode", nargs="?", const=True, default=None, required=False)
args = parser.parse_args()

if args.encode is not None:
    from .ectoplasm import encode_image, save_image
    im = encode_image(args.image_path, args.message, strength=args.strength, compress=args.compress)
    if not isinstance(args.encode, str):
        args.encode = (p := args.image_path.rsplit(".", 1))[0] + "~." + p[1]
    save_image(im, args.message, args.encode)
    print(f'Output successfully written to "{args.encode}".')
else:
    from .ectoplasm import decode_image
    message = decode_image(args.image_path, strength=args.strength)
    if args.message:
        assert message == args.message.encode("utf-8"), "Decoded message does not match."
    else:
        import sys
        sys.stdout.buffer.write(message)