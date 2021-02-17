from PIL import Image
from base64 import decodestring
import io
import os
import base64
import cv2
import numpy as np
import argparse
from indoor import tiles_generator, pred_evaluator, swapperTile

my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('--tile', '--t', action='store', type=str, required=True)
my_parser.add_argument('--input', '--i', action='store', type=str, required=True)
my_parser.add_argument('--output', '--o', action='store', type=str, required=True)

def main(tile_path, input_image, output_image):

    file_path = tile_path
    im = Image.open(input_image).convert('RGB')
    width, height = im.size
    im = im.resize((width//3, height//3))
    pred, img_original = pred_evaluator(im)
    predicted_image = pred
    original_image = img_original
    filled_tiles = tiles_generator(file_path, pred)
    print("Filled tile", filled_tiles.shape)
    final = swapperTile(pred, img_original, filled_tiles)
    cv2.imwrite(output_image, final)

if __name__=="__main__":
    args = my_parser.parse_args()
    main(args.tile, args.input, args.output)