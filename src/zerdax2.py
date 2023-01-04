#!/usr/bin/python

import argparse
from algorithm import algorithm


def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image', type=str, default=None,
                        help='Image filename')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseargs()
    image = args.image
    print(f"============ zerdax {image} ============")
    fen = algorithm(image)
    print("FEN:", fen)
