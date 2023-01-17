#!/usr/bin/python

import sys
import os
import glob
from algorithm import algorithm


if __name__ == '__main__':
    for image in sys.argv[1:]:
        print(f"============ zerdax2.py {image} ============")
        fen = algorithm(image)
        print("FEN:", fen)
        files = glob.glob('exp*')
        for f in files:
            os.rmdir(f)
