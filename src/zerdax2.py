#!/usr/bin/python

import sys
import logging as log
from algorithm import algorithm

if __name__ == '__main__':
    level = log.ERROR
    fmt = '[%(levelname)s] %(message)s'
    log.basicConfig(level=level, format=fmt)

    for filename in sys.argv[1:]:
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}): {fen}")
    for line in sys.stdin:
        filename = line[:-1]
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}): {fen}")
