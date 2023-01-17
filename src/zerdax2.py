#!/usr/bin/python

import sys
from algorithm import algorithm

if __name__ == '__main__':
    for filename in sys.argv[1:]:
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}: {fen}")
    for line in sys.stdin:
        filename = line[:-1]
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}: {fen}")
