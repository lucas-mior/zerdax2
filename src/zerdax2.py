#!/usr/bin/python

import sys
import logging as log
import argparse

from algorithm import algorithm


def parse_args(args):
    loglevels = {
        'critical': log.CRITICAL,
        'error': log.ERROR,
        'warning': log.WARNING,
        'info': log.INFO,
        'debug': log.DEBUG
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--loglevel', dest='level',
                        default='error', choices=loglevels.keys(),
                        help='verbosity level')
    args = parser.parse_args(args)

    level = loglevels[args.level]
    return args.filenames, level


if __name__ == '__main__':
    filenames, loglevel = parse_args(sys.argv[1:])
    log.basicConfig(level=loglevel, format='[%(levelname)s] %(message)s')

    for filename in filenames:
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}): {fen}")
    for line in sys.stdin:
        filename = line[:-1]
        print(f"============ zerdax2.py {filename} ============")
        fen = algorithm(filename)
        print(f"FEN({filename}): {fen}")
