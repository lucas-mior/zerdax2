#!/usr/bin/python

import sys
import logging as log
import argparse

import algorithm


if __name__ == '__main__':
    loglevels = {
        'critical': log.CRITICAL,
        'error': log.ERROR,
        'warning': log.WARNING,
        'info': log.INFO,
        'debug': log.DEBUG
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('-v', dest='verbose',
                        default='error', choices=loglevels.keys(),
                        help='verbosity level')
    args = parser.parse_args(sys.argv[1:])

    level = loglevels[args.verbose]
    log.basicConfig(level=level, format='[%(levelname)s] %(message)s')

    for filename in args.filenames:
        print(f"============ zerdax2.py {filename} ============")
        # try:
        fen = algorithm.main(filename)
        # except Exception:
        #     continue
        print(f"FEN({filename}): {fen}")
