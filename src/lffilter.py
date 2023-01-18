import sys
import cv2
import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np
import auxiliar as aux
import platform
import logging as log


def ffilter(image, h=1):
    image = np.array(image, dtype='float64')
    f = np.copy(image)
    W = np.zeros(image.shape, dtype='float64')
    N = np.zeros(image.shape, dtype='float64')
    g = np.zeros(image.shape, dtype='float64')

    if platform.uname()[0] == "Windows":
        # library = "./libffilter.dll"
        log.critical("Filter library does not run on windows yet")
        exit()
    elif platform.uname()[0] == "Linux":
        library = "./libffilter.so"
    lf = ct.CDLL(library)
    lf_filter = lf.ffilter

    lf_filter.restype = None
    lf_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ct.c_size_t, ct.c_size_t,
                          ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ct.c_double]

    lf_filter(f, f.shape[0], f.shape[1], W, N, g, h)
    lf_filter(g, f.shape[0], f.shape[1], W, N, f, h)
    lf_filter(f, f.shape[0], f.shape[1], W, N, g, h)

    g = np.round(g)
    g = np.clip(g, 0, 255)
    return np.array(g, dtype='uint8')


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = ffilter(image)
        aux.save("ffilter", image)
