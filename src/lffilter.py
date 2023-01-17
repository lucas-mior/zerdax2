import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np


def ffilter(image, h=1):
    image = np.array(image, dtype='float64')
    f = np.copy(image)
    W = np.zeros(image.shape, dtype='float64')
    N = np.zeros(image.shape, dtype='float64')
    g = np.zeros(image.shape, dtype='float64')

    lf = ct.CDLL("./libffilter.so")
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
