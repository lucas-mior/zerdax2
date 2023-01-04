import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np


def ffilter(image):
    f = np.copy(image/255)
    W = np.copy(image/255) * 0
    N = np.copy(image/255) * 0
    g = np.copy(image/255)

    lf = ct.CDLL("./libffilter.so")
    lf_filter = lf.ffilter

    lf_filter.restype = None
    lf_filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ct.c_size_t, ct.c_size_t,
                          ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ndp(ct.c_double, flags="C_CONTIGUOUS"),
                          ndp(ct.c_double, flags="C_CONTIGUOUS")]

    lf_filter(f, f.shape[0], f.shape[1], W, N, g)
    lf_filter(g, f.shape[0], f.shape[1], W, N, f)
    lf_filter(f, f.shape[0], f.shape[1], W, N, g)

    return np.array(g*255, dtype='uint8')
