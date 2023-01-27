import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np
import platform


if platform.uname()[0] == "Windows":
    # library = r".\libzerdax.dll"
    print("This ṕrogram doesn't work on windows yet")
    exit(1)
elif platform.uname()[0] == "Linux":
    library = "./libzerdax.so"
lib = ct.CDLL(library)


def filter(image, h=1):
    image = np.array(image, dtype='float64')
    f = np.copy(image)
    W = np.zeros(image.shape, dtype='float64')
    N = np.zeros(image.shape, dtype='float64')
    g = np.zeros(image.shape, dtype='float64')

    filter = lib.filter

    filter.restype = None
    filter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                       ct.c_size_t, ct.c_size_t,
                       ndp(ct.c_double, flags="C_CONTIGUOUS"),
                       ndp(ct.c_double, flags="C_CONTIGUOUS"),
                       ndp(ct.c_double, flags="C_CONTIGUOUS"),
                       ct.c_double]

    filter(f, f.shape[0], f.shape[1], W, N, g, h)
    filter(g, f.shape[0], f.shape[1], W, N, f, h)
    filter(f, f.shape[0], f.shape[1], W, N, g, h)

    g = np.round(g)
    g = np.clip(g, 0, 255)
    return np.array(g, dtype='uint8')


def segments_distance():
    func = lib.segments_distance

    func.restype = ct.c_double
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS")]

    return func


segments_distance = segments_distance()
