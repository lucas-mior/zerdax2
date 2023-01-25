import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import numpy as np
import platform


def segments_distance(line0, line1):
    line0 = np.array(line0[:4], dtype='float64')
    line1 = np.array(line1[:4], dtype='float64')

    if platform.uname()[0] == "Windows":
        library = r".\segs.dll"
    elif platform.uname()[0] == "Linux":
        library = "./segs.so"
    lib = ct.CDLL(library)
    func = lib.segs

    func.restype = ct.c_double
    func.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                     ndp(ct.c_double, flags="C_CONTIGUOUS")]

    dist = func(line0, line1)
    return dist
