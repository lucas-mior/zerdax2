import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import platform


def segments_distance(line0, line1):
    if platform.uname()[0] == "Windows":
        library = r".\segs.dll"
    elif platform.uname()[0] == "Linux":
        library = "./segs.so"
    lib = ct.CDLL(library)
    func = lib.segs

    func.restype = ct.c_double
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS")]

    dist = func(line0[:4], line1[:4])
    return dist
