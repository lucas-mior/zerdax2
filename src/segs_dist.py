import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import platform


def create_func():
    if platform.uname()[0] == "Windows":
        library = r".\segs.dll"
    elif platform.uname()[0] == "Linux":
        library = "./segs.so"
    lib = ct.CDLL(library)
    func = lib.segs

    func.restype = ct.c_double
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS")]

    return func


segments_distance = create_func()
