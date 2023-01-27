import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import platform


if platform.uname()[0] == "Windows":
    # library = r".\libzerdax.dll"
    print("This ṕrogram doesn't work on windows yet")
    exit(1)
elif platform.uname()[0] == "Linux":
    library = "./libzerdax.so"
lib = ct.CDLL(library)


def lfilter():
    lfilter = lib.filter

    lfilter.restype = None
    lfilter.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                        ct.c_size_t, ct.c_size_t,
                        ndp(ct.c_double, flags="C_CONTIGUOUS"),
                        ndp(ct.c_double, flags="C_CONTIGUOUS"),
                        ndp(ct.c_double, flags="C_CONTIGUOUS"),
                        ct.c_double]
    return lfilter


def segments_distance():
    func = lib.segments_distance

    func.restype = ct.c_double
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS")]

    return func


segments_distance = segments_distance()
lfilter = lfilter()
