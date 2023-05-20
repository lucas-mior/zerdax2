import ctypes as ct
from numpy.ctypeslib import ndpointer as ndp
import platform

uname = platform.uname()[0]
if uname == "Linux":
    library = "./libzerdax.so"
elif uname == "Windows":
    # library = r".\libzerdax.dll"
    print("This ṕrogram doesn't work on windows yet")
    exit(1)
else:
    print(f"Unsuported operating system: {uname}")
    exit(1)
lib = ct.CDLL(library)


def lfilter():
    func = lib.filter

    func.restype = None
    func.argtypes = [ndp(ct.c_double, flags="C_CONTIGUOUS"),
                     ct.c_size_t, ct.c_size_t,
                     ndp(ct.c_double, flags="C_CONTIGUOUS"),
                     ndp(ct.c_double, flags="C_CONTIGUOUS"),
                     ndp(ct.c_double, flags="C_CONTIGUOUS"),
                     ct.c_double]
    return func


def segments_distance():
    func = lib.segments_distance

    func.restype = ct.c_int32
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS")]

    return func


def lines_bundle():
    func = lib.lines_bundle

    func.restype = ct.c_int32
    func.argtypes = [ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndp(ct.c_int32, flags="C_CONTIGUOUS"),
                     ct.c_int32, ct.c_int32]

    return func


segments_distance = segments_distance()
lfilter = lfilter()
lines_bundle = lines_bundle()
