import ctypes as ct
from numpy.ctypeslib import ndpointer
import platform
import os

uname = platform.uname()[0]
match uname:
    case "Linux":
        library = "./libzerdax.so"
    case "Windows":
        # library = r".\libzerdax.dll"
        print("this program doesn't work on windows yet")
        exit(1)
    case _:
        print(f"unsuported operating system: {uname}")
        exit(1)
lib = ct.CDLL(library)
floaty = ct.c_float
nthreads = os.cpu_count()


def lfilter():
    function = lib.filter

    function.restype = None
    function.argtypes = [ndpointer(floaty, flags="C_CONTIGUOUS"),
                         ndpointer(floaty, flags="C_CONTIGUOUS"),
                         ndpointer(floaty, flags="C_CONTIGUOUS"),
                         ct.c_size_t,
                         ct.c_int]
    return function


def segments_distance():
    function = lib.segments_distance

    function.restype = ct.c_int32
    function.argtypes = [ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                         ndpointer(ct.c_int32, flags="C_CONTIGUOUS")]
    return function


def lines_bundle():
    function = lib.lines_bundle

    function.restype = ct.c_int32
    function.argtypes = [ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                         ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                         ct.c_int32, ct.c_int32]
    return function


segments_distance = segments_distance()
lfilter = lfilter()
lines_bundle = lines_bundle()
