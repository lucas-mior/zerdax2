import ctypes as ct
from numpy.ctypeslib import ndpointer
import platform

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
floaty = ct.c_double


def lfilter():
    func = lib.filter

    func.restype = None
    func.argtypes = [ndpointer(floaty, flags="C_CONTIGUOUS"),
                     ndpointer(floaty, flags="C_CONTIGUOUS"),
                     ndpointer(floaty, flags="C_CONTIGUOUS"),
                     ct.c_size_t]
    return func


def segments_distance():
    func = lib.segments_distance

    func.restype = ct.c_int32
    func.argtypes = [ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndpointer(ct.c_int32, flags="C_CONTIGUOUS")]
    return func


def lines_bundle():
    func = lib.lines_bundle

    func.restype = ct.c_int32
    func.argtypes = [ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                     ndpointer(ct.c_int32, flags="C_CONTIGUOUS"),
                     ct.c_int32, ct.c_int32]
    return func


segments_distance = segments_distance()
lfilter = lfilter()
lines_bundle = lines_bundle()
