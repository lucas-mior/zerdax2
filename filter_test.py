import numpy as np
import cv2
import sys

from c_load import lfilter
import draw

WIDTH_INPUT = 512


def filter_test(filename):
    BGR = cv2.imread(filename)
    image = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)

    aspect_ratio = WIDTH_INPUT / image.shape[1]
    height_input = round(image.shape[0] * aspect_ratio)
    image = cv2.resize(image, (WIDTH_INPUT, height_input))

    f = np.array(image, dtype='float64')
    weights = np.empty(image.shape, dtype='float64')
    normalization = np.empty(image.shape, dtype='float64')
    g = np.empty(image.shape, dtype='float64')

    lfilter(f, g, weights, normalization, f.shape[0])
    lfilter(g, f, weights, normalization, f.shape[0])
    lfilter(f, g, weights, normalization, f.shape[0])

    g = np.round(g)
    g = np.clip(g, 0, 255)
    g = np.array(g, dtype='uint8')
    draw.save("lfilter", g, title="lfilter.png")


if __name__ == "__main__":
    for i in range(5):
        for filename in sys.argv[1:]:
            filter_test(filename)
