import numpy as np
import cv2
import sys

from c_load import lfilter
import draw

WIDTH_BOARD = 512  # used for board crop and perspective transform


def filter_test(filename, i):
    BGR = cv2.imread(filename)
    image = cv2.cvtColor(BGR, cv2.COLOR_BGR2GRAY)

    aspect_ratio = WIDTH_BOARD / image.shape[1]
    height_input = round(image.shape[0] * aspect_ratio)
    image = cv2.resize(image, (WIDTH_BOARD, height_input))

    f = np.array(image, dtype='float64')
    weights = np.empty(image.shape, dtype='float64')
    normalization = np.empty(image.shape, dtype='float64')
    g = np.empty(image.shape, dtype='float64')

    for i in range(10000):
        lfilter(f, g, weights, normalization, f.shape[0])
        lfilter(g, f, weights, normalization, f.shape[0])

    g = np.round(g)
    g = np.clip(g, 0, 255)
    g = np.array(g, dtype='uint8')
    title = f"lfilter_{i}_{filename}"
    # draw.save(f"lfilter_{i}_{filename}.png", g, title=title)


if __name__ == "__main__":
    # for i in range(5):
    #     for j, filename in enumerate(sys.argv[1:]):
    #         filter_test(filename, j)
    filter_test("0test.jpg", 0)
