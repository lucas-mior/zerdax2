import cv2
import numpy as np

import auxiliar as aux
import drawings as draw

WLEN = 640


def find_corners(img):
    # img = create_cannys(img)
    # vert, hori = magic_lines(img)
    # inters = aux.calc_intersections(img.gray3ch, vert, hori)
    # canvas = draw.intersections(img.gray3ch, inters)
    # aux.save(img, "intersections", canvas)
    inters = None

    # img.corners = calc_corners(img, inters)

    intersq = inters.reshape(9, 9, 1, 2)
    intersq = np.flip(intersq, axis=1)
    squares = np.zeros((8, 8, 4, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    canvas = draw.squares(img.board, squares)
    aux.save(img, "A1E4C5H8", canvas)
    squares = np.float32(squares)
    # scale to input size
    squares[:, :, :, 0] /= img.bfact
    squares[:, :, :, 1] /= img.bfact
    # position board bounding box
    squares[:, :, :, 0] += img.x0
    squares[:, :, :, 1] += img.y0

    img.squares = np.array(np.round(squares), dtype='int32')
    return img


def perspective_transform(img):
    print("transforming perspective...")
    BR = img.corners[0]
    BL = img.corners[1]
    TR = img.corners[2]
    TL = img.corners[3]
    orig_points = np.array(((TL[0], TL[1]), (TR[0], TR[1]),
                            (BR[0], BR[1]), (BL[0], BL[1])), dtype="float32")

    width = WLEN
    height = WLEN
    img.wwidth = width
    img.wheigth = height

    newshape = np.array([[0, 0], [width-1, 0],
                        [width-1, height-1], [0, height-1]], dtype="float32")
    print("creating transform matrix...")
    img.warpMatrix = cv2.getPerspectiveTransform(orig_points, newshape)
    _, img.warpInvMatrix = cv2.invert(img.warpMatrix)
    print("warping image...")
    img.wg = cv2.warpPerspective(img.G, img.warpMatrix, (width, height))
    img.wv = cv2.warpPerspective(img.V, img.warpMatrix, (width, height))
    # aux.save(img, "warpclaheG", img.wg)
    # aux.save(img, "warpclaheV", img.wv)

    return img
