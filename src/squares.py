import cv2
import numpy as np
import logging as log

import auxiliar as aux
import drawings as draw


def calc_squares(img):
    vert, hori = img.vert, img.hori
    if (lv := len(vert)) != 9 or (lh := len(hori)) != 9:
        log.error("There should be 9 vertical lines and",
                  "9 horizontal lines")
        log.error(f"Got {lv} vertical and {lh} horizontal lines")
        canvas = draw.lines(img.gray3ch, vert, hori)
        aux.save(img, "magic_lines", canvas)
        exit()

    inters = aux.calc_intersections(img.gray3ch, vert, hori)
    if inters.shape != (9, 9, 2):
        log.error("There should be 81 intersections",
                  "in 9 rows and 9 columns")
        log.error(f"{inters.shape=}")
        canvas = draw.points(img.gray3ch, inters)
        aux.save(img, "intersections", canvas)
        exit()

    intersq = inters.reshape(9, 9, 1, 2)
    intersq = np.flip(intersq, axis=1)
    squares = np.zeros((8, 8, 5, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    if aux.debugging():
        canvas = draw.squares(img.board, squares)
        aux.save(img, "A1E4C5H8", canvas)
    squares = np.array(squares, dtype='float32')
    # scale to input size
    squares[:, :, :4, 0] /= img.bfact
    squares[:, :, :4, 1] /= img.bfact
    # position board bounding box
    squares[:, :, :4, 0] += img.x0
    squares[:, :, :4, 1] += img.y0

    img.squares = np.array(np.round(squares), dtype='int32')
    img.squares = fill_squares(img.squares, img.pieces)
    img.squares = check_bottom_right(img.BGR, img.squares)
    if aux.debugging():
        canvas = draw.squares(img.BGR, img.squares)
        aux.save(img, "A1E4C5H8", canvas)
    return img


def fill_squares(squares, pieces):
    log.info("filling squares...")
    piece_y_tol = round(abs(squares[0, 0, 0, 1] - squares[7, 7, 0, 1]) / 22)
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            got_piece = False
            for piece in pieces:
                x0, y0, x1, y1, _, number = piece[:6]
                xm = round((x0 + x1)/2)
                y = round(y1) - piece_y_tol
                if cv2.pointPolygonTest(sq[:4], (xm, y), True) >= 0:
                    sq[4] = [1, number]
                    got_piece = True
                    pieces.remove(piece)
                    break
            if not got_piece:
                sq[4] = [0, -1]

    return squares


def check_bottom_right(image, squares):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a8 = np.copy(squares[7, 0])
    contour = a8[:4]
    frame = cv2.boundingRect(contour)
    x0, y0, dx, dy = frame
    contour[:, 0] -= x0
    contour[:, 1] -= y0
    b = image[y0:y0+dy, x0:x0+dx]
    mask1 = np.zeros(b.shape, dtype='uint8')
    cv2.drawContours(mask1, [contour], -1, 255, -1)
    mask0 = cv2.bitwise_not(mask1)
    mean0 = round(cv2.mean(b, mask=mask0)[0])
    mean1 = round(cv2.mean(b, mask=mask1)[0])
    if a8[4, 1] < 0:  # no piece
        pass
    elif a8[4, 1] <= 6:  # white piece
        mean1 -= 30
    else:  # black piece
        mean1 += 30
    if mean1 < mean0:
        if squares[0, 0, 0, 1] > squares[1, 0, 0, 1]:
            squares = np.rot90(squares, k=1)
        else:
            squares = np.rot90(squares, k=-1)
    return squares
