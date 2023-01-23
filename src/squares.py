import cv2
import numpy as np
import logging as log

import algorithm as algo
import drawings as draw
import lines as li


def calc_squares(img, inters):
    intersq = inters.reshape(9, 9, 1, 2)
    intersq = np.flip(intersq, axis=1)
    squares = np.zeros((8, 8, 5, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    if algo.debugging():
        canvas = draw.squares(img.BGR, squares)
        draw.save("A1E4C5H8", canvas)

    log.info("filling squares...")
    squares, pieces = fill_squares(squares, img.pieces)
    if len(pieces) > 0:
        squares, pieces = fill_squares(squares, img.pieces, force=True)
    img.squares, changed = check_colors(img.BGR, squares)

    if algo.debugging() and changed:
        canvas = draw.squares(img.BGR, img.squares)
        draw.save("A1E4C5H8", canvas)
    return img


def fill_squares(squares, pieces, force=False):
    piece_y_tol = abs(squares[0, 0, 0, 1] - squares[7, 7, 0, 1]) / 22
    piece_y_tol = round(piece_y_tol)
    for index in np.ndindex(squares.shape[:2]):
        sq = squares[index]
        if sq[4, 0] == 1:
            continue
        dmax = 0
        for piece in pieces:
            x0, y0, x1, y1, _, number = piece[:6]
            xm = round((x0 + x1)/2)
            y = round(y1) - piece_y_tol
            if not force:
                if (d := cv2.pointPolygonTest(sq[:4], (xm, y), True)) >= 0:
                    if d > dmax:
                        npiece = piece
                        dmax = d
            else:
                if (d := cv2.pointPolygonTest(sq[:4], (xm, y-5), True)) >= 0:
                    if d > dmax:
                        npiece = piece
                        dmax = d
                elif (d := cv2.pointPolygonTest(sq[:4], (xm, y+2), True)) >= 0:
                    if d > dmax:
                        npiece = piece
                        dmax = d

        try:
            sq[4] = [1, npiece[5]]
            pieces.remove(npiece)
        except Exception:
            sq[4] = [0, -1]
    return squares, pieces


def check_colors(image, squares):
    changed = False
    player = "down"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _rotate(squares):
        nonlocal player, changed
        changed = True
        if squares[0, 0, 0, 1] > squares[1, 0, 0, 1]:
            player = "left"
            squares = np.rot90(squares, k=1)
        else:
            player = "right"
            squares = np.rot90(squares, k=-1)
        return squares

    def _calc_means(col, row):
        sq = np.copy(squares[col, row])
        contour = sq[:4]
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
        if sq[4, 1] < 0:  # no piece
            pass
        elif sq[4, 1] <= 6:  # white piece
            mean1 -= 30
        else:  # black piece
            mean1 += 30
        return mean1, mean0

    change_votes = 0
    for col in range(7):
        row = 7 - col
        mean1, mean0 = _calc_means(col, row)
        if mean1 < mean0:
            change_votes += 1
            if mean0 - mean1 > 40:
                change_votes += 4
                break
    if change_votes > 4:
        squares = _rotate(squares)

    white = squares[(squares[..., 4, 1] <= 6) & (squares[..., 4, 1] >= 0)]
    black = squares[squares[..., 4, 1] > 6]
    if player == "down":
        meanwhite = np.median(white[:, 0, 1])
        meanblack = np.median(black[:, 0, 1])
    else:
        if player == "left":
            meanwhite = np.median(white[:, 0, 0])
            meanblack = np.median(black[:, 0, 0])
        else:
            meanwhite = np.median(black[:, 0, 0])
            meanblack = np.median(white[:, 0, 0])
    if meanwhite < meanblack:
        squares = np.rot90(squares, k=2)
        changed = True
    return squares, changed
