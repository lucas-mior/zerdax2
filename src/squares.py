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

    squares = fill_squares(squares, img.pieces)
    img.squares = np.copy(squares)
    img.squares = check_colors(img.BGR, img.squares)

    if algo.debugging() and not np.array_equal(squares, img.squares):
        canvas = draw.squares(img.BGR, img.squares)
        draw.save("A1E4C5H8", canvas)
    return img


def fill_squares(squares, pieces):

    log.info("filling squares...")
    squares, pieces = iterate(squares, pieces)
    if len(pieces) > 0:
        squares, pieces = iterate(squares, pieces, force=True)
    return squares


def check_colors(image, squares):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _rotate(squares):
        if squares[0, 0, 0, 1] > squares[1, 0, 0, 1]:
            squares = np.rot90(squares, k=1)
        else:
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
    if change_votes > 4:
        squares = _rotate(squares)
    return squares


def iterate(squares, pieces, force=False):
    def _select_piece(sq, possible):
        if len(possible) == 1:
            return possible[0]

        xc = np.sum(sq[:, 0])/4
        yc = np.sum(sq[:, 1])/4
        dists = []
        for i, p in enumerate(possible):
            x, y = round((p[0] + p[2])/2), round(p[3] + piece_y_tol)
            dists.append(round(li.length((xc, yc, x, y))))
        possible = possible[np.argsort(dists)]
        return possible[0]

    piece_y_tol = abs(squares[0, 0, 0, 1] - squares[7, 7, 0, 1]) / 22
    piece_y_tol = round(piece_y_tol)
    for i in range(7, -1, -1):
        for j in range(0, 8):
            sq = squares[j, i]
            possible = []
            if sq[4, 0] == 1:
                continue
            for piece in pieces:
                x0, y0, x1, y1, _, number = piece[:6]
                xm = round((x0 + x1)/2)
                y = round(y1) - piece_y_tol
                if cv2.pointPolygonTest(sq[:4], (xm, y), True) >= 0:
                    possible.append(piece)
                elif force:
                    if cv2.pointPolygonTest(sq[:4], (xm, y-5), True) >= 0:
                        possible.append(piece)
                    elif cv2.pointPolygonTest(sq[:4], (xm, y+2), True) >= 0:
                        possible.append(piece)
            if len(possible) > 0:
                possible = np.array(possible)
                piece = _select_piece(sq, possible).tolist()
                sq[4] = [1, piece[5]]
                pieces.remove(piece)
            else:
                sq[4] = [0, -1]
    return squares, pieces
