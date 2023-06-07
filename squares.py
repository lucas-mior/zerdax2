import cv2
import numpy as np
import logging as log


def calculate(inters):
    intersq = inters.reshape(9, 9, 1, 2)
    intersq = np.flip(intersq, axis=1)
    squares = np.zeros((8, 8, 5, 2), dtype='int32')
    for i in range(0, 8):
        for j in range(0, 8):
            squares[i, j, 0] = intersq[i, j]
            squares[i, j, 1] = intersq[i+1, j]
            squares[i, j, 2] = intersq[i+1, j+1]
            squares[i, j, 3] = intersq[i, j+1]

    return squares


def fill(squares, pieces, force=False):
    log.info("filling squares...")
    piece_y_tol = abs(squares[0, 0, 0, 1] - squares[7, 7, 0, 1]) / 22
    piece_y_tol = round(piece_y_tol)
    for index in np.ndindex(squares.shape[:2]):
        square = squares[index]
        if square[4, 0] == 1:
            continue
        dmax = 0
        npiece = None
        for piece in pieces:
            if piece[5] == -1:
                continue
            x0, y0, x1, y1, _, number = piece[:6]
            xm = round((x0 + x1)/2)
            y = round(y1) - piece_y_tol
            if not force:
                dist = cv2.pointPolygonTest(square[:4], (xm, y), True)
                if dist >= 0:
                    if dist > dmax:
                        npiece = piece
                        dmax = dist
            else:
                dist1 = cv2.pointPolygonTest(square[:4], (xm, y-5), True)
                dist2 = cv2.pointPolygonTest(square[:4], (xm, y+2), True)
                if dist1 >= 0:
                    if dist1 > dmax:
                        npiece = piece
                        dmax = dist1
                elif dist2 >= 0:
                    if dist2 > dmax:
                        npiece = piece
                        dmax = dist2

        if npiece is not None:
            square[4] = [1, npiece[5]]
            npiece[5] = -1
        else:
            square[4] = [0, -1]

    if len(pieces) > 0 and not force:
        squares = fill(squares, pieces, force=True)
        return squares

    return squares


def calculate_means(image, squares, col, row):
    log.debug(f"calculate_means({col=}, {row=})")

    square = np.copy(squares[col, row])
    if square[4, 1] >= 0:
        log.debug("square is ocupied, skipping for calc_means")
        if col < 6:
            col += 2
        else:
            return 0, 0
        return calculate_means(image, squares, col, row)

    contour = square[:4]
    frame = cv2.boundingRect(contour)
    x0, y0, dx, dy = frame
    contour[:, 0] -= x0
    contour[:, 1] -= y0

    box = image[y0:y0+dy, x0:x0+dx]
    mask_in = np.zeros(box.shape, dtype='uint8')
    cv2.drawContours(mask_in, [contour], -1, 255, -1)
    mask_out = cv2.bitwise_not(mask_in)

    mean_out = round(cv2.mean(box, mask=mask_out)[0])
    mean_in = round(cv2.mean(box, mask=mask_in)[0])
    if (col + row) % 2 == 0:
        mean_in, mean_out = mean_out, mean_in
    return mean_in, mean_out


def check_colors(image, squares):
    changed = False
    shot_position = "down"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _rotate(squares):
        nonlocal shot_position, changed
        changed = True
        if squares[0, 0, 0, 1] > squares[1, 0, 0, 1]:
            shot_position = "left"
            squares = np.rot90(squares, k=-1)
        else:
            shot_position = "right"
            squares = np.rot90(squares, k=+1)
        return squares

    change_votes = 0
    checked = 0
    for col in range(7):
        for row in range(7):
            mean_in, mean_out = calculate_means(image, squares, col, row)
            checked += 1
            if mean_in < mean_out:
                change_votes += 1
            if change_votes > 6 or checked >= 12:
                break
        if change_votes > 6 or checked >= 12:
            break
    if change_votes > 6:
        squares = _rotate(squares)

    squares_pieces_white = squares[(squares[..., 4, 1] <= 6)
                                   & (squares[..., 4, 1] >= 0)]
    squares_pieces_black = squares[squares[..., 4, 1] > 6]

    if shot_position == "down":
        mean_squares_pieces_white = np.median(squares_pieces_white[:, 0, 1])
        mean_squares_pieces_black = np.median(squares_pieces_black[:, 0, 1])
    elif shot_position == "right":
        mean_squares_pieces_white = np.median(squares_pieces_white[:, 0, 0])
        mean_squares_pieces_black = np.median(squares_pieces_black[:, 0, 0])
    elif shot_position == "left":
        mean_squares_pieces_white = np.median(squares_pieces_black[:, 0, 0])
        mean_squares_pieces_black = np.median(squares_pieces_white[:, 0, 0])

    if mean_squares_pieces_white > mean_squares_pieces_black:
        squares = np.rot90(squares, k=2)
        changed = True
    return squares, changed
