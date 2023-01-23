import sys
import numpy as np
import algorithm as algo
import cv2

from misc import COLORS, SYMBOLS


def save(name, image):
    if not hasattr(save, "i"):
        save.i = 0
    save.i += 1
    title = f"{algo.img.basename}_{save.i:04d}_{name}.png"
    print(f"saving {title}...")
    cv2.imwrite(title, image)


def addweighted(image, canvas, w1=0.5, w2=0.5):
    cv2.addWeighted(image, w1, canvas, w2, 0, canvas)
    return canvas


def points(image, inters):
    canvas = np.zeros(image.shape, dtype='uint8')

    for inter in inters:
        for i, p in enumerate(inter):
            cv2.circle(canvas, p, radius=5,
                       color=(30+i*2, 0, 225-i*2), thickness=-1)

    canvas = addweighted(image, canvas)
    return canvas


def lines(image, vert, hori=None):
    canvas = np.zeros(image.shape, dtype='uint8')
    vert = np.array(vert)

    def _draw_lines(canvas, lines, color, number=True):
        for i, line in enumerate(lines[:, :4]):
            x0, y0, x1, y1 = line
            cv2.line(canvas, (x0, y0), (x1, y1),
                     color=color, thickness=2)
            if number:
                x, y = round((x0+x1)/2), round((y0+y1)/2)
                cv2.putText(canvas, str(i), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3,
                            color=color, thickness=2)
        return canvas

    if hori is not None:
        hori = np.array(hori)
        canvas = _draw_lines(canvas, vert, (255, 0, 0))
        canvas = _draw_lines(canvas, hori, (0, 255, 0))
    else:
        canvas = _draw_lines(canvas, vert, (0, 0, 255), number=False)

    canvas = addweighted(image, canvas)
    return canvas


def squares(image, squares):
    canvas = np.zeros(image.shape, dtype='uint8')
    scale = 2.5 * (image.shape[1]/1920)

    def _draw_square(canvas, coord, color, name):
        cv2.drawContours(canvas, [coord], -1, color=color, thickness=3)
        color = [max(0, c - 80) for c in color]
        cv2.putText(canvas, name, (coord[0, 0]+5, coord[0, 1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color=color, thickness=2)
        return canvas

    canvas = _draw_square(canvas, squares[0, 0, :4], [255, 0, 0], "A1")
    canvas = _draw_square(canvas, squares[4, 3, :4], [0, 255, 0], "E4")
    canvas = _draw_square(canvas, squares[2, 4, :4], [0, 0, 255], "C5")
    canvas = _draw_square(canvas, squares[7, 7, :4], [0, 255, 255], "H8")

    canvas = addweighted(image, canvas)
    return canvas


def boxes(pieces, image):
    canvas = np.zeros(image.shape, dtype='uint8')
    thick = round(2.4 * (image.shape[0] / 1280))

    for piece in pieces:
        x0, y0, x1, y1, conf, num, _ = piece
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        conf, num = round(float(conf), 2), int(num)
        color = COLORS[num]
        symbol = SYMBOLS[num]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color=color, thickness=thick)
        cv2.putText(canvas, f"{symbol} {conf}", (x0-5, y0-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thick)

    canvas = addweighted(image, canvas)
    return canvas


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        image = cv2.imread(sys.argv[1])
        canvas = cv2.imread(sys.argv[2])
        canvas = addweighted(image, canvas)
        save("addweighted", canvas)
