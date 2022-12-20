import yolov5.detect as yolo
import cv2
import numpy as np


def find_pieces(img):
    pieces = yolo.run(weights="yolov5/runs/train/exp1/weights/best.pt",
                      source=img.BGR_name,
                      data="yolov5/zerdax2.yaml",
                      conf_thres=0.7,  # confidence threshold
                      iou_thres=0.45,  # NMS IOU threshold
                      max_det=32,  # maximum detections per image
                      save_txt=False,  # save results to *.txt
                      save_conf=True,  # save confidences in --save-txt labels
                      project='.',  # save results to project/name
                      name='exp',  # save results to project/name
                      )
    print(pieces)
    img.pieces = pieces.tolist()
    img = determine_colors(img)
    return img


def determine_colors(img):
    pcolors = []
    i = 0
    for p in img.pieces:
        avg = 0
        w = 0
        x0, y0 = int(p[0]), int(p[1])
        x1, y1 = int(p[2]), int(p[3])
        xc = round((x1+x0)/2)
        yc = round((y1+y0)/2)
        print(x0, y0, x1, y1)
        a = img.BGR[y0:y1, x0:x1]
        b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        for (x, y), pixel in np.ndenumerate(b):
            weight = 1/max(abs(x-xc), 1) + 1/max(abs(y-yc), 1)
            w += weight
            avg += pixel * weight
        avg = round(avg/w, 2)
        i += 1
        p.append(avg)
        pcolors.append(p)

    pcolors = np.array(pcolors, dtype='float32')
    print(pcolors)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compact, labels, centers = cv2.kmeans(pcolors[:, 6], 2, None,
                                          criteria, 10, flags)
    if centers[0] < centers[1]:
        blacklabel = 0
    else:
        blacklabel = 1

    for i, p in enumerate(pcolors):
        if labels[i] == blacklabel:
            p[5] += 6

    print(pcolors)
    img.pieces = pcolors.tolist()

    return img


if __name__ == "__main__":
    find_pieces()
