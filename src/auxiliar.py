import cv2
import logging as log

i = 1


def save(name, image):
    global i
    title = f"z{i:04d}_{name}.png"
    print(f"saving {title}...")
    cv2.imwrite(title, image)
    i += 1


def debugging():
    return log.root.level < 20
