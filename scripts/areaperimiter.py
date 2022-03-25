import numpy as np
from PIL import Image


def measure(seg_file):
    perimiter = 0
    img = np.pad(np.array(Image.open(seg_file)), 1)
    img[img != 0] = 1
    for x1, x2 in zip(range(img.shape[0] - 1), range(3, img.shape[0] - 1)):
        for y1, y2 in zip(range(img.shape[1] - 1), range(3, img.shape[1] - 1)):
            view = img[x1:x2, y1:y2]
            if view[1, 1] and np.sum(view) < 9:
                perimiter += 1
    area = np.sum(img)
    return (area, perimiter)
    