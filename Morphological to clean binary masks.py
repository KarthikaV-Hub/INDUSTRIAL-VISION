#Apply morphological opening and closing to clean binary masks

import cv2
import numpy as np

def clean(mask, ksize=5, iterations=1):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    o = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=iterations)
    c = cv2.morphologyEx(o, cv2.MORPH_CLOSE, k, iterations=iterations)
    return c

img = cv2.imread('mask.png', 0)
_, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
res = clean(mask, ksize=7, iterations=2)
cv2.imwrite('mask_clean.png', res)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
