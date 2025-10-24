#Use connected components analysis on a tablet image to detect missing, broken, or extra objects. Display bounding boxes for identified defects.


import cv2
import numpy as np

img = cv2.imread('tablet.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
num, labels, stats, _ = cv2.connectedComponentsWithStats(th)

expected_count = 10
min_area = 200
max_area = 2000

detected = 0

for i in range(1, num):
    x, y, w, h, area = stats[i]
    if area < min_area:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(img, 'Extra', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    elif area > max_area:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(img, 'Broken', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    else:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        detected += 1

missing = expected_count - detected
cv2.putText(img, f'Missing: {missing}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

cv2.imshow('Tablet Defects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
