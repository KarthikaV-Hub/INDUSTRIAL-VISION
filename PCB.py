#Apply connected component labeling to count defective vs. good solder joints on PCB images. Provide statistics of the results


import cv2
import numpy as np

img = cv2.imread('pcb.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
num, labels, stats, _ = cv2.connectedComponentsWithStats(th)

good = 0
defect = 0
min_area = 50
max_area = 500

for i in range(1, num):
    x, y, w, h, area = stats[i]
    if area < min_area or area > max_area:
        defect += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    else:
        good += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

cv2.putText(img, f'Good: {good}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
cv2.putText(img, f'Defective: {defect}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

cv2.imshow('Solder Inspection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
