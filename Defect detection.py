#Perform defect detection on PCB images using edge detection and morphology to identify broken tracks or missing solder points. Provide annotated outputs.


import cv2
import numpy as np

img = cv2.imread('pcb.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)
edges = cv2.Canny(blur,50,150)
kernel = np.ones((3,3),np.uint8)
morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for c in contours:
    if 100<cv2.contourArea(c)<5000:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(out,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('defects',out)
cv2.waitKey(0)
cv2.destroyAllWindows()
