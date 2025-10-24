#Design of Classical CV Pipelines for Defect Localization and Pattern Matching


import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

files.upload()
x = cv2.imread('sample.jpg')
g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
b = cv2.GaussianBlur(g, (5,5), 0)
e = cv2.Canny(b, 50, 150)
c, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
o = x.copy()
cv2.drawContours(o, c, -1, (0,255,0), 2)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Defect Localization")
plt.imshow(cv2.cvtColor(o, cv2.COLOR_BGR2RGB))
plt.axis('off')

t = cv2.imread('template.jpg', 0)
r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
_, v, _, p = cv2.minMaxLoc(r)
h, w = t.shape
m = x.copy()
cv2.rectangle(m, p, (p[0]+w, p[1]+h), (255,0,0), 2)
plt.subplot(1,2,2)
plt.title("Pattern Matching")
plt.imshow(cv2.cvtColor(m, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
