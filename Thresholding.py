#Thresholding

import cv2
import matplotlib.pyplot as plt
from google.colab import files

files.upload()
a = cv2.imread('Thres_sample.jpg', 0)
_, t1 = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)
_, t2 = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(a, cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title("Binary Threshold")
plt.imshow(t1, cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title("Otsu Threshold")
plt.imshow(t2, cmap='gray')
plt.axis('off')
plt.show()
