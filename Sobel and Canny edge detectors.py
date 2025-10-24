#Implement Sobel and Canny edge detectors on sample industrial images

import cv2
import matplotlib.pyplot as plt
from google.colab import files

files.upload()
a = cv2.imread('industrial_sample.jpg', 0)

sx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sx), 0.5, cv2.convertScaleAbs(sy), 0.5, 0))

canny = cv2.Canny(a, 50, 150)

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(a, cmap='gray')
plt.axis('off')
plt.subplot(1,3,2)
plt.title("Sobel Edge")
plt.imshow(sobel, cmap='gray')
plt.axis('off')
plt.subplot(1,3,3)
plt.title("Canny Edge")
plt.imshow(canny, cmap='gray')
plt.axis('off')
plt.show()
