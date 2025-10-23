#Perform gray-level and color image quantization.

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("img.avif")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
num_levels = 6
quant_gray = (gray // (256 // num_levels)) * (256 // num_levels)
pixels = img_rgb.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
k = 6
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 15, cv2.KMEANS_RANDOM_CENTERS)
quant_color = centers[labels.flatten()].reshape(img_rgb.shape).astype(np.uint8)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original Gray")
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(quant_gray, cmap='gray')
plt.title(f"Gray Quantized ({num_levels} levels)")
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(quant_color)
plt.title(f"Color Quantized (k={k})")
plt.axis('off')
plt.suptitle("Gray-Level and Color Image Quantization", fontsize=16)
plt.show()
