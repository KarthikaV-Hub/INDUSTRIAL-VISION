#Apply spatial filtering 
#enhance the surface of an automotive part image and highlight scratches or dents.

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Downloads/car.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

median = cv2.medianBlur(image, 5)
high_boost = cv2.addWeighted(image, 1.5, median, -0.5, 0)

laplacian = cv2.Laplacian(high_boost, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

sobelx = cv2.Sobel(high_boost, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(high_boost, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)
sobel_combined = cv2.convertScaleAbs(sobel_combined)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
tophat = cv2.morphologyEx(high_boost, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(high_boost, cv2.MORPH_BLACKHAT, kernel)

enhanced = cv2.add(tophat, blackhat)
_, scratches_mask = cv2.threshold(enhanced, 40, 255, cv2.THRESH_BINARY)

overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
contours, _ = cv2.findContours(scratches_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) > 30:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)

plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Automotive Part")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(high_boost, cmap='gray')
plt.title("High-Boost Filtered (Surface Enhanced)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian (Scratches/Dents)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(sobel_combined, cmap='gray')
plt.title("Sobel (Scratches/Dents)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(scratches_mask, cmap='gray')
plt.title("Morphological Scratch Mask")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(overlay[..., ::-1])
plt.title("Final Highlighted Scratches/Dents")
plt.axis("off")

plt.tight_layout()
plt.show()
