#Use morphological operations (opening/closing) to isolate defective regionsin a fabric image

import cv2
import numpy as np
import matplotlib.pyplot as plt

fabric_path = 'fabric_sample.jpg'
fabric_img = cv2.imread(fabric_path)

smooth_img = cv2.GaussianBlur(fabric_img, (5, 5), 0)
gray_img = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2GRAY)
_, defect_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
open_img = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=2)
close_img = cv2.morphologyEx(open_img, cv2.MORPH_CLOSE, kernel, iterations=2)

titles = [
    "Original Fabric",
    "Smoothed Image",
    "Grayscale",
    "Binary Threshold",
    "After Opening",
    "After Closing (Defects Isolated)"
]

results = [fabric_img, smooth_img, gray_img, defect_mask, open_img, close_img]

plt.figure(figsize=(14, 9))
for idx, (title, result) in enumerate(zip(titles, results)):
    plt.subplot(2, 3, idx + 1)
    if len(result.shape) == 2:
        plt.imshow(result, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

