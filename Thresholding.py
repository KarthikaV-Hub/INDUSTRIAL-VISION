#Apply global, adaptive, and Otsu thresholding on metal surface images to separate defective vs. non-defective regions.
#Provide comparative results

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

image_path = "Downloads/Metal.webp"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 25, 5)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
_, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def threshold_metrics(binary_img, name):
    defect_pixels = np.sum(binary_img == 0)
    entropy = shannon_entropy(binary_img)
    print(f"{name}: DefectPixels={defect_pixels}, Entropy={entropy:.4f}")

print("===== Thresholding Comparison =====")
threshold_metrics(global_thresh, "Global Thresholding")
threshold_metrics(adaptive_thresh, "Adaptive Gaussian Thresholding")
threshold_metrics(otsu_thresh, "Otsu Thresholding")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Metal Surface")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(global_thresh, cmap="gray")
plt.title("Global Thresholding (T=127)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(adaptive_thresh, cmap="gray")
plt.title("Adaptive Gaussian Thresholding")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(otsu_thresh, cmap="gray")
plt.title(f"Otsu Thresholding (Auto T)")
plt.axis("off")

plt.tight_layout()
plt.show()
