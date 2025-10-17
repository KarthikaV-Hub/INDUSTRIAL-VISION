#Implement Sobel, Canny, and Laplacian edge detectors on bridge/concrete images to detect cracks.
#Compare their effectiveness

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

image_path = "Downloads/bridge.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

median_val = np.median(img)
low = int(max(0, 0.66 * median_val))
high = int(min(255, 1.33 * median_val))
canny = cv2.Canny(img, low, high)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

def edge_metrics(edge_img, name):
    edge_pixels = np.sum(edge_img > 0)
    similarity, _ = ssim(img, edge_img, full=True)
    print(f"{name}: EdgePixels={edge_pixels}, SSIM={similarity:.4f}")

print("===== Edge Detection Comparison =====")
edge_metrics(sobel, "Sobel")
edge_metrics(canny, "Canny")
edge_metrics(laplacian, "Laplacian")

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original (Bridge Surface)")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel, cmap="gray")
plt.title("Sobel Gradient Magnitude (Cracks)")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge Detection (Cracks)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Edge Detection (Cracks)")
plt.axis("off")

plt.tight_layout()
plt.show()
