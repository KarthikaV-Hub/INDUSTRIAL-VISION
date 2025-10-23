#Demonstrate given in IITM Virtual Labs for Image Processing.

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("img.avif")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_eq = cv2.equalizeHist(gray)
blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
edges = cv2.Canny(gray, 100, 200)
titles = ["Original", "Gray", "Histogram Equalized", "Gaussian Blur", "Edges"]
images = [img_rgb, gray, hist_eq, blurred, edges]
plt.figure(figsize=(12, 6))
for i, image in enumerate(images):
    plt.subplot(2, 3, i+1)
    cmap_type = "gray" if len(image.shape) == 2 else None
    plt.imshow(image, cmap=cmap_type)
    plt.title(titles[i])
    plt.axis("off")
plt.show()
