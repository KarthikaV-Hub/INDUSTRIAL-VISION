#Analysis of images with different color models.

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("img.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
titles = ["RGB", "HSV", "LAB"]
images = [img_rgb, img_hsv, img_lab]
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(f"2. Color models: {titles[i]}")
    plt.axis("off")
plt.show()
