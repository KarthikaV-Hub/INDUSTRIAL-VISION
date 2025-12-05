import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("defect.jpg", 0)

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret_otsu, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(img, cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Otsu Threshold")
plt.imshow(otsu, cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Adaptive Threshold")
plt.imshow(adaptive, cmap="gray")
plt.show()

sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img, None)
sift_img = cv2.drawKeypoints(img, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(img, None)
orb_img = cv2.drawKeypoints(img, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("SIFT Features")
plt.imshow(sift_img, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("ORB Features")
plt.imshow(orb_img, cmap="gray")
plt.show()
