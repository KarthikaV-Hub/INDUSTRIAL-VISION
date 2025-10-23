#Obtain or simulate an industrial dataset (e.g., machine parts, PCB boards, mechanical
#tools) captured under: Different lighting conditions (bright, dim, directional, uneven illumination) and Different lens parameters (focal length, aperture, exposure time).

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

bright_img = cv2.convertScaleAbs(img_rgb, alpha=1.2, beta=50)
dim_img = cv2.convertScaleAbs(img_rgb, alpha=0.7, beta=-30)
directional_img = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=0)

uneven_img = img_rgb.copy()
uneven_img[:, :uneven_img.shape[1]//2] = cv2.convertScaleAbs(
    uneven_img[:, :uneven_img.shape[1]//2], alpha=0.5, beta=-50
)

blurred_img = cv2.GaussianBlur(img_rgb, (11, 11), 10)
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_img = cv2.filter2D(img_rgb, -1, sharpen_kernel)
overexposed_img = cv2.convertScaleAbs(img_rgb, alpha=2, beta=100)

titles = [
    "Original", "Bright", "Dim", "Directional Light", "Uneven Light",
    "Blurred (Defocus)", "Sharpened", "Overexposed"
]
images = [
    img_rgb, bright_img, dim_img, directional_img, uneven_img,
    blurred_img, sharpened_img, overexposed_img
]

plt.figure(figsize=(14, 10))
for i, image in enumerate(images):
    plt.subplot(2, 4, i+1)
    plt.imshow(image)
    plt.title(titles[i])
    plt.axis("off")
plt.show()
