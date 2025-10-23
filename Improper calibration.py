#Analyze two real-world examples where improper calibration or quantization causes
#visible artifacts.

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def quantize_image(img, levels=16):
    return np.floor(img / (256 / levels)) * (256 / levels)

quantized_img = quantize_image(img_rgb, levels=8).astype(np.uint8)
calibration_img = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=50)

titles = ["Original", "Quantization Artifact", "Calibration Artifact"]
images = [img_rgb, quantized_img, calibration_img]

plt.figure(figsize=(12, 6))
for i, image in enumerate(images):
    plt.subplot(1, 3, i+1)
    plt.imshow(image)
    plt.title(titles[i])
    plt.axis("off")
plt.show()
