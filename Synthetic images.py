#Create two synthetic images with different RGB values but perceptually similar when
#converted to LAB.

import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = np.full((100, 100, 3), [200, 50, 50], dtype=np.uint8)
img2 = np.full((100, 100, 3), [210, 60, 60], dtype=np.uint8)
lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
print("LAB values of Image1:", lab1[0,0])
print("LAB values of Image2:", lab2[0,0])
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img1)
plt.title("RGB1")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img2)
plt.title("RGB2")
plt.axis('off')
plt.show()
