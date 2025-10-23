#Plot the chromaticity coordinates of the pixels of an image on the CIE 1931 diagram.

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

pixels = img.reshape(-1, 3)

mask = pixels <= 0.04045
pixels[mask] = pixels[mask] / 12.92
pixels[~mask] = ((pixels[~mask] + 0.055) / 1.055) ** 2.4

M = np.array([[0.4124564, 0.3575761, 0.1804375],
              [0.2126729, 0.7151522, 0.0721750],
              [0.0193339, 0.1191920, 0.9503041]])

xyz = np.dot(pixels, M.T)

X = xyz[:,0]
Y = xyz[:,1]
Z = xyz[:,2]
xy = np.stack([X / (X + Y + Z), Y / (X + Y + Z)], axis=1)

if xy.shape[0] > 5000:
    idx = np.random.choice(xy.shape[0], 5000, replace=False)
    xy = xy[idx]
    colors = img.reshape(-1,3)[idx]
else:
    colors = img.reshape(-1,3)

plt.figure(figsize=(6,6))
plt.scatter(xy[:,0], xy[:,1], s=1, alpha=0.5, color=colors)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Image Pixels on CIE 1931 Chromaticity Diagram")
plt.xlim(0,0.8)
plt.ylim(0,0.9)
plt.show()
