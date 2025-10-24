#Develop a color-based thresholding method (HSV space) to classify fruits as ripe or unripe. Test on at least 5 samples.


import cv2
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

files.upload()
imgs = []
for i in range(5):
    imgs.append(cv2.imread(f'fruit{i+1}.jpg'))

for img in imgs:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    r1 = np.array([0,120,70])
    r2 = np.array([10,255,255])
    r3 = np.array([170,120,70])
    r4 = np.array([180,255,255])
    red = cv2.inRange(hsv, r1, r2) + cv2.inRange(hsv, r3, r4)
    
    g1 = np.array([35,50,50])
    g2 = np.array([85,255,255])
    green = cv2.inRange(hsv, g1, g2)
    
    red_count = cv2.countNonZero(red)
    green_count = cv2.countNonZero(green)
    
    status = "Ripe" if red_count > green_count else "Unripe"
    
    res = cv2.bitwise_and(img, img, mask=red)
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title("Red Mask")
    plt.imshow(red, cmap='gray')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title(f"Detected: {status}")
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
