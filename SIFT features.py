#Extract SIFT/ORB features for part identification or alignment tasks

import cv2
import matplotlib.pyplot as plt
from google.colab import files

files.upload()
a = cv2.imread('part1.jpg', 0)
b = cv2.imread('part2.jpg', 0)

s = cv2.SIFT_create()
k1, d1 = s.detectAndCompute(a, None)
k2, d2 = s.detectAndCompute(b, None)
m = cv2.BFMatcher()
r = m.knnMatch(d1, d2, k=2)
g = []
for x, y in r:
    if x.distance < 0.75 * y.distance:
        g.append([x])
img_s = cv2.drawMatchesKnn(a, k1, b, k2, g[:30], None, flags=2)
plt.figure(figsize=(15,6))
plt.title("SIFT")
plt.imshow(img_s)
plt.axis('off')
plt.show()

o = cv2.ORB_create()
k3, d3 = o.detectAndCompute(a, None)
k4, d4 = o.detectAndCompute(b, None)
m2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
r2 = m2.match(d3, d4)
r2 = sorted(r2, key=lambda x: x.distance)
img_o = cv2.drawMatches(a, k3, b, k4, r2[:30], None, flags=2)
plt.figure(figsize=(15,6))
plt.title("ORB")
plt.imshow(img_o)
plt.axis('off')
plt.show()
