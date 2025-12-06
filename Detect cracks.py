import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("karthi2.png")
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b = cv2.GaussianBlur(g, (5,5), 0)
e = cv2.Canny(b, 50, 150)

k = np.ones((3,3), np.uint8)
m = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=2)
d = cv2.dilate(m, k, iterations=2)

ct, _ = cv2.findContours(d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
out = img.copy()

for c in ct:
    a = cv2.contourArea(c)
    if a > 100:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(out, "Crack", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

plt.figure(figsize=(14,7))
plt.subplot(1,3,1)
plt.imshow(g, cmap="gray")
plt.title("Grayscale")

plt.subplot(1,3,2)
plt.imshow(d, cmap="gray")
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.title("Crack Detection")
plt.show()
