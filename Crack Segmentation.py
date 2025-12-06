import cv2
import numpy as np
import matplotlib.pyplot as plt

def crack_width_and_severity(p):
    img = cv2.imread(p)
    if img is None:
        print("Image not found")
        return

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = cv2.GaussianBlur(g, (5,5), 0)
    _, t = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = np.ones((3,3), np.uint8)
    d = cv2.dilate(t, k, iterations=1)

    dist = cv2.distanceTransform(d, cv2.DIST_L2, 5)
    w = dist[d == 255] * 2

    if len(w) == 0:
        print("No crack found")
        return

    avg = np.mean(w)

    if avg < 2:
        s = "NARROW"
    elif avg < 5:
        s = "MEDIUM"
    else:
        s = "WIDE"

    print(f"Avg Crack Width: {avg:.2f}px → {s}")

    out = img.copy()
    out[d == 255] = [0, 0, 255]

    plt.figure(figsize=(14,7))

    plt.subplot(1,3,1)
    plt.imshow(g, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(d, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title("Crack Overlay")
    plt.axis("off")

    plt.show()

crack_width_and_severity("weld.jpg")
