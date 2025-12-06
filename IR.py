import cv2
import numpy as np
import matplotlib.pyplot as plt

def abnormal_heat_zone_analysis(path):
    thermal = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if thermal is None:
        print("Image not loaded")
        return

    smoothed = cv2.GaussianBlur(thermal, (5,5), 0)

    limit = np.percentile(smoothed, 95)
    _, hot_mask = cv2.threshold(smoothed, int(limit), 255, cv2.THRESH_BINARY)

    kern = np.ones((3,3), np.uint8)
    hot_refined = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kern, 1)

    heat_view = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)

    zones, _ = cv2.findContours(hot_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = heat_view.copy()

    for z in zones:
        if cv2.contourArea(z) < 20:
            continue
        a, b, c, d = cv2.boundingRect(z)
        cv2.rectangle(overlay, (a, b), (a+c, b+d), (0,255,255), 2)

    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    plt.imshow(thermal, cmap='gray')
    plt.title("IR Thermal Frame")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(hot_refined, cmap='gray')
    plt.title("Isolated Hot Regions")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Abnormal Heating Detection")
    plt.axis("off")

    plt.show()


abnormal_heat_zone_analysis("ir.jpg")
