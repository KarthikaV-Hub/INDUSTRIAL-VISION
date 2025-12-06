import cv2
import numpy as np
import matplotlib.pyplot as plt

def thermal_anomaly_mapper(path):

    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print("Unable to load image")
        return

    smooth = cv2.GaussianBlur(frame, (5,5), 0)

    high_cut = np.percentile(smooth, 95)
    _, anomaly_mask = cv2.threshold(smooth, int(high_cut), 255, cv2.THRESH_BINARY)

    k = np.ones((3,3), np.uint8)
    refined = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, k, 1)

    thermal_map = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

    zones, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = thermal_map.copy()
    tag = 1

    print("Thermal Anomaly Report:")
    for z in zones:
        if cv2.contourArea(z) < 20:
            continue

        x, y, w, h = cv2.boundingRect(z)
        region = frame[y:y+h, x:x+w]
        avg_val = np.mean(region)

        cv2.rectangle(overlay, (x,y), (x+w,y+h), (255,255,255), 2)
        cv2.putText(overlay, f"T{tag}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        print(f"T{tag}: Area={cv2.contourArea(z):.2f}, Average Intensity={avg_val:.2f}")

        tag += 1

    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    plt.imshow(frame, cmap="gray")
    plt.title("Input Thermal Frame")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(refined, cmap="gray")
    plt.title("Anomaly Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Annotated Thermal Hotspots")
    plt.axis("off")

    plt.show()

thermal_anomaly_mapper("VISU.jpg")
