import cv2
import numpy as np
import matplotlib.pyplot as plt

def weld_seam_inspector(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failure")
        return

    den = cv2.bilateralFilter(img, 11, 75, 75)
    grad = cv2.Canny(den, 40, 120)

    h, w = grad.shape
    vertical_sum = np.sum(grad, axis=0)
    seam_col = np.argmax(vertical_sum)

    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(line_img, (seam_col, 0), (seam_col, h), (0,255,0), 2)

    scan = grad[:, seam_col-2:seam_col+3]
    seam_profile = np.sum(scan, axis=1)

    thresh = np.mean(seam_profile) * 0.4
    breaks = np.where(seam_profile < thresh)[0]

    alerts = line_img.copy()
    for b in breaks:
        cv2.circle(alerts, (seam_col, b), 5, (0,0,255), -1)

    print("Discontinuities:", len(breaks))

    plt.figure(figsize=(14,6))

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Weld Surface")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(line_img[:,:,::-1])
    plt.title("Detected Weld Seam")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(alerts[:,:,::-1])
    plt.title("Flagged Discontinuities")
    plt.axis("off")

    plt.show()

weld_seam_inspector("weld.jpg")
