#Capture images at different focal lengths to analyze FOV changes.

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("roa.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def simulate_fov(image, zoom_factor):
    h, w = image.shape[:2]
    crop_h, crop_w = int(h/zoom_factor), int(w/zoom_factor)
    y1, x1 = (h-crop_h)//2, (w-crop_w)//2
    cropped = image[y1:y1+crop_h, x1:x1+crop_w]
    return cv2.resize(cropped, (w, h))
fov1 = simulate_fov(img, 1.2)  # slight zoom
fov2 = simulate_fov(img, 2.0)  # higher zoom
plt.title("6. Capture image at different focal length")
plt.subplot(1,3,1); plt.imshow(img); plt.title("Original")
plt.subplot(1,3,2); plt.imshow(fov1); plt.title("Zoom 1.2x")
plt.subplot(1,3,3); plt.imshow(fov2); plt.title("Zoom 2x")
plt.show()
