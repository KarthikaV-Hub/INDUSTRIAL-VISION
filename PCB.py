import cv2
import numpy as np
import matplotlib.pyplot as plt

ref_img_path = "pcb.png"
test_img_path = "form.png"

img_ref = cv2.imread(ref_img_path, 0)
img_test = cv2.imread(test_img_path, 0)

scale_x = img_ref.shape[1] / img_test.shape[1]
scale_y = img_ref.shape[0] / img_test.shape[0]
img_test = cv2.resize(img_test, None, fx=scale_x, fy=scale_y)

img_diff = cv2.absdiff(img_ref, img_test)
img_blur = cv2.GaussianBlur(img_diff, (5,5), 0)
_, img_thresh = cv2.threshold(img_blur, 30, 255, cv2.THRESH_BINARY)

kern = np.ones((5,5), np.uint8)
mask_clean = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kern, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kern, iterations=1)

img_overlay = cv2.cvtColor(img_test, cv2.COLOR_GRAY2BGR)
img_overlay[mask_clean > 0] = (0,0,255)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img_ref, cmap='gray'); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(img_test, cmap='gray'); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(img_overlay); plt.axis('off')
plt.show()
