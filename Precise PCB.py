import cv2
import numpy as np
import matplotlib.pyplot as plt

gold_path = "PCB.png"
test_path = "form.png"

gold_img = cv2.imread(gold_path, 0)
test_img = cv2.imread(test_path, 0)

kp_extractor = cv2.ORB_create(5000)
kp_gold, des_gold = kp_extractor.detectAndCompute(gold_img, None)
kp_test, des_test = kp_extractor.detectAndCompute(test_img, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
pairings = matcher.match(des_gold, des_test)
pairings = sorted(pairings, key=lambda p: p.distance)

src_points = np.float32([kp_gold[p.queryIdx].pt for p in pairings]).reshape(-1,1,2)
dst_points = np.float32([kp_test[p.trainIdx].pt for p in pairings]).reshape(-1,1,2)

H_matrix, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)

aligned_img = cv2.warpPerspective(test_img, H_matrix, (gold_img.shape[1], gold_img.shape[0]))

diff_map = cv2.absdiff(gold_img, aligned_img)
smooth = cv2.GaussianBlur(diff_map, (7,7), 0)
_, mask = cv2.threshold(smooth, 25, 255, cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)

highlight = cv2.cvtColor(gold_img, cv2.COLOR_GRAY2BGR)
highlight[clean_mask > 0] = (0, 0, 255)

plt.figure(figsize=(14,4))
plt.subplot(1,4,1); plt.imshow(gold_img, cmap='gray'); plt.title("Golden Template"); plt.axis("off")
plt.subplot(1,4,2); plt.imshow(test_img, cmap='gray'); plt.title("Test PCB"); plt.axis("off")
plt.subplot(1,4,3); plt.imshow(aligned_img, cmap='gray'); plt.title("Aligned PCB"); plt.axis("off")
plt.subplot(1,4,4); plt.imshow(highlight); plt.title("Localized Defects"); plt.axis("off")
plt.show()
