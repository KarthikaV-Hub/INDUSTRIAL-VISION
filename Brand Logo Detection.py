#Implement SIFT or ORB feature matching to detect brand logos/serial numbers in product images.
#Compare with template matching

import cv2
import numpy as np
import matplotlib.pyplot as plt

product_path = "Downloads/biscuit.webp"
logo_path = "Downloads/logo.jpeg"

product_img = cv2.imread(product_path, cv2.IMREAD_GRAYSCALE)
template_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(template_img, None)
kp2, des2 = sift.detectAndCompute(product_img, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

sift_matched_img = cv2.drawMatches(template_img, kp1, product_img, kp2, good_matches[:30], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

if len(good_matches) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = template_img.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    sift_localized = cv2.cvtColor(product_img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(sift_localized, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
else:
    sift_localized = product_img

res = cv2.matchTemplate(product_img, template_img, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
h, w = template_img.shape
template_match_img = cv2.cvtColor(product_img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(template_match_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.imshow(template_img, cmap="gray")
plt.title("Template (Logo/Serial)")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(product_img, cmap="gray")
plt.title("Product Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(sift_matched_img)
plt.title(f"SIFT Feature Matching (Good={len(good_matches)})")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(template_match_img[..., ::-1])
plt.title(f"Template Matching (Score={max_val:.2f})")
plt.axis("off")

plt.tight_layout()
plt.show()
