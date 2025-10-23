#Estimate pixel resolution using known object dimensions to compute DOF.

import cv2
import numpy as np
def pixel_resolution(img_path, obj_mm, pts):
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError("Image not found")
    (x1, y1), (x2, y2) = pts
    return obj_mm / np.hypot(x2 - x1, y2 - y1)
def depth_of_field(f_mm, N, coc_mm, s_mm):
    H = f_mm**2 / (N * coc_mm)
    near = (H * s_mm) / (H + (s_mm - f_mm))
    far = (H * s_mm) / (H - (s_mm - f_mm)) if s_mm < H else float('inf')
    total = far - near if far != float('inf') else float('inf')
    return near, far, total
image_path = "roa.jpg"
known_length_mm = 10
ref_points = ((200, 300), (300, 300))
res = pixel_resolution(image_path, known_length_mm, ref_points)
print(f"Pixel resolution: {res:.4f} mm/pixel")
focal_length_mm, f_number, coc_mm, subj_dist_mm = 50, 8, 0.03, 1000
near, far, total = depth_of_field(focal_length_mm, f_number, coc_mm, subj_dist_mm)
print(f"DOF Near: {near:.2f} mm, DOF Far: {far:.2f} mm, Total DOF: {total:.2f} mm")
