#Apply chromatic adaptation transform (Bradford or von Kries) on industrial images to
#demonstrate constancy.

import numpy as np
import cv2
import matplotlib.pyplot as plt
D65 = np.array([0.95047, 1.00000, 1.08883])
A = np.array([1.09850, 1.00000, 0.35585])

def sRGB_to_XYZ(img):
    mask = img > 0.04045
    img_linear = np.where(mask, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    return img_linear @ M.T

def XYZ_to_sRGB(img):
    M = np.array([[ 3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660,  1.8760108,  0.0415560],
                  [ 0.0556434, -0.2040259,  1.0572252]])
    img_rgb = img @ M.T
    mask = img_rgb > 0.0031308
    img_out = np.where(mask, 1.055 * (img_rgb ** (1/2.4)) - 0.055, 12.92 * img_rgb)
    return img_out

def bradford_matrix(src_white, tgt_white):
    M = np.array([[ 0.8951,  0.2664, -0.1614],
                  [-0.7502,  1.7135,  0.0367],
                  [ 0.0389, -0.0685,  1.0296]])
    M_inv = np.linalg.inv(M)
    src_cone = M @ src_white
    tgt_cone = M @ tgt_white
    D = np.diag(tgt_cone / src_cone)
    return M_inv @ D @ M

img = cv2.imread("imgg.avif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img_xyz = sRGB_to_XYZ(img)
M_bradford = bradford_matrix(D65, A)
img_xyz_bradford = img_xyz @ M_bradford.T
img_bradford = np.clip(XYZ_to_sRGB(img_xyz_bradford), 0, 1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(img_bradford)
plt.title("Adaptation")
plt.axis("off")
plt.tight_layout()
plt.show()
