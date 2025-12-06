def surface_defect_classification(img_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    raw_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    smooth_img = cv2.GaussianBlur(gray_img, (5,5), 0)
    _, defect_mask = cv2.threshold(
        smooth_img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    defect_cnts, _ = cv2.findContours(
        defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = raw_img.copy()

    for c in defect_cnts:
        defect_area = cv2.contourArea(c)
        if defect_area < 20:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        if defect_area < 100:
            tag, clr = "Small Defect", (0,255,0)
        elif defect_area < 500:
            tag, clr = "Medium Defect", (0,255,255)
        else:
            tag, clr = "Large Defect", (0,0,255)

        cv2.rectangle(annotated, (bx,by), (bx+bw,by+bh), clr, 2)
        cv2.putText(annotated, tag, (bx,by-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1)

    plt.figure(figsize=(14,5))

    plt.subplot(1,3,1)
    plt.imshow(gray_img, cmap="gray")
    plt.title("Grayscale View")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(defect_mask, cmap="gray")
    plt.title("Extracted Defect Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(annotated[:,:,::-1])
    plt.title("Surface Defects Classified")
    plt.axis("off")

    plt.show()
