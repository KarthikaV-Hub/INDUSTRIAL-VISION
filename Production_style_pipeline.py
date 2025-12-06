import time, json
from PIL import Image
import numpy as np
import torch
from datetime import datetime

def prep_img(pil_in):
    arr = np.array(pil_in)
    arr = apply_preprocess(arr)
    return Image.fromarray(arr)

def run_pipe(pil_in, cls_net, det_net=None):
    t_start = time.time()
    img_p = prep_img(pil_in)

    tfm = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    tensor_x = tfm(img_p).unsqueeze(0).to(device)

    with torch.no_grad():
        out_cls = cls_net(tensor_x)
        pred_lbl = out_cls.argmax(1).item()

    t_mid = time.time()
    det_out = None

    if pred_lbl == 1 and det_net is not None:
        det_out = {"detections": []}

    t_end = time.time()

    rep = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "timing": {
            "pre": round(t_mid - t_start, 4),
            "inf": round(t_end - t_mid, 4),
            "total": round(t_end - t_start, 4)
        },
        "classification": {"label": int(pred_lbl)},
        "detection": det_out
    }
    return rep

img_in = Image.open(base_path).convert("RGB")
output_rep = run_pipe(img_in, model, det_net=None)
print(json.dumps(output_rep, indent=2))
