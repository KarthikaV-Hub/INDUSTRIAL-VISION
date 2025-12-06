import time
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
!pip install ultralytics

det_engine = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def run_pipeline(img_obj, net_cls, net_det):
    t_start = time.time()
    prep = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])(img_obj).unsqueeze(0).to(device)

    with torch.no_grad():
        out_cls = net_cls(prep)
        tag = int(out_cls.argmax(dim=1).item())
    t_mid = time.time()

    det_out = None
    det_dur = None

    if tag == 1:
        arr = np.array(img_obj)
        d0 = time.time()
        raw = net_det(arr)
        d1 = time.time()
        det_dur = d1 - d0
        det_out = raw.pandas().xyxy[0].to_dict(orient='records')

    t_end = time.time()

    summary = {
        "classification": {"label": tag},
        "detection": {"time_s": det_dur, "detections": det_out},
        "timing": {
            "classification_s": round(t_mid - t_start, 4),
            "detection_s": det_dur,
            "total_s": round(t_end - t_start, 4)
        }
    }
    return summary

img_input = Image.open(base_path).convert("RGB")
result = run_pipeline(img_input, model, det_engine)
print(json.dumps(result, indent=2))
