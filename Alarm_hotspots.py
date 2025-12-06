import numpy as np
from collections import deque

def generate_sequence(n=60, peak_point=30):
    data = []
    for idx in range(n):
        frame = np.random.normal(loc=40, scale=2, size=(64,64))
        if abs(idx - peak_point) < 3:
            a, b = np.random.randint(10,54), np.random.randint(10,54)
            frame[a-3:a+3, b-3:b+3] += 30
        data.append(np.clip(frame, 0, 255).astype(np.uint8))
    return data

frames = generate_sequence()
limit = 70.0
history = deque(maxlen=3)
flags = []

for t, img in enumerate(frames):
    m = img.max()
    history.append(m > limit)
    if sum(history) >= 2:
        flags.append((t, m))

print("Alarm events:", flags[:5])
