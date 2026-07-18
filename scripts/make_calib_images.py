#!/usr/bin/env python3
"""Generate synthetic calibration JPEGs for HHB INT8 (PPE / YOLOv8)."""
import os
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calib")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


def save(name, img):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(path, img.shape)


def main():
    for i, color in enumerate(
        [(40, 40, 40), (180, 180, 180), (30, 60, 120), (20, 90, 20), (90, 40, 20)]
    ):
        img = np.full((640, 640, 3), color, np.uint8)
        noise = rng.integers(-15, 16, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        save(f"bg_{i}.jpg", img)

    for i in range(8):
        img = np.full((480, 640, 3), (50 + i * 10, 55, 60), np.uint8)
        cv2.rectangle(img, (0, 300), (640, 480), (70, 70, 75), -1)
        for _ in range(5):
            x1, y1 = int(rng.integers(20, 500)), int(rng.integers(80, 350))
            x2 = x1 + int(rng.integers(40, 120))
            y2 = y1 + int(rng.integers(40, 100))
            col = (int(rng.integers(40, 200)),) * 3
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.integers(80, 560))
            cy = int(rng.integers(120, 280))
            cv2.rectangle(
                img,
                (cx - 18, cy),
                (cx + 18, cy + 110),
                (
                    int(rng.integers(20, 80)),
                    int(rng.integers(40, 100)),
                    int(rng.integers(120, 200)),
                ),
                -1,
            )
            cv2.circle(img, (cx, cy - 15), 16, (200, 180, 160), -1)
            helmet = [(0, 220, 255), (0, 140, 255), (240, 240, 240), (40, 40, 220)][
                int(rng.integers(0, 4))
            ]
            cv2.ellipse(img, (cx, cy - 22), (20, 14), 0, 0, 180, helmet, -1)
            cv2.rectangle(img, (cx - 18, cy + 20), (cx + 18, cy + 50), (0, 255, 255), -1)
        img = cv2.resize(img, (640, 640))
        if i % 2 == 0:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        save(f"ppe_scene_{i}.jpg", img)

    for i in range(6):
        img = np.zeros((640, 640, 3), np.uint8)
        img[:] = (
            int(rng.integers(20, 100)),
            int(rng.integers(20, 100)),
            int(rng.integers(20, 100)),
        )
        cx, cy = 320, 280
        helmet = [(0, 230, 255), (0, 165, 255), (250, 250, 250)][i % 3]
        cv2.ellipse(img, (cx, cy), (140, 100), 0, 0, 180, helmet, -1)
        cv2.circle(img, (cx, cy + 40), 70, (210, 190, 170), -1)
        cv2.rectangle(img, (cx - 100, cy + 100), (cx + 100, cy + 280), (0, 255, 200), -1)
        cv2.rectangle(img, (cx - 100, cy + 140), (cx + 100, cy + 160), (0, 0, 0), -1)
        if i % 2:
            M = np.float32(
                [[1, 0, int(rng.integers(-8, 9))], [0, 1, int(rng.integers(-8, 9))]]
            )
            img = cv2.warpAffine(img, M, (640, 640), borderMode=cv2.BORDER_REFLECT)
        save(f"closeup_{i}.jpg", img)

    for i, mean in enumerate([40, 90, 140, 190]):
        img = rng.normal(mean, 25, (640, 640, 3)).clip(0, 255).astype(np.uint8)
        for _ in range(8):
            x, y = int(rng.integers(0, 500)), int(rng.integers(0, 500))
            cv2.rectangle(
                img,
                (x, y),
                (x + int(rng.integers(30, 120)), y + int(rng.integers(30, 120))),
                (
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                    int(rng.integers(0, 255)),
                ),
                -1,
            )
        save(f"noise_obj_{i}.jpg", img)

    print(f"DONE: {len(os.listdir(OUT))} files in {OUT}")


if __name__ == "__main__":
    main()
