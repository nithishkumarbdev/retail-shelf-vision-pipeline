"""
Generates synthetic retail-shelf images for testing/demo purposes.

No product dataset was attached to the, so this script builds
reproducible synthetic shelves: rows of "bottles/boxes" with repeating
per-row color+texture families (simulating brands), which is exactly the
structure the detector/grouper are designed for. This is clearly a
synthetic stand-in, not a real shelf photo -- documented here and in the
README so no one mistakes it for real evaluation data.

Usage: python3 sample_images/generate_sample.py
"""
import os
import random

import cv2
import numpy as np

random.seed(7)
np.random.seed(7)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_shelf_image(path, n_rows=4, min_items=5, max_items=8, item_w=70, item_h=140, gap=6, margin=30):
    row_brand_colors = [
        [(40, 160, 60), (35, 150, 55), (45, 170, 65)],       # green family (row 1 brand)
        [(200, 60, 40), (190, 55, 35), (210, 65, 45)],       # blue-ish family (BGR: high B)
        [(30, 90, 200), (25, 85, 190), (35, 95, 210)],       # orange/red family
        [(180, 40, 160), (170, 35, 150), (190, 45, 170)],    # purple family
    ]

    row_heights = [item_h] * n_rows
    shelf_width = margin * 2 + max_items * (item_w + gap)
    shelf_height = margin * 2 + sum(row_heights) + (n_rows - 1) * 40

    img = np.full((shelf_height, shelf_width, 3), (25, 25, 25), dtype=np.uint8)

    y = margin
    for row_idx in range(n_rows):
        n_items = random.randint(min_items, max_items)
        colors = row_brand_colors[row_idx % len(row_brand_colors)]
        x = margin
        for i in range(n_items):
            base_color = random.choice(colors)
            jitter = np.random.randint(-10, 10, size=3)
            color = tuple(int(max(0, min(255, c + j))) for c, j in zip(base_color, jitter))
            w = item_w + random.randint(-6, 6)
            h = row_heights[row_idx] + random.randint(-8, 8)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (10, 10, 10), 2)
            # a horizontal "label" stripe to add texture variation, like packaging print
            stripe_y = y + h // 3
            cv2.rectangle(img, (x + 4, stripe_y), (x + w - 4, stripe_y + 10),
                           (255, 255, 255), -1)
            x += w + gap
        y += row_heights[row_idx] + 40

    cv2.imwrite(path, img)
    return path


def make_empty_shelf(path, w=500, h=300):
    img = np.full((h, w, 3), (60, 60, 60), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_single_product(path, w=300, h=300):
    img = np.full((h, w, 3), (20, 20, 20), dtype=np.uint8)
    cv2.rectangle(img, (90, 60), (210, 260), (50, 140, 220), -1)
    cv2.rectangle(img, (90, 60), (210, 260), (10, 10, 10), 2)
    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    p1 = make_shelf_image(os.path.join(OUT_DIR, "sample_shelf_1.jpg"))
    p2 = make_empty_shelf(os.path.join(OUT_DIR, "sample_empty_shelf.jpg"))
    p3 = make_single_product(os.path.join(OUT_DIR, "sample_single_product.jpg"))
    print("Generated:", p1, p2, p3, sep="\n  ")
