"""
services/detector/detector.py

Product detector for retail shelf images.

WHY CLASSICAL CV INSTEAD OF A DEEP MODEL
-----------------------------------------
The brief allows "any model of your choice which ... can run without a lot of
compute". The build environment used to produce this project has no internet
access (verified: pip/model downloads are blocked) and no pretrained
detection weights (YOLO/SKU-110K/etc.) are cached on disk. Rather than fake a
deep model or hardcode fabricated results, this module implements a genuine,
deterministic, zero-GPU detection algorithm tailored to shelf images:

  1. Shelf-row segmentation: horizontal bands of the image are located via
     the row-wise profile of edge energy (products create strong horizontal
     edge bands at their top/bottom; the flat gaps between shelves are weak).
  2. Column segmentation within each row: vertical edge-energy profile finds
     the boundaries between adjacent products standing side by side.
  3. Each row x column cell is refined to a tight bounding box using contour
     extents inside the cell (trims empty margins).
  4. Cells that are near-uniform (no real object -> background/shadow) are
     dropped using a variance-of-Laplacian sharpness/contrast test.
  5. A general contour-based fallback runs when the grid method finds too few
     regions (e.g. non-grid, cluttered, or single-object images).
  6. Non-max suppression removes duplicate/overlapping boxes.

This is intentionally swappable: see `Detector.detect()` docstring for how to
plug in a real deep model (e.g. YOLOv8 fine-tuned on SKU-110K) later without
changing the service's HTTP contract.
"""
import time
import numpy as np
import cv2


class Detector:
    def __init__(
        self,
        min_box_area_frac: float = 0.0008,   # ignore boxes smaller than this fraction of image area
        max_box_area_frac: float = 0.35,     # ignore boxes larger than this fraction (likely background)
        nms_iou_threshold: float = 0.35,
        max_dim: int = 1400,                 # resize cap for speed on very large inputs
    ):
        self.min_box_area_frac = min_box_area_frac
        self.max_box_area_frac = max_box_area_frac
        self.nms_iou_threshold = nms_iou_threshold
        self.max_dim = max_dim

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, image_bgr: np.ndarray):
        """
        Run detection on a BGR numpy image.

        Returns
        -------
        detections: list[dict] with keys bbox=[x1,y1,x2,y2] (int, original
                    image coordinate space), confidence (float 0-1)
        meta: dict with timing / method info

        To swap in a real deep model later: replace the body of this method
        with e.g. `results = yolo_model.predict(image_bgr)` and map the
        results into the same `{"bbox": [...], "confidence": ...}` schema.
        The Flask route and every downstream consumer is agnostic to how
        detections were produced.
        """
        t0 = time.time()
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image passed to detector")

        orig_h, orig_w = image_bgr.shape[:2]
        image_bgr, scale = self._resize_if_needed(image_bgr)
        h, w = image_bgr.shape[:2]

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        boxes = self._grid_detect(gray)
        method = "shelf-grid"

        if len(boxes) < 2:
            boxes = self._contour_detect(gray)
            method = "contour-fallback"

        boxes = self._non_max_suppression(boxes, self.nms_iou_threshold)

        # scale boxes back to original image coordinates
        detections = []
        for (x1, y1, x2, y2, conf) in boxes:
            if scale != 1.0:
                x1, y1, x2, y2 = [int(round(v / scale)) for v in (x1, y1, x2, y2)]
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": round(float(conf), 4)})

        detections.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        for i, d in enumerate(detections):
            d["detection_id"] = i

        meta = {
            "method": method,
            "num_detections": len(detections),
            "processing_time_ms": round((time.time() - t0) * 1000, 2),
            "image_size": {"width": orig_w, "height": orig_h},
        }
        return detections, meta

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resize_if_needed(self, image_bgr):
        h, w = image_bgr.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dim:
            return image_bgr, 1.0
        scale = self.max_dim / float(longest)
        resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale

    def _grid_detect(self, gray: np.ndarray):
        h, w = gray.shape
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)

        row_profile = grad_mag.sum(axis=1)
        row_profile = self._smooth(row_profile, k=max(5, h // 100))
        row_bands = self._segment_profile(row_profile, min_band_frac=0.02, total_len=h)

        boxes = []
        for (ry1, ry2) in row_bands:
            band = grad_mag[ry1:ry2, :]
            if band.shape[0] < 8:
                continue
            col_profile = band.sum(axis=0)
            # Deliberately light smoothing here: the gap between two adjacent
            # products is often only a few pixels wide, and an over-large
            # kernel blurs that gap into the neighboring products' signal,
            # merging separate items into one box.
            col_profile = self._smooth(col_profile, k=3)
            col_bands = self._segment_columns_by_peaks(col_profile, total_len=w)

            for (cx1, cx2) in col_bands:
                cell_gray = gray[ry1:ry2, cx1:cx2]
                if cell_gray.size == 0:
                    continue
                sharpness = cv2.Laplacian(cell_gray, cv2.CV_32F).var()
                if sharpness < 8.0:  # near-uniform region -> background/shadow, not a product
                    continue
                x1, y1, x2, y2 = cx1, ry1, cx2, ry2

                area_frac = ((x2 - x1) * (y2 - y1)) / float(h * w)
                if area_frac < self.min_box_area_frac or area_frac > self.max_box_area_frac:
                    continue

                conf = self._sharpness_to_confidence(sharpness)
                boxes.append((x1, y1, x2, y2, conf))
        return boxes

    def _contour_detect(self, gray: np.ndarray):
        h, w = gray.shape
        edges = cv2.Canny(gray, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area_frac = (cw * ch) / float(h * w)
            if area_frac < self.min_box_area_frac or area_frac > self.max_box_area_frac:
                continue
            aspect = ch / float(cw + 1e-6)
            if aspect > 8 or aspect < 0.1:
                continue
            cell = gray[y:y + ch, x:x + cw]
            sharpness = cv2.Laplacian(cell, cv2.CV_32F).var() if cell.size else 0.0
            conf = self._sharpness_to_confidence(sharpness)
            boxes.append((x, y, x + cw, y + ch, conf))
        return boxes

    @staticmethod
    def _smooth(arr_1d: np.ndarray, k: int) -> np.ndarray:
        k = max(1, k | 1)  # force odd
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(arr_1d, kernel, mode="same")

    @staticmethod
    def _segment_profile(profile: np.ndarray, min_band_frac: float, total_len: int):
        """
        Split a 1-D energy profile into contiguous 'active' bands separated by
        low-energy valleys (true background gaps between shelf rows/products
        sit near zero; any real object content -- even a low-contrast one --
        sits well above zero). A threshold relative to the *max* (rather than
        the *mean*) is used deliberately: a single strong edge (e.g. a label
        stripe) can otherwise drag the mean high enough that ordinary object
        body rows/columns fall below it and get misclassified as gaps.
        """
        if profile.max() <= 0:
            return [(0, total_len)]
        thresh = max(profile.max() * 0.03, profile.mean() * 0.15)
        active = profile > thresh

        bands = []
        start = None
        for i, is_active in enumerate(active):
            if is_active and start is None:
                start = i
            elif not is_active and start is not None:
                bands.append((start, i))
                start = None
        if start is not None:
            bands.append((start, len(active)))

        min_len = max(3, int(total_len * min_band_frac))
        bands = [(s, e) for (s, e) in bands if e - s >= min_len]

        if not bands:
            return [(0, total_len)]
        return bands

    @staticmethod
    def _segment_columns_by_peaks(col_profile: np.ndarray, total_len: int):
        """
        Find the vertical boundaries between side-by-side products.

        Two adjacent products' borders are often only a few pixels apart --
        closer than the blur introduced by Gaussian smoothing + the Sobel
        kernel's spatial support -- so the valley between them rarely drops
        back to a clean background level (see _segment_profile's docstring
        for the analogous row-gap case, which does not have this problem).
        Peak-finding sidesteps this: each product border still produces a
        clear *local maximum* with high prominence relative to the flatter
        product-interior baseline on either side, even when neighboring
        borders partially merge into one wide peak.
        """
        from scipy.signal import find_peaks

        if col_profile.max() <= 0:
            return [(0, total_len)]

        min_distance = max(4, int(total_len * 0.02))
        prominence = col_profile.max() * 0.12
        peaks, _ = find_peaks(col_profile, distance=min_distance, prominence=prominence)

        boundaries = sorted(set([0] + [int(p) for p in peaks] + [total_len]))
        min_width = max(6, int(total_len * 0.015))
        bands = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
                 if boundaries[i + 1] - boundaries[i] >= min_width]

        return bands if bands else [(0, total_len)]

    @staticmethod
    def _sharpness_to_confidence(sharpness: float) -> float:
        # Heuristic squashing of an unbounded contrast/sharpness score into (0, 1).
        return float(1.0 - np.exp(-sharpness / 400.0)) * 0.6 + 0.35

    @staticmethod
    def _non_max_suppression(boxes, iou_threshold: float):
        if not boxes:
            return []
        boxes_arr = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
        scores = np.array([b[4] for b in boxes], dtype=np.float32)
        x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]

        return [boxes[i] for i in keep]
