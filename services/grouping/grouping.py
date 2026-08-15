"""
services/grouping/grouping.py

Product grouping: assigns a unique group_id to detected products that appear
to belong to the same brand/packaging family.

WHY FEATURE CLUSTERING INSTEAD OF A BRAND CLASSIFIER
------------------------------------------------------
A real brand classifier needs a labeled brand dataset and (ideally) an
embedding model such as CLIP. The offline build environment has neither
internet access nor a cached embedding model. Instead, this module clusters
products by *visual similarity* -- color palette + texture -- which is a
reasonable, zero-training proxy for "same brand": products from the same
brand/SKU family share packaging color scheme and print texture, which is
exactly what repeats on a shelf (see the reference image in the samples:
each shelf row is visually homogeneous per brand).

Swap-in path for production: replace `_extract_features()` with embeddings
from a pretrained model (e.g. CLIP image encoder, or a brand-classification
head) and keep everything downstream (clustering + the HTTP contract)
unchanged.
"""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist


class ProductGrouper:
    def __init__(self, crop_size: int = 64, cluster_percentile: float = 35.0):
        self.crop_size = crop_size
        self.cluster_percentile = cluster_percentile  # lower -> more, smaller groups

    def group(self, image_bgr: np.ndarray, detections: list):
        """
        Parameters
        ----------
        image_bgr : np.ndarray  full original image
        detections : list[dict] each with "bbox": [x1,y1,x2,y2]

        Returns
        -------
        grouped_detections : list[dict]  same detections, each with an added "group_id" (int)
        groups_summary : list[dict]  [{"group_id": int, "member_detection_ids": [...], "size": int}]
        """
        n = len(detections)
        if n == 0:
            return [], []

        if n == 1:
            d = dict(detections[0])
            d["group_id"] = 0
            return [d], [{"group_id": 0, "member_detection_ids": [d.get("detection_id", 0)], "size": 1}]

        features = np.stack([self._extract_features(image_bgr, d["bbox"]) for d in detections])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        labels = self._cluster(features_scaled)

        grouped = []
        for d, label in zip(detections, labels):
            gd = dict(d)
            gd["group_id"] = int(label)
            grouped.append(gd)

        groups_summary = {}
        for d in grouped:
            gid = d["group_id"]
            groups_summary.setdefault(gid, []).append(d.get("detection_id"))
        groups_summary = [
            {"group_id": gid, "member_detection_ids": members, "size": len(members)}
            for gid, members in sorted(groups_summary.items())
        ]
        return grouped, groups_summary

    # ------------------------------------------------------------------ #
    def _extract_features(self, image_bgr: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        color_feat = np.concatenate([hist_h, hist_s, hist_v])
        color_feat = color_feat / (color_feat.sum() + 1e-6)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        n_bins = int(lbp.max() + 1)
        texture_feat, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        mean_color = crop.reshape(-1, 3).mean(axis=0) / 255.0

        return np.concatenate([color_feat, texture_feat, mean_color]).astype(np.float32)

    def _cluster(self, features_scaled: np.ndarray) -> np.ndarray:
        n = features_scaled.shape[0]
        dists = pdist(features_scaled)
        if dists.size == 0 or np.allclose(dists, 0):
            return np.zeros(n, dtype=int)

        threshold = np.percentile(dists, self.cluster_percentile)
        threshold = max(threshold, 1e-3)

        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, linkage="average", metric="euclidean"
        )
        labels = clustering.fit_predict(features_scaled)

        # relabel so group ids are 0..k-1 ordered by first appearance
        seen = {}
        relabeled = []
        for lab in labels:
            if lab not in seen:
                seen[lab] = len(seen)
            relabeled.append(seen[lab])
        return np.array(relabeled, dtype=int)
