"""Class-wise counting, a light centroid tracker and stationary-vehicle detection.

Stationary heuristic (used for queue estimation): a track is 'stationary' when its ground
point moved less than `stationary_px` over its last `min_points` observations. This is a
heuristic - it depends on detector jitter and frame stride, so tune it on your own video.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from computer_vision.detector import Detection
from config.cv_config import UNIFIED_CLASSES


def count_by_class(detections: Sequence[Detection]) -> Dict[str, int]:
    counts = {c: 0 for c in UNIFIED_CLASSES}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    return counts


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]
    history: Deque[Tuple[float, float]]
    votes: Counter
    hits: int = 1
    missed: int = 0

    @property
    def label(self) -> str:
        return self.votes.most_common(1)[0][0]

    def is_stationary(self, min_points: int, max_disp: float) -> bool:
        if len(self.history) < min_points:
            return False
        pts = np.asarray(self.history, dtype=float)
        return float(np.linalg.norm(pts[-1] - pts[0])) <= max_disp and \
            float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1))) <= max_disp * 1.5


class CentroidTracker:
    """Greedy nearest-neighbour association on bottom-centre points."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 8, history: int = 12) -> None:
        self.max_distance, self.max_missed, self.history = max_distance, max_missed, history
        self.tracks: Dict[int, Track] = {}
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1

    def update(self, detections: Sequence[Detection]) -> Dict[int, Track]:
        """Returns {detection_index: Track}."""
        pts = np.array([d.bottom_center for d in detections], dtype=float).reshape(-1, 2)
        ids = list(self.tracks)
        used_t, used_d = set(), set()
        out: Dict[int, Track] = {}
        if ids and len(pts):
            tp = np.array([self.tracks[t].centroid for t in ids], dtype=float)
            dist = np.linalg.norm(tp[:, None, :] - pts[None, :, :], axis=2)
            order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
            for ti, di in order:
                if ti in used_t or di in used_d:
                    continue
                if dist[ti, di] > self.max_distance:
                    break
                tr = self.tracks[ids[ti]]
                tr.centroid = (float(pts[di][0]), float(pts[di][1]))
                tr.history.append(tr.centroid)
                tr.votes[detections[di].label] += 1
                tr.hits += 1
                tr.missed = 0
                used_t.add(ti)
                used_d.add(di)
                out[int(di)] = tr
        for ti, tid in enumerate(ids):
            if ti not in used_t:
                self.tracks[tid].missed += 1
                if self.tracks[tid].missed > self.max_missed:
                    del self.tracks[tid]
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            c = (float(pts[di][0]), float(pts[di][1]))
            tr = Track(self._next, c, deque([c], maxlen=self.history), Counter({d.label: 1}))
            self.tracks[self._next] = tr
            self._next += 1
            out[di] = tr
        return out


@dataclass
class FrameTracking:
    track_ids: List[Optional[int]]
    stationary: List[bool]
    class_counts: Dict[str, int]


class VehicleCounter:
    """Per-frame counts + cumulative unique-vehicle counts (confirmed tracks only)."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 8, min_hits: int = 3,
                 min_points: int = 10, stationary_px: float = 8.0) -> None:
        self.tracker = CentroidTracker(max_distance, max_missed, history=max(min_points, 12))
        self.min_hits, self.min_points, self.stationary_px = min_hits, min_points, stationary_px
        self.reset()

    def reset(self) -> None:
        self.tracker.reset()
        self._counted: set = set()
        self.unique_by_class: Counter = Counter()
        self.unique_by_approach: Dict[str, Counter] = defaultdict(Counter)

    def update(self, detections: Sequence[Detection],
               approaches: Optional[Sequence[Optional[str]]] = None) -> FrameTracking:
        mapping = self.tracker.update(detections)
        n = len(detections)
        ids: List[Optional[int]] = [None] * n
        stat = [False] * n
        for di, tr in mapping.items():
            ids[di] = tr.track_id
            stat[di] = tr.is_stationary(self.min_points, self.stationary_px)
            if tr.hits >= self.min_hits and tr.track_id not in self._counted:
                self._counted.add(tr.track_id)
                self.unique_by_class[tr.label] += 1
                ap = approaches[di] if approaches is not None else None
                self.unique_by_approach[ap or "unassigned"][tr.label] += 1
        return FrameTracking(ids, stat, count_by_class(detections))

    @property
    def unique_total(self) -> int:
        return int(sum(self.unique_by_class.values()))