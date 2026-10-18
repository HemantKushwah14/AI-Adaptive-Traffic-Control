"""Computer-vision units. A FAKE detector (defined here, test-only) exercises the video pipeline;
no fake output is ever shown to users."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from computer_vision.ambulance_detector import (AmbulanceConfirmer, AmbulanceDetector,
                                                AmbulanceDetectorUnavailableError, AmbulanceSighting,
                                                locate_sightings)
from computer_vision.detector import Detection, VehicleDetector, draw_detections
from computer_vision.lane_analyzer import ApproachRegion, LaneAnalyzer, LaneRegionConfig, load_or_default
from computer_vision.vehicle_counter import CentroidTracker, VehicleCounter, count_by_class
from computer_vision.video_processor import TrafficVisionPipeline, decode_image_bytes, image_to_png_bytes
from config.cv_config import PRETRAINED_WEIGHTS, UNIFIED_CLASSES, VEHICLE_WEIGHTS


def det(x1, y1, x2, y2, label="car", conf=0.9):
    return Detection(x1, y1, x2, y2, conf, label, label, 0)


# ------------------------------------------------------------------ detector basics
def test_detection_geometry():
    d = det(10, 20, 30, 60)
    assert d.center == (20.0, 40.0) and d.bottom_center == (20.0, 60.0) and d.area == 800.0


def test_detector_argument_and_input_validation():
    for kw in (dict(conf=0.001), dict(iou=0.99), dict(imgsz=100)):
        with pytest.raises(ValueError):
            VehicleDetector(**kw)
    d = VehicleDetector()
    for bad in (None, np.zeros((10, 10), np.uint8), "image"):
        with pytest.raises(ValueError):
            d.predict(bad)           # rejected before any model is loaded


def test_draw_detections_returns_copy():
    frame = np.zeros((50, 80, 3), np.uint8)
    out = draw_detections(frame, [det(10, 10, 40, 40)])
    assert out is not frame and frame.sum() == 0 and out.sum() > 0 and out.shape == frame.shape


def test_image_encoding_roundtrip_and_bad_bytes():
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(decode_image_bytes(image_to_png_bytes(img)), img)
    with pytest.raises(ValueError):
        decode_image_bytes(b"definitely not an image")


@pytest.mark.slow
@pytest.mark.skipif(not (VEHICLE_WEIGHTS.exists() or PRETRAINED_WEIGHTS.exists()), reason="no local detector weights")
def test_real_detector_runs_and_reports_mode():
    d = VehicleDetector()
    assert d.model_info()["mode"] in {"finetuned", "pretrained-coco"}
    assert isinstance(d.predict(np.zeros((480, 640, 3), np.uint8)), list)


# ------------------------------------------------------------------ tracking / counting
def test_count_by_class_includes_all_unified_classes():
    c = count_by_class([det(0, 0, 5, 5, "car"), det(0, 0, 5, 5, "bus"), det(0, 0, 5, 5, "car")])
    assert set(c) >= set(UNIFIED_CLASSES) and c["car"] == 2 and c["bus"] == 1 and c["truck"] == 0


def test_tracker_keeps_identity_and_drops_lost_tracks():
    tr = CentroidTracker(max_distance=50, max_missed=2)
    ids = [tr.update([det(100 + 5 * k, 50, 120 + 5 * k, 70)])[0].track_id for k in range(5)]
    assert len(set(ids)) == 1
    far = tr.update([det(100, 50, 120, 70), det(400, 300, 420, 320)])
    assert far[0].track_id != far[1].track_id
    for _ in range(3):
        tr.update([])
    assert tr.tracks == {}
    assert tr.update([det(100, 50, 120, 70)])[0].track_id not in ids


def test_counter_counts_unique_vehicles_and_detects_stationary():
    vc = VehicleCounter(max_distance=50, max_missed=3, min_hits=3, min_points=10, stationary_px=3.0)
    for _ in range(12):
        last = vc.update([det(100, 50, 120, 70)], approaches=["N"])
    assert last.stationary == [True] and vc.unique_total == 1
    assert vc.unique_by_class["car"] == 1 and vc.unique_by_approach["N"]["car"] == 1
    mv = VehicleCounter(max_distance=50, max_missed=3, min_hits=3, min_points=10, stationary_px=3.0)
    for k in range(12):
        last = mv.update([det(20 * k, 50, 20 * k + 20, 70)])
    assert last.stationary == [False] and mv.unique_total == 1


# ------------------------------------------------------------------ lane analyzer
def template_analyzer():
    return LaneAnalyzer(LaneRegionConfig.default_template())


def test_region_assignment_density_and_queue():
    an = template_analyzer()
    shape = (100, 200, 3)                                     # h=100, w=200
    dets = [det(90, 0, 110, 10), det(90, 80, 110, 90, "bus"), det(180, 40, 200, 50), det(90, 20, 110, 30)]
    a = an.analyze(dets, shape, stationary=None)
    assert a.counts == {"N": 2, "S": 1, "E": 1, "W": 0} and a.class_counts["S"]["bus"] == 1
    assert a.density["N"] == pytest.approx(2 / 40) and a.queue == {k: None for k in "NSEW"}
    assert a.calibrated is False and a.assignments == ["N", "S", "E", "N"]
    b = an.analyze(dets, shape, stationary=[False, False, False, True])
    assert b.queue["N"] == 1 and b.queue["S"] == 0           # only the stationary vehicle inside the N queue zone
    assert len(b.density_vector()) == 4 and b.queue_vector()[0] == pytest.approx(1 / 30)


def test_unmeasured_approaches_are_none_not_zero():
    cfg = LaneRegionConfig({"N": ApproachRegion([(0, 0), (1, 0), (0.5, 0.5)])}, calibrated=True)
    a = LaneAnalyzer(cfg).analyze([det(90, 80, 110, 90)], (100, 200, 3))
    assert a.counts["S"] is None and a.density["S"] is None and a.unassigned == 1 and a.density_vector()[1] == 0.0


def test_lane_level_counts():
    n = [(0, 0), (1, 0), (0.5, 0.5)]
    reg = ApproachRegion(n, lanes={0: [(0, 0), (0.5, 0), (0.5, 0.5)], 1: [(0.5, 0), (1, 0), (0.5, 0.5)]})
    a = LaneAnalyzer(LaneRegionConfig({"N": reg})).analyze([det(40, 0, 60, 10), det(140, 0, 160, 10)], (100, 200, 3))
    assert a.lane_counts["N"] == {0: 1, 1: 1, 2: 0}


def test_region_config_validation_and_io(tmp_path):
    ok = [(0, 0), (1, 0), (1, 1)]
    for bad in (LaneRegionConfig({}), LaneRegionConfig({"X": ApproachRegion(ok)}),
                LaneRegionConfig({"N": ApproachRegion([(0, 0), (1, 1)])}),
                LaneRegionConfig({"N": ApproachRegion([(0, 0), (1.5, 0), (1, 1)])}),
                LaneRegionConfig({"N": ApproachRegion(ok, lanes={5: ok})})):
        with pytest.raises(ValueError):
            bad.validate()
    cfg = LaneRegionConfig.default_template()
    cfg.calibrated = True
    cfg.save(tmp_path / "r.json")
    back = LaneRegionConfig.load(tmp_path / "r.json")
    assert back.calibrated and set(back.approaches) == set("NSEW") and back.approaches["N"].polygon == cfg.approaches["N"].polygon
    (tmp_path / "bad.json").write_text("not json")
    with pytest.raises(ValueError):
        LaneRegionConfig.load(tmp_path / "bad.json")
    assert load_or_default(tmp_path / "missing.json").calibrated is False


def test_overlay_is_a_copy():
    frame = np.zeros((100, 200, 3), np.uint8)
    out = template_analyzer().draw_overlay(frame)
    assert out.shape == frame.shape and frame.sum() == 0 and out.sum() > 0


# ------------------------------------------------------------------ video pipeline with a test-only fake detector
class FakeDetector:
    mode = "fake"

    def predict(self, frame):
        return [det(70, 20, 90, 30)]               # a parked car: bottom-centre (80, 30) inside the N region

    def model_info(self):
        return {"mode": "fake"}


def test_video_pipeline_counts_tracks_and_writes_output(tmp_path):
    video = tmp_path / "in.avi"
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
    if not wr.isOpened():
        pytest.skip("OpenCV cannot write MJPG AVI on this machine")
    for k in range(20):
        wr.write(np.full((120, 160, 3), 40 + k, np.uint8))
    wr.release()
    pipe = TrafficVisionPipeline(FakeDetector(), template_analyzer())
    try:
        res = pipe.process_video(video, tmp_path / "out.mp4", stride=1, max_frames=100)
    except RuntimeError as exc:
        pytest.skip(f"no usable MP4 codec: {exc}")
    s = res.summary
    assert res.frames_processed == 20 and s["unique_vehicles_total"] == 1
    assert (res.per_frame["count_N"] == 1).all() and s["avg_count_by_approach"]["N"] == 1.0
    assert 0.0 < s["avg_queue_by_approach"]["N"] <= 1.0, "a parked car must be recognised as queued after enough frames"
    assert s["regions_calibrated"] is False and res.output_path.exists() and res.output_path.stat().st_size > 0
    with pytest.raises(ValueError):
        pipe.process_video(tmp_path / "missing.avi", tmp_path / "o.mp4")


# ------------------------------------------------------------------ ambulance detection
def test_ambulance_model_missing_gives_clear_error_without_fallback(tmp_path):
    d = AmbulanceDetector(weights=tmp_path / "nope.pt")
    assert not d.available()
    with pytest.raises(AmbulanceDetectorUnavailableError, match="No pretrained fallback"):
        d.load()
    with pytest.raises(ValueError):
        d.predict(None)
    with pytest.raises(ValueError):
        AmbulanceDetector(conf=0.01)


def test_locate_sightings_uses_regions_not_guesses():
    out = locate_sightings([det(90, 0, 110, 10, "ambulance")], (100, 200, 3), template_analyzer())
    assert out[0].approach == "N" and out[0].lane_index is None and out[0].calibrated is False


def test_confirmer_needs_multiple_frames_and_known_approach():
    d = det(0, 0, 10, 10, "ambulance", 0.9)
    cf = AmbulanceConfirmer(min_conf=0.5, confirm_frames=3, window=5)
    assert cf.update([AmbulanceSighting(d, "E", 2, True)]) == []
    assert cf.update([AmbulanceSighting(d, "E", 2, True)]) == []
    assert [s.approach for s in cf.update([AmbulanceSighting(d, "E", 2, True)])] == ["E"]
    cf2 = AmbulanceConfirmer(confirm_frames=1, window=1)
    assert cf2.update([AmbulanceSighting(d, None, None, True)]) == []                       # unknown approach
    assert cf2.update([AmbulanceSighting(det(0, 0, 1, 1, "ambulance", 0.2), "N", 0, True)]) == []   # low confidence
    with pytest.raises(ValueError):
        AmbulanceConfirmer(confirm_frames=6, window=5)