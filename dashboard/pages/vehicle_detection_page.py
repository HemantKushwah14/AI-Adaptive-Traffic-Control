"""Page 3 - AI Vehicle Detection. Real YOLO predictions only (fine-tuned or COCO-pretrained,
clearly labelled). Results live in st.session_state so download clicks never recompute."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from computer_vision.detector import DetectorUnavailableError, VehicleDetector
from computer_vision.lane_analyzer import LaneAnalyzer, load_or_default
from computer_vision.video_processor import TrafficVisionPipeline, decode_image_bytes, image_to_png_bytes
from config.constants import APPROACH_NAMES, APPROACHES
from config.cv_config import UNIFIED_CLASSES, VISION_OUT_DIR
from dashboard.components import ui
from dashboard.styles.theme import COLORS
from dashboard.visualizations import charts

CLASS_HEX = {"car": COLORS["cyan"], "motorcycle": COLORS["amber"], "bus": COLORS["green"],
             "truck": COLORS["purple"]}


@st.cache_resource(show_spinner=False)
def _detector(imgsz: int) -> VehicleDetector:
    det = VehicleDetector(imgsz=imgsz)
    det.load()
    return det


def _class_bar(counts: dict, title: str, height: int = 260) -> go.Figure:
    names = list(counts)
    vals = [counts[n] for n in names]
    fig = go.Figure(go.Bar(x=names, y=vals, text=vals, textposition="outside",
                           marker_color=[CLASS_HEX.get(n, COLORS["blue"]) for n in names]))
    fig = charts.style(fig, height, title, legend=False)
    fig.update_xaxes(title_text="")
    return fig


def _model_banner(det: VehicleDetector) -> None:
    info = det.model_info()
    tone = {"finetuned": "green", "pretrained-coco": "amber"}.get(info["mode"], "blue")
    ui.alert(f'{info["note"]}  Weights: {info["weights"]}', tone, f'Detector mode: {info["mode"]}')


def _approach_table(a: dict) -> str:
    rows = []
    for ap in APPROACHES:
        cnt, dens, q = a["counts"][ap], a["density"][ap], a["queue"][ap]
        rows.append((APPROACH_NAMES[ap], "not measured" if cnt is None else str(cnt),
                     "—" if dens is None else f"{dens * 100:.0f}%",
                     "n/a" if q is None else str(q)))
    return ui.html_table(["Approach", "Vehicles in region", "Density (capacity-normalised)", "Queue"], rows)


def _calibration_notice(calibrated: bool) -> None:
    if not calibrated:
        ui.alert("Lane regions are UNCALIBRATED, so per-approach numbers are not meaningful. Class counts and "
                 "boxes are unaffected. Calibrate with: python scripts\\calibrate_regions.py --image <frame.jpg>",
                 "amber", "Approach assignment is a demo template")


# ------------------------------------------------------------------ image tab
def _image_tab(det: VehicleDetector, conf: float, iou: float) -> None:
    up = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"], key="det_img_up")
    if st.button("Run vehicle detection", type="primary", disabled=up is None, key="det_img_run"):
        try:
            det.conf, det.iou = conf, iou
            pipe = TrafficVisionPipeline(det, LaneAnalyzer(load_or_default()))
            with st.spinner("Detecting vehicles ..."):
                r = pipe.process_image(decode_image_bytes(up.getvalue()))
            rows = [{"class": d.label, "model_label": d.raw_label, "confidence": round(d.confidence, 3),
                     "x1": round(d.x1), "y1": round(d.y1), "x2": round(d.x2), "y2": round(d.y2),
                     "approach": r.analysis.assignments[i] or "-"} for i, d in enumerate(r.detections)]
            df = pd.DataFrame(rows, columns=["class", "model_label", "confidence", "x1", "y1", "x2", "y2", "approach"])
            st.session_state["det_img"] = {
                "name": up.name, "annotated": r.annotated, "png": image_to_png_bytes(r.annotated), "df": df,
                "class_counts": r.class_counts, "analysis": r.analysis.to_dict(), "ms": r.inference_ms,
                "mode": det.mode, "conf": conf}
        except (DetectorUnavailableError, ValueError, RuntimeError) as exc:
            ui.alert(str(exc), "red", "Detection failed", preformatted=True)
    res = st.session_state.get("det_img")
    if not res:
        st.caption("Upload an image, then press Run. No test image? Use any frame under data\\detrac_raw.")
        return
    st.caption(f'Result for **{res["name"]}** · mode {res["mode"]} · confidence ≥ {res["conf"]:.2f} · '
               f'inference {res["ms"]:.0f} ms (CPU)')
    left, right = st.columns([1.6, 1])
    with left:
        st.image(res["annotated"], channels="BGR", use_column_width=True)
    with right:
        cc = res["class_counts"]
        ui.kpi_grid([dict(label="Total vehicles", value=str(sum(cc.values())), sub="above threshold", accent="cyan", icon="🚗"),
                     dict(label="Cars", value=str(cc.get("car", 0)), sub="", accent="blue", icon="🚙"),
                     dict(label="Buses / trucks", value=f'{cc.get("bus", 0)} / {cc.get("truck", 0)}', sub="", accent="green", icon="🚌"),
                     dict(label="Motorcycles", value=str(cc.get("motorcycle", 0)), sub="", accent="amber", icon="🏍")], cols=2)
        st.plotly_chart(_class_bar(cc, "Class-wise counts"), use_container_width=True,
                        config=charts.PLOT_CONFIG, key="det_img_bar")
    _calibration_notice(res["analysis"]["calibrated"])
    ui.md(ui.card("Per-approach analysis (region-based)", _approach_table(res["analysis"])
                  + f'<div class="muted" style="margin-top:8px">Queue: {ui.esc(res["analysis"]["queue_method"])}.'
                    f' Detections outside every region: {res["analysis"]["unassigned"]}.</div>'))
    st.dataframe(res["df"].sort_values("confidence", ascending=False), use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("⬇  Annotated image (PNG)", res["png"], file_name=f'{Path(res["name"]).stem}_detected.png',
                       mime="image/png", use_container_width=True)
    d2.download_button("⬇  Detections (CSV)", res["df"].to_csv(index=False).encode("utf-8"),
                       file_name=f'{Path(res["name"]).stem}_detections.csv', mime="text/csv",
                       use_container_width=True)


# ------------------------------------------------------------------ video tab
def _video_tab(det: VehicleDetector, conf: float, iou: float) -> None:
    up = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"], key="det_vid_up")
    c = st.columns(2)
    stride = c[0].slider("Process every N-th frame", 1, 10, 3, key="det_vid_stride")
    max_frames = c[1].slider("Maximum frames to process", 30, 1500, 300, 30, key="det_vid_max")
    st.caption("Runs on CPU: roughly 0.1-0.5 s per processed frame for yolov8n at 640 px. Start small.")
    if st.button("Run vehicle detection on video", type="primary", disabled=up is None, key="det_vid_run"):
        in_path = None
        try:
            det.conf, det.iou = conf, iou
            pipe = TrafficVisionPipeline(det, LaneAnalyzer(load_or_default()))
            upload_dir = VISION_OUT_DIR / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            in_path = upload_dir / f"{int(time.time())}_{Path(up.name).name}"
            in_path.write_bytes(up.getvalue())
            out_path = VISION_OUT_DIR / f"{in_path.stem}_detected.mp4"
            bar = st.progress(0.0, text="Processing video ...")
            r = pipe.process_video(in_path, out_path, stride, max_frames,
                                   progress_cb=lambda d, t: bar.progress(min(d / max(t, 1), 1.0),
                                                                         text=f"Frame {d} / {t}"))
            bar.empty()
            st.session_state["det_vid"] = {
                "name": up.name, "summary": r.summary, "df": r.per_frame, "samples": r.sample_frames,
                "video": Path(r.output_path).read_bytes(), "codec": r.codec, "stride": stride,
                "mode": det.mode, "conf": conf}
        except (DetectorUnavailableError, ValueError, RuntimeError) as exc:
            ui.alert(str(exc), "red", "Video processing failed", preformatted=True)
        finally:
            if in_path is not None:
                in_path.unlink(missing_ok=True)
    res = st.session_state.get("det_vid")
    if not res:
        return
    s, df = res["summary"], res["df"]
    st.caption(f'Result for **{res["name"]}** · mode {res["mode"]} · confidence ≥ {res["conf"]:.2f} · every {res["stride"]}-th frame')
    ui.kpi_grid([dict(label="Frames processed", value=str(s["frames_processed"]), sub="", accent="blue", icon="🎞"),
                 dict(label="Unique vehicles (est.)", value=str(s["unique_vehicles_total"]), sub="tracker-based estimate", accent="cyan", icon="🚗"),
                 dict(label="Inference", value=f'{s["avg_inference_ms"]:.0f} ms', sub="per processed frame (CPU)", accent="amber", icon="⚡"),
                 dict(label="Regions", value="Calibrated" if s["regions_calibrated"] else "Template",
                      sub="approach assignment", accent="green" if s["regions_calibrated"] else "amber", icon="📐")], cols=4)
    _calibration_notice(s["regions_calibrated"])
    ui.alert(s["tracking_note"], "blue", "How to read the counts")
    left, right = st.columns([1.5, 1])
    with left:
        if res["codec"] == "avc1":
            st.video(res["video"])
        else:
            st.caption(f'Codec "{res["codec"]}" may not play in the browser. Use the download below; sample frames are shown here.')
        samples = res["samples"]
        for i in range(0, len(samples), 3):
            cols = st.columns(3)
            for col, fr in zip(cols, samples[i:i + 3]):
                col.image(fr, channels="BGR", use_column_width=True)
    with right:
        st.plotly_chart(_class_bar(s["unique_vehicles_by_class"] or {k: 0 for k in UNIFIED_CLASSES},
                                   "Unique vehicles by class"), use_container_width=True,
                        config=charts.PLOT_CONFIG, key="det_vid_bar")
        rows = [(APPROACH_NAMES[a],
                 "—" if s["avg_count_by_approach"][a] is None else f'{s["avg_count_by_approach"][a]:.1f}',
                 "—" if s["peak_count_by_approach"][a] is None else str(s["peak_count_by_approach"][a]),
                 "—" if s["avg_queue_by_approach"][a] is None else f'{s["avg_queue_by_approach"][a]:.1f}')
                for a in APPROACHES]
        ui.md(ui.card("Per approach (region-based)", ui.html_table(["Approach", "Avg count", "Peak", "Avg queue"], rows)))
    fig = go.Figure()
    for col, name, color in (("total", "All vehicles", COLORS["cyan"]), ("cls_car", "Cars", COLORS["blue"]),
                             ("cls_bus", "Buses", COLORS["green"]), ("cls_truck", "Trucks", COLORS["purple"]),
                             ("cls_motorcycle", "Motorcycles", COLORS["amber"])):
        if col in df:
            fig.add_trace(go.Scatter(x=df["time_s"], y=df[col], name=name, mode="lines", line=dict(color=color, width=2)))
    fig = charts.style(fig, 280, "Detections per processed frame")
    fig.update_xaxes(title_text="video time (s)")
    st.plotly_chart(fig, use_container_width=True, config=charts.PLOT_CONFIG, key="det_vid_line")
    d = st.columns(3)
    d[0].download_button("⬇  Processed video (MP4)", res["video"], file_name=f'{Path(res["name"]).stem}_detected.mp4',
                         mime="video/mp4", use_container_width=True)
    d[1].download_button("⬇  Per-frame data (CSV)", df.to_csv(index=False).encode("utf-8"),
                         file_name=f'{Path(res["name"]).stem}_per_frame.csv', mime="text/csv", use_container_width=True)
    d[2].download_button("⬇  Summary (JSON)", json.dumps(s, indent=2, default=str).encode("utf-8"),
                         file_name=f'{Path(res["name"]).stem}_summary.json', mime="application/json",
                         use_container_width=True)


def render() -> None:
    ui.page_header("AI Vehicle Detection",
                   "YOLO (CNN) detection, class-wise counting, region-based density and queue estimation.", "🎥")
    with st.container(border=True):
        c = st.columns(3)
        conf = c[0].slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
        iou = c[1].slider("NMS IoU threshold", 0.30, 0.90, 0.50, 0.05)
        imgsz = c[2].select_slider("Inference size (px)", [320, 480, 640], value=640)
    try:
        with st.spinner("Loading detector (first run downloads yolov8n.pt, needs internet) ..."):
            det = _detector(int(imgsz))
        _model_banner(det)
    except DetectorUnavailableError as exc:
        ui.alert(str(exc), "red", "Detector unavailable", preformatted=True)
        return
    tab_img, tab_vid = st.tabs(["🖼  Image", "🎬  Video"])
    with tab_img:
        _image_tab(det, conf, iou)
    with tab_vid:
        _video_tab(det, conf, iou)