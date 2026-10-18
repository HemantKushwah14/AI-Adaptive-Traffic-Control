"""Dataset parsing, YOLO conversion, leakage-free splitting and validation (synthetic data only)."""
from __future__ import annotations

import shutil

import pytest
from PIL import Image

from dataset.convert_annotations import (Box, box_to_yolo, boxes_to_label_lines, build_class_list,
                                         parse_detrac_xml, parse_merge_args)
from dataset.download_dataset import select_files
from dataset.prepare_ambulance_dataset import parse_label
from dataset.prepare_ambulance_dataset import prepare as prepare_ambulance
from dataset.prepare_dataset import prepare, split_sequences
from dataset.validate_dataset import inspect_raw, validate_yolo

XML = """<sequence name="MVI_00001">
 <ignored_region><box left="0" top="0" width="5" height="5"/></ignored_region>
 <frame num="1" density="2"><target_list>
  <target id="1"><box left="10" top="20" width="30" height="20"/><attribute vehicle_type="Car"/></target>
  <target id="2"><box left="1" top="2" width="3" height="4"/></target>
 </target_list></frame>
 <frame num="2"><target_list/></frame>
</sequence>"""


def test_parse_detrac_xml(tmp_path):
    p = tmp_path / "MVI_00001.xml"
    p.write_text(XML)
    ann = parse_detrac_xml(p)
    assert ann.name == "MVI_00001" and ann.ignored_regions == [(0.0, 0.0, 5.0, 5.0)]
    assert [b.cls for b in ann.frames[1]] == ["car"] and ann.frames[2] == []
    assert ann.malformed == 1, "a target without <attribute> must be counted, not invented"


def test_parse_rejects_bad_files(tmp_path):
    (tmp_path / "a.xml").write_text("<annotation/>")
    (tmp_path / "b.xml").write_text("<sequence")
    for name in ("a.xml", "b.xml", "missing.xml"):
        with pytest.raises(ValueError):
            parse_detrac_xml(tmp_path / name)


def test_box_to_yolo_normalises_and_clips():
    assert box_to_yolo(Box("car", 10, 20, 30, 20), 100, 80) == pytest.approx((0.25, 0.375, 0.3, 0.25))
    assert box_to_yolo(Box("car", 90, 70, 30, 30), 100, 80) == pytest.approx((0.95, 0.9375, 0.1, 0.125))
    assert box_to_yolo(Box("car", 10, 10, 1, 30), 100, 80) is None


def test_class_list_is_built_from_found_classes_only():
    names, mapping = build_class_list(["others", "bus", "car", "van"], {})
    assert names == ["car", "van", "bus", "others"] and mapping["bus"] == 2
    names, mapping = build_class_list(["car", "van", "bus"], {"van": "car"})
    assert names == ["car", "bus"] and mapping == {"car": 0, "van": 0, "bus": 1}
    lines, dropped = boxes_to_label_lines([Box("car", 10, 20, 30, 20), Box("tuktuk", 1, 1, 20, 20)], mapping, 100, 80)
    assert lines == ["0 0.250000 0.375000 0.300000 0.250000"] and dropped == 1


def test_merge_argument_parsing():
    assert parse_merge_args(["van=car", "others=truck"]) == {"van": "car", "others": "truck"}
    for bad in (["van"], ["=car"]):
        with pytest.raises(ValueError):
            parse_merge_args(bad)


def test_split_is_disjoint_complete_and_deterministic():
    names = [f"MVI_{i:05d}" for i in range(10)]
    s = split_sequences(names, 0.7, 0.15, seed=1)
    assert sorted(sum(s.values(), [])) == sorted(names)
    assert not (set(s["train"]) & set(s["val"]) or set(s["train"]) & set(s["test"]) or set(s["val"]) & set(s["test"]))
    assert all(len(v) >= 1 for v in s.values()) and s == split_sequences(names, 0.7, 0.15, seed=1)
    with pytest.raises(ValueError):
        split_sequences(names[:2], 0.7, 0.15, 1)
    with pytest.raises(ValueError):
        split_sequences(names, 0.9, 0.2, 1)


# ------------------------------------------------------------------ synthetic raw dataset
def make_raw(root, n_seq=5, frames=4):
    ann_dir = root / "DETRAC-Annotations"
    ann_dir.mkdir(parents=True)
    for s in range(1, n_seq + 1):
        seq = f"MVI_{s:05d}"
        img_dir = root / "DETRAC-Images" / seq
        img_dir.mkdir(parents=True)
        fx = []
        for f in range(1, frames + 1):
            Image.new("RGB", (100, 80), (s * 20, f * 30, 100)).save(img_dir / f"img{f:05d}.jpg")
            fx.append(f'<frame num="{f}" density="2"><target_list>'
                      '<target id="1"><box left="10" top="20" width="30" height="20"/><attribute vehicle_type="car"/></target>'
                      '<target id="2"><box left="50" top="10" width="40" height="30"/><attribute vehicle_type="bus"/></target>'
                      "</target_list></frame>")
        (ann_dir / f"{seq}.xml").write_text(f'<sequence name="{seq}">{"".join(fx)}</sequence>')


def test_prepare_pipeline_end_to_end(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "yolo"
    make_raw(raw)
    assert inspect_raw(raw) == 0
    text = capsys.readouterr().out
    assert "car" in text and "bus" in text and "motorcycle" in text, "missing classes must be reported honestly"
    rep = prepare(raw, out, {})
    assert rep["training_classes"] == ["car", "bus"]
    sp = rep["split_sequences"]
    assert len(sp["train"]) == 3 and len(sp["val"]) == 1 and len(sp["test"]) == 1
    assert rep["splits"]["train"]["images"] == 12 and rep["splits"]["train"]["instances"] == {"car": 12, "bus": 12}
    label = next((out / "labels" / "train").glob("*.txt")).read_text()
    assert "0 0.250000 0.375000 0.300000 0.250000" in label
    assert validate_yolo(out) == 0
    with pytest.raises(FileExistsError):
        prepare(raw, out, {})
    assert prepare(raw, out, {"bus": "car"}, overwrite=True)["training_classes"] == ["car"]
    assert prepare(raw, out, {}, frame_stride=2, overwrite=True)["splits"]["train"]["images"] == 6


def test_prepare_rejects_missing_or_too_few_sequences(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "nope", tmp_path / "o", {})
    make_raw(tmp_path / "raw2", n_seq=2)
    with pytest.raises(ValueError):
        prepare(tmp_path / "raw2", tmp_path / "o2", {})


def test_validation_detects_sequence_leakage(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "yolo"
    make_raw(raw)
    prepare(raw, out, {})
    src = next((out / "images" / "train").glob("*.jpg"))
    shutil.copy2(src, out / "images" / "val" / f"{src.stem}_copy.jpg")
    shutil.copy2(out / "labels" / "train" / f"{src.stem}.txt", out / "labels" / "val" / f"{src.stem}_copy.txt")
    assert validate_yolo(out) == 1


def test_validation_flags_missing_files(tmp_path):
    assert inspect_raw(tmp_path / "nothing") == 1
    assert validate_yolo(tmp_path / "nothing") == 1


def test_download_selection_is_bounded_and_complete():
    files = []
    for s in range(1, 6):
        seq = f"MVI_{s:05d}"
        files += [f"DETRAC/DETRAC-Images/DETRAC-Images/{seq}/img{f:05d}.jpg" for f in range(1, 21)]
        files.append(f"DETRAC/DETRAC-Annotations-XML/{seq}.xml")
    files += [f"DETRAC/DETRAC-Images/DETRAC-Images/MVI_00009/img{f:05d}.jpg" for f in range(1, 5)]   # no XML
    chosen, selected, usable = select_files(files, max_sequences=3, frame_stride=10, seed=42)
    assert usable == 5 and len(chosen) == 3 and "MVI_00009" not in chosen
    imgs = [f for f in selected if f.endswith(".jpg")]
    assert imgs and all(int(f[-9:-4]) in (1, 11) for f in imgs)
    assert all(any(f"{c}.xml" in f for f in selected) for c in chosen)
    assert all(any(c in f for c in chosen) for f in selected)
    with pytest.raises(RuntimeError):
        select_files([f for f in files if not f.endswith(".xml")], 3, 10, 42)


# ------------------------------------------------------------------ ambulance dataset
def test_parse_label_boxes_polygons_and_bad_lines(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.4 0.4\n0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n1 0.5 0.5 0.2 0.2\n0 a b\n0 0.5 0.5 0 0\n")
    lines, bad = parse_label(p, {0})
    assert lines == ["0 0.500000 0.500000 0.400000 0.400000", "0 0.300000 0.300000 0.400000 0.400000"]
    assert bad == 2


def make_ambulance_src(root, empty_test=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("names: [ambulance, car]\n")
    k = 0
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(3):
            k += 1
            Image.new("RGB", (32, 32), (10 * k, 200 - 5 * k, k)).save(root / split / "images" / f"{split}{i}.jpg")
            body = "" if (empty_test and split == "test") else "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n"
            (root / split / "labels" / f"{split}{i}.txt").write_text(body)


def test_ambulance_preparation_single_class_dedup_and_negatives(tmp_path):
    src, out, neg = tmp_path / "src", tmp_path / "out", tmp_path / "neg"
    make_ambulance_src(src)
    shutil.copy2(src / "train" / "images" / "train0.jpg", src / "test" / "images" / "dup.jpg")
    shutil.copy2(src / "train" / "labels" / "train0.txt", src / "test" / "labels" / "dup.txt")
    for sp in ("train", "val", "test"):
        (neg / "images" / sp).mkdir(parents=True)
        for i in range(7):
            Image.new("RGB", (16, 16), (200 + i, 3 * i, 9)).save(neg / "images" / sp / f"n{sp}{i}.jpg")
    rep = prepare_ambulance(src, out, keep_names={"ambulance"}, keep_ids_opt=None, negatives_from=neg,
                            max_negatives=10, seed=42, overwrite=False)
    assert rep["kept_source_class_ids"] == [0]
    assert rep["splits"]["train"]["boxes"] == 3 and rep["splits"]["test"]["boxes"] == 3, "duplicate must be dropped"
    assert any("duplicate" in w for w in rep["warnings"]) and rep["negatives_added"] == 10
    assert "ambulance" in (out / "data.yaml").read_text()
    neg_label = next((out / "labels" / "test").glob("neg_*.txt"))
    assert neg_label.read_text() == ""


def test_ambulance_preparation_requires_positives_in_every_split(tmp_path):
    make_ambulance_src(tmp_path / "src", empty_test=True)
    with pytest.raises(ValueError, match="test"):
        prepare_ambulance(tmp_path / "src", tmp_path / "out", keep_names={"ambulance"}, keep_ids_opt=None,
                          negatives_from=None, max_negatives=400, seed=1, overwrite=False)
    with pytest.raises(ValueError):
        prepare_ambulance(tmp_path / "src", tmp_path / "out2", keep_names={"fire_truck"}, keep_ids_opt=None,
                          negatives_from=None, max_negatives=400, seed=1, overwrite=False)