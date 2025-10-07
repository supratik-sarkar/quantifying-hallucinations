import os, json, csv
from typing import List, Dict
from pathlib import Path
import yaml
from PIL import Image

# ---------- path-safe YAML loader (relative to repo root) ----------
def _load_yaml_rel_to_repo(rel_path: str):
    repo_root = Path(__file__).resolve().parents[2]  # src/io -> src -> REPO
    p = (Path(rel_path) if Path(rel_path).is_absolute() else (repo_root / rel_path))
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _make_synthetic(n: int, task: str):
    recs=[]
    for i in range(n):
        if "audio" in task:
            recs.append({"id": i, "audio": f"dummy_{i}.wav", "text": f"synthetic audio caption {i}"})
        else:
            recs.append({"id": i, "image": f"dummy_{i}.jpg", "text": f"synthetic caption {i}"})
    return recs

# ------------------- COCO Captions loader -------------------
def _load_coco_val_from_fs(n: int, image_dir: str, captions_json: str) -> List[Dict]:
    image_dir = Path(image_dir); ann_path = Path(captions_json)
    if not image_dir.exists() or not ann_path.exists():
        raise RuntimeError(f"COCO paths missing: {image_dir} or {ann_path}")
    ann = json.load(open(ann_path))
    # Build id->file, id->captions
    id_to_file = {img["id"]: img["file_name"] for img in ann["images"]}
    id_to_caps = {}
    for c in ann["annotations"]:
        id_to_caps.setdefault(c["image_id"], []).append(c["caption"])
    # COCO val2017 files are directly in image_dir
    recs=[]
    for img_id, fname in id_to_file.items():
        fpath = image_dir / fname
        if not fpath.exists(): continue
        caps = id_to_caps.get(img_id, [])
        if not caps: continue
        try:
            im = Image.open(fpath).convert("RGB")
        except Exception:
            continue
        recs.append({"id": int(img_id), "image": im, "text": caps[0]})
        if len(recs) >= n: break
    if not recs:
        raise RuntimeError("COCO val set found but no records were loaded; check paths.")
    return recs

# ------------------- VQAv2 loader -------------------
def _load_vqa2_val_from_fs(n: int, image_dir: str, questions_json: str, annotations_json: str) -> List[Dict]:
    image_dir = Path(image_dir)
    q_path = Path(questions_json); a_path = Path(annotations_json)
    if not image_dir.exists() or not q_path.exists() or not a_path.exists():
        raise RuntimeError(f"VQAv2 paths missing: {image_dir}, {q_path}, or {a_path}")
    qs = json.load(open(q_path))["questions"]
    anns = json.load(open(a_path))["annotations"]
    ann_by_qid = {a["question_id"]: a for a in anns}
    recs=[]
    for q in qs:
        qid = q["question_id"]; img_id = q["image_id"]
        # VQA val uses MSCOCO val2014 naming: COCO_val2014_000000XXXXXX.jpg
        fname = f"COCO_val2014_{int(img_id):012d}.jpg"
        fpath = image_dir / fname
        if not fpath.exists(): continue
        a = ann_by_qid.get(qid)
        if not a: continue
        answers = a.get("answers", [])
        answer_text = answers[0]["answer"] if answers else ""
        try:
            im = Image.open(fpath).convert("RGB")
        except Exception:
            continue
        recs.append({"id": int(qid), "image": im, "text": q["question"], "answer": answer_text})
        if len(recs) >= n: break
    if not recs:
        raise RuntimeError("VQAv2 val set found but no records were loaded; check paths.")
    return recs

# ------------------- AudioCaps loader -------------------
def _load_audiocaps_val_from_fs(n: int, captions_csv: str, audio_dir: str=None) -> List[Dict]:
    csv_path = Path(captions_csv)
    if not csv_path.exists():
        raise RuntimeError(f"AudioCaps CSV missing: {csv_path}")
    recs=[]
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i,row in enumerate(reader):
            cap = row.get("caption") or row.get("Cap") or row.get("text") or ""
            ytid = row.get("ytid") or row.get("youtube_id") or f"id{i}"
            aud = None
            if audio_dir:
                # optional: if you have wavs, you can point to them
                cand = Path(audio_dir) / f"{ytid}.wav"
                aud = str(cand) if cand.exists() else None
            recs.append({"id": i, "audio": aud or f"dummy_{i}.wav", "text": cap})
            if len(recs) >= n: break
    if not recs:
        raise RuntimeError("AudioCaps CSV read but no rows parsed; check schema.")
    return recs

# ------------------- public API -------------------
def try_load(name: str, split: str, max_samples: int, task: str, allow_synth: bool=None):
    # honor config flag (robust path regardless of CWD)
    if allow_synth is None:
        try:
            cfg = _load_yaml_rel_to_repo("configs/default.yaml")
            allow_synth = bool(cfg.get("allow_synthetic", False))
        except FileNotFoundError:
            allow_synth = False

    name_l = name.lower()
    # Resolve dataset-specific config (from its YAML)
    # The run_pipeline passes cfg_path; we can't import it here cleanly, so read
    # the three known YAMLs if present, otherwise rely on default fields.
    repo_root = Path(__file__).resolve().parents[2]
    cfgs = {}
    for tag in ["coco-clip.yaml","vqa2-llava.yaml","audiocaps.yaml"]:
        p = repo_root / "configs" / tag
        if p.exists():
            cfgs[tag] = _load_yaml_rel_to_repo(f"configs/{tag}")

    try:
        if "coco" in name_l:
            ds = cfgs.get("coco-clip.yaml", {}).get("dataset", {})
            return _load_coco_val_from_fs(
                max_samples, ds.get("image_dir",""), ds.get("captions_json","")
            )

        if "vqa" in name_l:
            ds = cfgs.get("vqa2-llava.yaml", {}).get("dataset", {})
            return _load_vqa2_val_from_fs(
                max_samples, ds.get("image_dir",""), ds.get("questions_json",""), ds.get("annotations_json","")
            )

        if "audio" in name_l:
            ds = cfgs.get("audiocaps.yaml", {}).get("dataset", {})
            return _load_audiocaps_val_from_fs(
                max_samples, ds.get("captions_csv",""), ds.get("audio_dir", None)
            )

    except Exception as e:
        if allow_synth:
            print(f"[WARN] {name} unavailable ({e}); using synthetic samples.")
            return _make_synthetic(max_samples, task)
        raise

    # default synthetic only if explicitly allowed
    if allow_synth:
        print("[WARN] Unknown dataset; using synthetic samples.")
        return _make_synthetic(max_samples, task)
    raise RuntimeError(f"Dataset {name} not available and allow_synthetic=False")
