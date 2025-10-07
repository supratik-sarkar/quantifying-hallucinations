#!/usr/bin/env python
import argparse, os, json
from pathlib import Path

def _w(p): p.parent.mkdir(parents=True, exist_ok=True); return p

def write_paths_yaml(out_path, coco_dir, vqa_dir, ac_dir):
    txt = f"""# Auto-generated local data paths
dataset:
  coco_dir: {coco_dir}
  vqa2_dir: {vqa_dir}
  audiocaps_dir: {ac_dir}
"""
    _w(Path(out_path)).write_text(txt, encoding="utf-8")

def build_sanity_sets(root):
    root = Path(root)
    (root/"sanity"/"coco_captions").mkdir(parents=True, exist_ok=True)
    (root/"sanity"/"vqa2").mkdir(parents=True, exist_ok=True)
    (root/"sanity"/"audiocaps").mkdir(parents=True, exist_ok=True)
    for p in [
        root/"sanity"/"coco_captions"/"val.jsonl",
        root/"sanity"/"vqa2"/"val.jsonl",
        root/"sanity"/"audiocaps"/"val.jsonl",
    ]:
        if not p.exists():
            p.write_text(json.dumps({"stub": True})+"\n", encoding="utf-8")
    return {
        "coco": str(root/"sanity"/"coco_captions"),
        "vqa2": str(root/"sanity"/"vqa2"),
        "ac":   str(root/"sanity"/"audiocaps"),
    }

def cache_full_sets(root):
    from datasets import load_dataset
    root = Path(root)
    cache_dir = root/"hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # partial validation splits (fast)
    _ = load_dataset("coco_captions", "2017", split="validation[:500]", cache_dir=str(cache_dir))
    _ = load_dataset("HuggingFaceM4/VQAv2", split="validation[:500]", cache_dir=str(cache_dir))
    _ = load_dataset("audiocaps", split="validation[:500]", cache_dir=str(cache_dir))
    return {
        "coco": str(root/"coco_captions"),
        "vqa2": str(root/"vqa2"),
        "ac":   str(root/"audiocaps"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--cfg-out", default="configs/data_paths_local.yaml")
    args = ap.parse_args()

    try:
        paths = cache_full_sets(args.root) if args.full else build_sanity_sets(args.root)
    except Exception as e:
        print("[WARN] Falling back to sanity sets:", e)
        paths = build_sanity_sets(args.root)

    write_paths_yaml(args.cfg_out, paths["coco"], paths["vqa2"], paths["ac"])
    print("[OK] Data prepared. Paths:", args.cfg_out)

if __name__ == "__main__":
    main()
