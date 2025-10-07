import os, sys, json
from pathlib import Path

def load_results_json(ds_dir: Path):
    p = ds_dir / "results.json"
    if p.exists():
        return json.loads(p.read_text())
    # reconstruct if missing
    models = {}
    for m in ["clip_whisper_t5","blip_clip_whisper","siglip_whisper_t5"]:
        mdir = ds_dir / m
        if not mdir.is_dir():
            continue
        def _try(f):
            fp = mdir / f
            return json.loads(fp.read_text()) if fp.exists() else {}
        models[m] = {
            "metrics": _try("metrics.json"),
            "energies": _try("energies.json"),
            "throughput": _try("throughput.json"),
        }
    return {"dataset_tag": ds_dir.name, "models": models}

def resolve_dir(out_dir: str, tag: str) -> Path:
    p = Path(out_dir)
    if (p / "results.json").exists():
        return p
    # try common subdir names
    for name in [tag, f"{tag}_clip", f"{tag}_llava", "coco_clip","vqa2_llava","audiocaps"]:
        q = p / name
        if (q / "results.json").exists() or any((q / m / "metrics.json").exists() for m in ["clip_whisper_t5","blip_clip_whisper","siglip_whisper_t5"]):
            return q
    # last guess: if out_dir already looks like a dataset dir, use it
    if any((p / m / "metrics.json").exists() for m in ["clip_whisper_t5","blip_clip_whisper","siglip_whisper_t5"]):
        return p
    raise FileNotFoundError(f"Could not locate dataset results under {out_dir} (tag={tag})")

def main(out_dir: str, tag: str):
    ds_dir = resolve_dir(out_dir, tag)
    res = load_results_json(ds_dir)

    # Minimal text output; also write a concise summary JSON
    models = list(res.get("models", {}).keys())
    print(f"[REPORT] dataset={res.get('dataset_tag', ds_dir.name)} models={models}")
    summ = {
        "dataset": res.get("dataset_tag", ds_dir.name),
        "dir": str(ds_dir),
        "models": models,
        "n_samples": res.get("n_samples", None),
    }
    (ds_dir / "report_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"[OK] Wrote {ds_dir}/report_summary.json")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.entrypoints.export_report <out_dir or dataset_dir> <tag>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
