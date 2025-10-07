import os, sys, json, math
from pathlib import Path

DATASETS = ["coco_clip", "vqa2_llava", "audiocaps"]
MODELS = ["clip_whisper_t5", "blip_clip_whisper", "siglip_whisper_t5"]
ALG_ORDER = ["Entropy", "MaxProb", "Margin", "OURS"]

def _fmt(x, nd=2):
    if x is None: return "0.00"
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf): return "0.00"
        return f"{xf:.{nd}f}"
    except Exception:
        return "0.00"

def _get(d, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d

def load_metrics(ds_dir: Path):
    acc = {alg: {"AUROC": [], "AUPRC": []} for alg in ALG_ORDER}
    for m in MODELS:
        mdir = ds_dir / m
        f = mdir / "metrics.json"
        if not f.exists(): continue
        try:
            mj = json.loads(f.read_text())
        except Exception:
            continue
        for alg in ALG_ORDER:
            au = _get(mj, alg, "AUROC")
            ap = _get(mj, alg, "AUPRC")
            if au is not None: acc[alg]["AUROC"].append(float(au))
            if ap is not None: acc[alg]["AUPRC"].append(float(ap))
    out = {}
    for alg in ALG_ORDER:
        aus, aps = acc[alg]["AUROC"], acc[alg]["AUPRC"]
        au = sum(aus)/len(aus) if aus else float("nan")
        ap = sum(aps)/len(aps) if aps else float("nan")
        out[alg] = (au, ap)
    return out

def load_energy(ds_dir: Path):
    stats = {}
    for m in MODELS:
        mdir = ds_dir / m
        efile, tfile = mdir / "energies.json", mdir / "throughput.json"
        if not efile.exists() or not tfile.exists(): continue
        try:
            e = json.loads(efile.read_text())
            t = json.loads(tfile.read_text())
            stats[m] = (_get(e,"median",default=float("nan")),
                        _get(e,"lo",default=float("nan")),
                        _get(e,"hi",default=float("nan")),
                        _get(t,"ex_per_s",default=float("nan")))
        except Exception:
            pass
    return stats

def main():
    out_root = Path("outputs")
    ds_map = {"coco_clip":"COCO", "vqa2_llava":"VQAv2", "audiocaps":"AudioCaps"}

    ds_metrics = {}
    for ds in DATASETS:
        d = out_root / ds
        if d.is_dir(): ds_metrics[ds] = load_metrics(d)

    def cell(ds, alg):
        au, ap = ds_metrics.get(ds, {}).get(alg, (float("nan"), float("nan")))
        return f"{_fmt(au)} / {_fmt(ap)}"

    def avg_cell(alg):
        vals = [(au, ap) for ds in DATASETS for (au, ap) in [ds_metrics.get(ds, {}).get(alg, (None, None))] if au is not None and ap is not None]
        if not vals: return "0.00 / 0.00"
        au = sum(v[0] for v in vals)/len(vals)
        ap = sum(v[1] for v in vals)/len(vals)
        return f"{_fmt(au)} / {_fmt(ap)}"

    print("% --------- (a) Detection (AUROC / AUPRC) ---------")
    print("\\begin{subtable}{\\columnwidth}")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Algorithm} & \\multicolumn{1}{c}{COCO} & \\multicolumn{1}{c}{VQAv2} & \\multicolumn{1}{c}{AudioCaps} & \\multicolumn{1}{c}{Avg.} \\\\")
    print(" & AUROC / AUPRC & AUROC / AUPRC & AUROC / AUPRC & AUROC / AUPRC \\\\")
    print("\\midrule")
    for alg in ALG_ORDER:
        label = r"$d_{\mathrm{sem}}^{(\varepsilon,h)}$ (ours)" if alg=="OURS" else alg
        c1, c2, c3, c4 = cell("coco_clip", alg), cell("vqa2_llava", alg), cell("audiocaps", alg), avg_cell(alg)
        if alg == "OURS":
            c1 = "\\textbf{" + c1 + "}"
            c2 = "\\textbf{" + c2 + "}"
            c3 = "\\textbf{" + c3 + "}"
            c4 = "\\textbf{" + c4 + "}"
        print(f"{label} & {c1} & {c2} & {c3} & {c4} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{subtable}\n")
    print("\\vspace{3em}\n")

    # ---------- (b) Energy/Runtime ----------
    print("% ---------- (b) Energy/Runtime ----------")
    print("\\begin{subtable}{\\columnwidth}")
    print("\\centering")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{Model} & \\multicolumn{1}{c}{COCO} & \\multicolumn{1}{c}{VQAv2} & \\multicolumn{1}{c}{AudioCaps} & \\multicolumn{1}{c}{Avg.} & \\multicolumn{1}{c}{Throughput$\\uparrow$} & \\multicolumn{1}{c}{Asymp.} \\\\")
    print(" & median (lo / hi) & median (lo / hi) & median (lo / hi) & median & ex/s &  \\\\")
    print("\\midrule")

    ds_energy = {ds: load_energy(out_root / ds) for ds in DATASETS if (out_root / ds).is_dir()}
    MODEL_LABELS = {
        "clip_whisper_t5": "CLIP+Whisper+T5",
        "blip_clip_whisper": "BLIP+CLIP+Whisper",
        "siglip_whisper_t5": "SigLIP+Whisper+T5",
    }

    for m in MODELS:
        label = MODEL_LABELS.get(m, m)
        cells = []
        meds = []
        thr = []
        for ds in DATASETS:
            e = ds_energy.get(ds, {}).get(m)
            if e is None:
                cells.append("---")
            else:
                med, lo, hi, t = e
                cells.append(f"{_fmt(med)} \\;({_fmt(lo)} / {_fmt(hi)})")
                meds.append(med); thr.append(t)
        avg_med = _fmt(sum(meds)/len(meds)) if meds else "0.00"
        thr_show = f"\\textbf{{{_fmt(max(thr) if thr else float('nan'), nd=0)}}}"
        asymp = "$O(|E| + N\\log k + m d)$"
        # If BLIP has no AudioCaps, print '---' there (already handled).
        print(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & {avg_med} & {thr_show} & {asymp} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}% end resizebox")
    print("\\end{subtable}")
    print("\\end{table}")
if __name__ == '__main__':
    main()
