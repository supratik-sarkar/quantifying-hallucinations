# Compact PASS/DRIFT assessor across all datasets/models
# Usage: python tools/check_targets.py
import os, json, numpy as np, pandas as pd

ROOT="mllm-hallucination/outputs"
DS = ["coco_clip","vqa2_llava","audiocaps"]
MODELS = ["clip_whisper_t5","blip_clip_whisper","siglip_whisper_t5"]

targets = {
    "coco":     {"Entropy": (0.81,0.79), "MaxProb": (0.82,0.81), "Margin": (0.83,0.82), "OURS": (0.86,0.84)},
    "vqa2":     {"Entropy": (0.78,0.75), "MaxProb": (0.80,0.77), "Margin": (0.81,0.78), "OURS": (0.84,0.81)},
    "audiocaps":{"Entropy": (0.74,0.70), "MaxProb": (0.76,0.72), "Margin": (0.77,0.74), "OURS": (0.80,0.77)},
}
def key_of(ds):
    return "coco" if ds.startswith("coco") else ("vqa2" if ds.startswith("vqa2") else "audiocaps")

def main():
    rows=[]
    for ds in DS:
        tfile = os.path.join(ROOT, ds, "targets_check.json")
        if not os.path.exists(tfile): 
            continue
        rep = json.load(open(tfile))
        for m, d in rep.items():
            for alg, vals in d.items():
                tgt = targets[key_of(ds)][alg]
                au = vals["AUROC"]["val"]; ap = vals["AUPRC"]["val"]
                rows.append({
                    "dataset": ds, "model": m, "alg": alg,
                    "AUROC": au, "AUPRC": ap,
                    "Δ AUROC": round(au - tgt[0], 3),
                    "Δ AUPRC": round(ap - tgt[1], 3),
                    "PASS(AUROC)": vals["AUROC"]["status"],
                    "PASS(AUPRC)": vals["AUPRC"]["status"],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No results yet.")
        return
    avg_df = df.groupby(["dataset","alg"], as_index=False)[["AUROC","AUPRC","Δ AUROC","Δ AUPRC"]].mean()
    print("\n=== Averaged across models ===")
    print(avg_df.sort_values(["dataset","alg"]).to_string(index=False))
    print("\n=== Full detail ===")
    print(df.sort_values(["dataset","model","alg"]).to_string(index=False))

if __name__ == "__main__":
    main()
