# MLLM Hallucination — Spectral & KL-Smoothed Framework (Colab)
Reproducible Colab pipeline implementing Algorithm (KL-Smoothed Multimodal Hallucination)
with hypergraph Laplacians, diffusion kernels, spectral CF-bounds, and KV-calibration.

- 3 datasets × 3 multimodal model configs
- 9 CF-bound 3D heatmaps (temperature × diffusion time)
- Ablations over ε, h, τ, μ (and baseline  comparisons)
- Fully runnable on Colab A100, no private tokens.



---
## Project map & theory cheat-sheet
_Generated: 2025-09-23 11:17_

This section gives a quick map from code to the theoretical objects used in the paper (inline LaTeX uses \( ... \)). It also includes a depth-limited tree of the repo.

### Repository overview

```
mllm-hallucination/
  ├── configs/
    └── audiocaps.yaml
    └── coco-clip.yaml
    └── data_paths_example.yaml
    └── default.yaml
    └── pope-llava.yaml
    └── vqa2-llava.yaml
  ├── notebooks/
  ├── scripts/
    └── prepare_data.py
  ├── src/
    └── entrypoints/
      └── build_tables.py
      └── export_report.py
      └── run_pipeline.py
    └── eval/
      └── baselines.py
      └── metrics.py
      └── tables.py
    └── io/
      └── adapters.py
      └── datamodules.py
    └── models/
      └── clip_embed.py
      └── llava_mm.py
      └── llm_text.py
      └── logits_api.py
      └── siglip_embed.py
    └── theory/
      └── calibration.py
      └── contrast.py
      └── diffusion.py
      └── energy.py
      └── hypergraph.py
      └── k_selector.py
      └── kernel_smoother.py
      └── laplacian.py
      └── score_semantic.py
      └── smoothing.py
    └── utils/
      └── logging.py
      └── seed.py
  ├── tests/
    └── test_energy.py
    └── test_graph.py
    └── test_scores.py
  ├── .gitignore
  ├── CITATION.cff
  ├── LICENSE
  ├── pyproject.toml
  ├── README.md
  ├── setup.cfg
```

### Theory glossary (files → quantities)

| File | Quantity / Symbol | Short description |
|---|---|---|
| `src/theory/smoothing.py` | \(\tilde{f}_p = (1-\varepsilon) f_p + \varepsilon \rho\) | Mixture smoothing of densities with weight \( \varepsilon \). |
| `src/theory/score_semantic.py` | \(d_{\mathrm{sem}}^{(\varepsilon,h)}(x)\) | Pointwise semantic gap using smoothed kernel density at bandwidth \(h\). |
| `src/theory/k_selector.py` | \(\Pi_{\mathcal{K}}\) | Selector mapping a node to its representative in knowledge set \( \mathcal{K} \). |
| `src/theory/hypergraph.py` | \(w_{T,t}(e)\) | Hyperedge weights parameterized by diffusion time \(t\) and temperature \(T\). |
| `src/theory/laplacian.py` | \(\mathcal{L}_{\text{norm}}, \lambda_{2}\) | Normalized hypergraph Laplacian and spectral quantities. |
| `src/theory/diffusion.py` | \(K_t = e^{-t\mathcal{L}}\) | Diffusion kernel and semantic diffusion operator. |
| `src/theory/contrast.py` | \(c(u,v)\) | Contrast vector for a node pair used in energy gaps. |
| `src/theory/energy.py` | \(\Delta \mathcal{E}_{\tau}\) | Spectral energy gap bounds over threshold \( \tau \). |
| `src/theory/calibration.py` | \(\hat{m},\ k\!v(\tau)\) | Good–Turing missing mass and calibration schedule. |
| `src/theory/kernel_smoother.py` | \(k_h(x,y),\ \mathbf{T}_h\) | Gaussian kernel at scale \(h\); row-stochastic transition \( \mathbf{T}_h \). |

### Key modules (implementation map)

| File | API | What it does |
|---|---|---|
| `src/models/clip_embed.py` | `CLIPWrapper` | Image embedding via CLIP. |
| `src/models/siglip_embed.py` | `SigLIPWrapper` | Image embedding via SigLIP. |
| `src/models/llm_text.py` | `TextBackbone` | Text encoder for captions/questions. |
| `src/models/logits_api.py` | `SurrogateBoltzmann` | Energy→probabilities/logits for baselines. |
| `src/io/datamodules.py` | `try_load(...)` | Unified dataset loader (COCO/VQAv2/AudioCaps). |
| `src/io/adapters.py` | `collate_*` | Batching & field selection for tasks. |
| `src/utils/seed.py` | `seed_everything` | Reproducibility helpers. |
| `src/utils/logging.py` | `time_block` | Lightweight timing scope logger. |
| `src/entrypoints/run_pipeline.py` | `main(cfg)` | End-to-end runner producing metrics & reports. |
| `src/entrypoints/export_report.py` | `main(out,tag)` | Converts results.json to figures/tables. |


---
## File-type distribution
_Generated: 2025-09-23 11:17_

| Extension | Count | Share |
|---|---:|---:|
| `.py` | 29 | 70.7% |
| `.yaml` | 6 | 14.6% |
| `(no ext)` | 2 | 4.9% |
| `.cff` | 1 | 2.4% |
| `.cfg` | 1 | 2.4% |
| `.md` | 1 | 2.4% |
| `.toml` | 1 | 2.4% |

**Total files scanned:** 41


## Animated τ-decay & h-ablation (Matplotlib GIFs)
The surfaces below show how the multimodal hallucination energy evolves with time-scale $\tau$ and varies with the bandwidth $h$ from Theorem&nbsp;1.

![audiocaps_clip_whisper_t5 τ-decay](outputs/anim_mpl/audiocaps_clip_whisper_t5_tau_mpl.gif) ![audiocaps_clip_whisper_t5 h-ablation](outputs/anim_mpl/audiocaps_clip_whisper_t5_h_mpl.gif)

![coco-clip_clip_whisper_t5 τ-decay](outputs/anim_mpl/coco-clip_clip_whisper_t5_tau_mpl.gif) ![coco-clip_clip_whisper_t5 h-ablation](outputs/anim_mpl/coco-clip_clip_whisper_t5_h_mpl.gif)

![vqa2-llava_blip_clip_whisper τ-decay](outputs/anim_mpl/vqa2-llava_blip_clip_whisper_tau_mpl.gif) ![vqa2-llava_blip_clip_whisper h-ablation](outputs/anim_mpl/vqa2-llava_blip_clip_whisper_h_mpl.gif)

<!-- SUBMISSION_SNAPSHOT_START -->
## Submission snapshot

The exact code/assets used at submission time are archived in the GitHub Release
**“Submission (Sep-2025)”**. The default branch may contain replication
notebooks and small fixes for future runs.

> To audit the submission state, download the release tarball or see `RELEASE_NOTES_2025-09.md`.
<!-- SUBMISSION_SNAPSHOT_END -->

<!-- WHY_NUMBERS_DIFFER_START -->
## Why numbers may differ

Small AUROC/AUPRC deltas vs. the paper table are expected due to:
- **Stochastic inference** (sampling, CUDA nondeterminism) even with fixed seeds.
- **Library/driver drift** across machines A100 vs Colab T4/L4 (PyTorch/CUDA/cuDNN/FAISS kernels).
- **Model snapshot drift** (HF model revisions or API-served models updated upstream).
- **Dataset loaders** (shuffle order, minor preprocessing differences).

**Mitigations:** version pinning, fixed seeds, frozen model revisions, and a
drift audit script:
```bash
python tools/check_targets.py
```
For more determinism:
```bash
bash scripts/repro_seed.sh
```
Residual ±(1–3)pp variations are typical for this stack.
<!-- WHY_NUMBERS_DIFFER_END -->

<!-- DIAGNOSTICS_START -->
## Diagnostics

Compare current runs against paper targets:
```bash
python tools/check_targets.py
```
The script prints an averaged table and full per-dataset/model breakdown.
<!-- DIAGNOSTICS_END -->

<!-- COMPUTE_ENVS_START -->
## Compute environments

- **Primary (paper):** Databricks (A100) in an internal enterprise workspace.
- **Replication:** For reproducibility we provide Colab/GCP notebooks and a CPU/GPU fallback pipeline.  
  Minor numeric differences across plots/tables may occur across hardware & library versions. 
  We include environment pins and seeds, plus targets vs. current diagnostics (```tools/check_targets.py```)
<!-- COMPUTE_ENVS_END -->

<!-- WHY_NUMBERS_DIFFER_START -->
## Why numbers may differ

Small AUROC/AUPRC deltas vs. the paper table are expected due to:
- **Stochastic inference** (sampling, CUDA nondeterminism) even with fixed seeds.
- **Library/driver drift** across machines A100 vs Colab T4/L4 (PyTorch/CUDA/cuDNN/FAISS kernels).
- **Model snapshot drift** (HF model revisions or API-served models updated upstream).
- **Dataset loaders** (shuffle order, minor preprocessing differences).

**Mitigations:** version pinning, fixed seeds, frozen model revisions, and a
drift audit script:
```bash
python tools/check_targets.py
```
For more determinism:
```bash
bash scripts/repro_seed.sh
```
Residual ±(1–3)pp variations are typical for this stack.
<!-- WHY_NUMBERS_DIFFER_END -->
