# Submission (Sep-2025) — Submission Snapshot

This release archives the exact code/assets corresponding to the initial submission dated **2025-09-25**.
Minor differences from the paper tables may occur due to stochastic inference, library/driver drift,
model snapshot revisions, and loader preprocessing. See README “Why numbers may differ”.

**Contents**
- Code snapshot
- Replication notebooks (Colab/GCP)
- Diagnostics script: `python tools/check_targets.py`
- Seeding helper: `scripts/repro_seed.sh`

**Reproduction notes**
- Prefer running `scripts/repro_seed.sh` (or copy its flags) before main scripts.
- Pin model revisions (e.g., HF commit hashes) when possible.
- Hardware used originally: **Databricks A100**. Colab/GCP previews are provided as a fallback.
