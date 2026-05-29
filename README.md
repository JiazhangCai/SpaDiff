# SpaDiff

**Mass-particle reverse denoising for spatial transcriptomics**

SpaDiff denoises sequencing-based spatial transcriptomics data (e.g. 10x Visium) by modelling spot-swapping contamination as a reverse diffusion process. UMI counts are treated as mass particles that move along a kernel-density score field, progressively sharpening spatial gene-expression patterns while **strictly preserving total mass** across the tissue.

This repository contains the reference implementation accompanying:

> Cai J. *et al.* "**[Paper title — fill in]**". *[Journal]*, 2026. DOI: [10.xxxx/xxxxx](https://doi.org/10.xxxx/xxxxx)

---

## Why mass-conservation?

Unlike per-spot imputation methods, SpaDiff moves mass *across* spots. This makes it uniquely capable of denoising off-tissue contamination (transporting stray UMIs back into the tissue boundary) — but it also means the total UMI count of every gene is conserved by construction.

Key features:

- **Mass-conserving** — total UMI count per gene is preserved exactly.
- **Boundary-aware** — ghost-spot symmetrisation prevents off-tissue mass loss.
- **Density-aware suppression** — `α_ρ` mixing prevents mode-collapse in sparse genes.
- **Pure NumPy + SciPy** — no GPU required, scales to ~5,000 spots × ~10,000 genes on a laptop in a few minutes.
- **Parallel by gene block** — uses Python multiprocessing for the per-gene update phase.

---

## Installation

Clone the repository and install with `pip`:

```bash
git clone https://github.com/JiazhangCai/SpaDiff.git
cd spadiff
pip install -e .
```

Or, to also install the dependencies needed to run the example notebook:

```bash
pip install -e ".[examples]"
```

Requirements: Python ≥ 3.9, NumPy ≥ 1.24, SciPy ≥ 1.10.

---

## Quick start

```python
import numpy as np
from scipy import sparse
from spadiff import denoise

# spot_coords: (M, 2) array of spot xy-coordinates (any scale)
# counts:      (M, G) sparse CSR matrix of integer UMI counts
spot_coords = np.load("my_spot_coords.npy")
counts      = sparse.load_npz("my_counts.npz")

result = denoise(
    spot_coords,
    counts,
    T=15,                       # number of diffusion steps
    eta_base=0.005,             # step size
    H_START=0.20, H_END=0.03,   # KDE bandwidth schedule
    L_score_global=5,           # KNN layers for global score field
    L_score_gene=3,             # KNN layers for per-gene score field
    n_procs=4,                  # parallel workers
)

denoised_counts = result["counts_denoised"]   # sparse CSR, same shape as input
```

For a complete end-to-end walkthrough on synthetic data, see [`examples/minimal_demo.ipynb`](examples/minimal_demo.ipynb).

---

## Reproducing the paper

This release contains the SpaDiff package and a minimal worked example. The full set of analysis scripts (Figures 2–6, ablations, sweeps) and the denoised output matrices are too large to host on GitHub. They are archived on Zenodo:

- **Code archive (this repo)**: [doi:10.5281/zenodo.20437584](https://doi.org/10.5281/zenodo.20437584)
- **Paper data and denoised outputs**: [doi:10.5281/zenodo.YYYYYYY](https://doi.org/10.5281/zenodo.YYYYYYY) *(optional second record)*

Raw input data:

- **Chimeric human-mouse Visium (Figure 5)**: [GEO accession — fill in]
- **DLPFC (Figure 4)**: [SpatialLIBD](https://research.libd.org/spatialLIBD/), sample 151507
- **Breast / colorectal cancer (Figure 6)**: [10x Genomics public datasets — fill in]

---

## Parameter tuning

See [`docs/tuning_notes.md`](docs/tuning_notes.md) for a practical guide based on chimeric-Visium experiments. It documents the two operating regimes (quantification vs. mass-transport), recommended starting configs, and common failure modes.

---

## Repository layout

```
spadiff/
├── spadiff/                  # the SpaDiff Python package
│   ├── __init__.py           # exposes denoise()
│   ├── utils.py              # high-level denoise() API
│   └── utils_dev.py          # core algorithm implementation
├── examples/
│   └── minimal_demo.ipynb    # synthetic-data quick start
├── docs/
│   └── tuning_notes.md       # parameter tuning guide
├── pyproject.toml
├── requirements.txt
├── LICENSE                   # MIT
├── CITATION.cff
└── README.md
```

---

## Citation

If you use SpaDiff in your research, please cite both the paper and this code release:

```bibtex
@article{cai2026spadiff,
  title   = {SpaDiff: Denoising for Sequence-based Spatial Transcriptomics via Diffusion Process},
  author  = {Cai, Jiazhang and others},
  journal = {REPLACE WITH JOURNAL},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}

@software{cai2026spadiff_code,
  title     = {SpaDiff: Mass-particle reverse denoising for spatial transcriptomics (v1.0.0)},
  author    = {Cai, Jiazhang},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20437584},
  url       = {https://doi.org/10.5281/zenodo.20437584}
}
```

---

## License

[MIT](LICENSE) © 2026 Jiazhang Cai

---

## Contact

Issues and questions: please open a [GitHub issue](https://github.com/JiazhangCai/SpaDiff/issues).

Author: Jiazhang Cai &lt;caijiazhang1@gmail.com&gt;
