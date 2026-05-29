# SpaDiff Parameter Tuning — Lessons from Chimeric Experiments (v6–v22)

A practical reference for applying SpaDiff to new spatial transcriptomics datasets.
Written after iterating on a 4992-spot human–mouse chimeric Visium dataset with
2,373 human genes + 7,313 mouse genes. Captures what parameters do, how to tune
them, common failure modes, and recommended starting points.

---

## 1. Problem context

SpaDiff denoises sequencing-based ST data by modeling spot-swapping as a
reverse diffusion process. It moves "mass particles" (UMI counts) along a
gradient of a kernel-density score field toward biologically correct
locations while strictly preserving total counts.

Key design property: **mass conservation**. Unlike per-spot imputation methods,
SpaDiff moves mass across spots. This makes it uniquely capable of denoising
off-tissue contamination (physical transport back into tissue) — but also
means aggressive settings can redistribute mass in ways that look visually
compelling but alter per-spot quantification.

---

## 2. Two operating regimes (found empirically)

Depending on the downstream use, tune SpaDiff for one of two goals:

### Regime A — Quantification-preservation (`L_gene = 9`)
- Goal: maximize per-gene / per-spot expression fidelity
- Use for: method comparison benchmarks, per-spot clustering, DE analysis
- Metric profile: highest purity, lowest cross-species contamination
- Visual: moderate pattern sharpening, most per-spot counts preserved

### Regime B — Mass-transport / pattern-recovery (`L_gene = 18`)
- Goal: maximize off-tissue cleanup and spatial morphology recovery
- Use for: figures showcasing spatial structure, off-tissue denoising
- Metric profile: highest MRec, lowest `off_tissue_mass_frac`
- Visual: crisp stripe/band patterns matching ground-truth morphology
- Caveat: per-spot classification drops because mass is aggressively relocated

One configuration can't optimize both. Decide upfront which downstream task
matters more for your paper.

---

## 3. Final recommended config (chimeric)

```python
T = 15
eta_sched   = np.full(T, 0.010, dtype=np.float32)   # constant, no decay
sigma_sched = np.zeros(T,       dtype=np.float32)   # deterministic (no noise)
h_sched     = np.full(T, 0.16,  dtype=np.float32)   # constant bandwidth

# Density-aware suppression
pct               = 90          # alpha_rho rho_ref percentile
alpha_rho_mode    = 'linear'
rho_ref_method    = 'p95'
nonzero_rho_ref   = True
gene_score_on_spots = False

# Score mixing
alpha_mode = 'beta'
beta       = 0.02               # keep small for species-asymmetric data

# Neighbourhood layer counts (determine KNN K values)
L_assign        = 1             # K=7   Phase A deposition
L_interp        = 1             # K=7   spot -> particle interpolation
L_score_global  = 5             # K=91  Phase B global score
L_score_gene    = 9             # K=271  (or 18 for mass-transport regime)

# Kernel scale
bandwidth_scale = 0.04
tau_mass        = 0.0           # low-mass gene gating threshold (0 = off)
lam             = 1e-3

# Boundary
use_ghost_spots = True
K_boundary      = 7
n_layers_ghost  = 1

# Infra
n_procs      = 8
random_state = 42
```

---

## 4. Parameter reference & how each one moves the needle

### 4.1 Schedule parameters (`eta`, `sigma`, `h`)

| param | role | effect | recommendation |
|---|---|---|---|
| `eta` | Euler-Maruyama step size | Too large → particles overshoot, off-tissue particles go anywhere. Too small → mass doesn't flow. | Start with constant 0.010. For new data, sweep {0.005, 0.010, 0.020}. Must be **monotonically decreasing or constant** per physics assumption — never ramp up. |
| `sigma` | noise amplitude (multiplies gradient) | Single-step noise magnitude is `eta * sigma * sqrt(2)` ≈ step-spacings. If > 1 spot spacing, step-1 blur permanently damages pattern. | **Default to 0.** Only add small sigma (0.02–0.05) if you observe particles stuck in local minima. |
| `h` | KDE bandwidth (Gaussian) | Determines gradient direction quality. Controls trade-off between local-noise robustness (large h) and spatial resolution (small h). | **Use constant** h = 0.16 in unit-box coordinates (~11× spot spacing). Do NOT use exponential decay from 0.20 — that caused step-1 blur in legacy configs. |

### 4.2 Neighbourhood layer counts (`L_*`)

These are the critical knobs. They map to KNN truncation via `K = 3L(L+1) + 1`
(hex layer formula).

| L | K | reach (spots) | 
|---|---|---|
| 1 | 7   | 1.5  |
| 3 | 37  | 3.4  |
| 5 | 91  | 5.4  |
| 9 | 271 | 9.3  |
| 12 | 469 | 12.2 |
| 15 | 721 | 15.2 |
| 18 | 1027 | 18.1 |
| 20 | 1261 | 20.0 |

**Reach is a HARD cutoff, not a Gaussian tail.** Queries use KNN with fixed K;
spots outside top-K contribute exactly 0 regardless of `h`. This was a common
misunderstanding — bandwidth `h` controls weighting *within* K neighbours, not
how far particles can "see".

| param | role | direction |
|---|---|---|
| `L_assign` | Phase A soft deposition KNN | Keep at 1 (K=7). Larger = mass spreads more in single deposition step. |
| `L_interp` | Spot→particle score interpolation KNN | Keep at 1. Larger = particles feel spot-level scores from farther away. |
| `L_score_global` | Global score KDE KNN | When `beta` is small, this is **dead** (alpha ≈ 1, global contributes 0). Keep at 5 by default. |
| `L_score_gene` | Gene-specific score KDE KNN | **Primary knob.** Controls both (a) how far off-tissue mass can be pulled back toward correct gene density and (b) how aggressive pattern concentration is. |

**L_score_gene sweep behavior** (on chimeric):

| L_gene | K | MRec | purity | cls_acc | off_tissue% | Notes |
|---|---|---|---|---|---|---|
| 3  | 37   | 22%  | 0.72 | 79% | 20% | v16 baseline — low transport, low denoising |
| 9  | 271  | 39%  | **0.82** | **88%** | 15% | Sweet spot for quantification |
| 12 | 469  | 46%  | 0.81 | 87% | 14% | Acceptable |
| 15 | 721  | 56%  | 0.77 | 81% | 11% | Over-correction starting |
| 18 | 1027 | 62%  | 0.71 | 76% | 10% | Mass-transport regime (visual showcase) |
| 20 | 1261 | 65%  | 0.65 | 72% | 9%  | Contamination self-amplification: mass gets pulled to noise contamination spots and amplifies — classification reverses to below raw |

### 4.3 Alpha mixing (`alpha_mode`, `beta`)

`s_mix = alpha * s_gene + (1 - alpha) * s_global`, with
`alpha = rho_gene / (rho_gene + beta)` when `alpha_mode='beta'`.

| beta | behavior |
|---|---|
| Small (0.02) | alpha ≈ 1 anywhere gene has mass → **pure gene-specific score** → L_score_global becomes irrelevant. Safe when species/subtypes have very asymmetric total mass (chimera: mouse >> human). |
| Large (>median rho_gene) | alpha scales with local gene density → global score contributes. Use when you want low-density genes to benefit from population-wide gradient. |

**For chimera-like asymmetric data, keep `beta = 0.02`** — otherwise global
score (dominated by higher-mass species) pulls minority-species mass into
majority-species tissue.

### 4.4 Density-aware suppression (`alpha_rho`, `pct`)

`alpha_rho = 1 - w_gene / rho_ref`, clipped to [0, 1]. High-density spots get
`alpha_rho ≈ 0` (freeze). Low-density spots get `alpha_rho ≈ 1` (full motion).

| pct | rho_ref | effect |
|---|---|---|
| 70 | 70th percentile of gene density | Aggressive freeze: many in-tissue spots treated as "already settled" |
| 90 | 90th percentile | Moderate freeze. **v22 default.** |
| 99 | 99th percentile | Minimal freeze, almost all particles move freely |

**Without alpha_rho, positive feedback causes mode collapse** — once a spot
is slightly denser it attracts more mass, collapsing everything into one peak.

### 4.5 Other parameters (usually leave at defaults)

- `bandwidth_scale = 0.04`: scales h for Phase A/interpolation
- `tau_mass = 0.0`: low-mass gene gating threshold. Set to ~10 for very sparse
  gene lists to skip score computation on near-empty genes
- `lam = 1e-3`: density denominator regularization. Almost never needs changing
- `use_ghost_spots = True`: critical for tissue boundary handling. Keep on
- `K_boundary = 7`: boundary detection; default is fine
- `T = 15`: diffusion steps. Increase if MRec is limited by total step budget
  (`T × eta` is cumulative displacement capacity in unit-box)

---

## 5. Diagnostic metrics — what they mean and which ones to trust

### 5.1 True correctness metrics (trust these)

| metric | definition | good direction |
|---|---|---|
| `classification_accuracy` | per-spot species prediction accuracy vs ground truth | ↑ |
| `species_auc` | ROC AUC of species prediction | ↑ (but watch small drops — usually OK) |
| `purity` | mean per-gene fraction of mass localized in correct tissue | ↑ |
| `cscr` (cross-species contamination rate) | avg of mouse-mass-in-human-spot and human-mass-in-mouse-spot | ↓ |
| `mass_recovery_rate` (MRec) | fraction of originally-off-tissue mass now in-tissue | ↑ |
| `off_tissue_mass_frac` | total off-tissue mass / total mass (post-denoise) | ↓ |

### 5.2 Misleading metrics (don't trust as correctness)

| metric | definition | why misleading |
|---|---|---|
| `human_pattern_corr`, `mouse_pattern_corr`, `total_pattern_corr` | `corr(raw_per_spot_sum, denoised_per_spot_sum)` per species | Measures **conservatism**, not correctness. HIGH = denoiser changed little. LOW = aggressive redistribution (could be good or bad). **Original (no denoising) trivially has pattern_corr=1.0.** |
| `mode_collapse_score` (Gini of final mass) | Wealth-inequality-style distribution metric | Doesn't distinguish "mass concentrated to correct biological stripe" (good) vs "mass collapsed to single point" (bad). **Use visual inspection of GIFs instead.** |

### 5.3 Diagnostic metrics (run-integrity)

| metric | interpretation |
|---|---|
| `step1_corr_step0` | Pearson corr of per-spot mass at step 0 vs step 1. Should be > 0.95 for well-tuned config. If < 0.85, your step-0 is catastrophically blurring the pattern. |
| `max_displacement_step0` | 95th percentile per-particle displacement in step 0 (unit-box). Compare to spot spacing `1/sqrt(M)`. If > 2× spot spacing, eta or sigma too aggressive. |
| `stability` | custom column: 'ok' / 'mass_violation' / 'nan_inf' / 'chaos'. Any non-'ok' means the run blew up. |

---

## 6. Tuning strategy for new data

### Step 0 — smoke test (verify pipeline)

Run once with the v22 recommended config above. Check:
- Runtime roughly matches scaling (~200s per step per K_gene=271 × T=15 on 4992 spots)
- `stability = 'ok'`
- Outputs loadable

### Step 1 — choose regime

Decide whether your paper cares more about:
- Per-spot quantification / downstream analysis → **L_gene = 9**
- Spatial morphology recovery / off-tissue showcase → **L_gene = 18**

If unsure, run both. Keep the mtx files for both (they're the anchors for
figures vs tables).

### Step 2 — validate key metrics

Compare against raw data:
- purity should increase (good denoising)
- cscr should decrease
- classification_accuracy should increase or stay comparable
- MRec positive and off_tissue_mass_frac decreases

If purity **decreases** or cscr **increases**, something is wrong:
- Check species asymmetry: is `beta` small enough?
- Check whether gene-score reach is over-ambitious (try smaller L_gene first)

### Step 3 — visual inspection of trajectory GIFs

Generate per-step GIFs of total/species mass. Look for:
- Step-0 → step-1 should not catastrophically blur pattern (sigma=0 guarantees this)
- Stripe-like concentration matching expected tissue morphology = good
- Random scatter into point-like clumps = bad (reduce L_gene)
- Off-tissue mass "zeroing out" or flowing into tissue = good
- Off-tissue local collapse (stuck in wrong location) = h too small OR L_gene too small

### Step 4 — hyperparameter sweep (if defaults don't work)

Priority order (most to least impactful):

1. **`L_score_gene`** — sweep {5, 9, 12, 18} (or narrower around a guess)
2. **`h`** — sweep {0.08, 0.16, 0.24}. Larger h = cleaner gradient direction but weaker magnitude
3. **`pct`** — sweep {80, 90, 99} if alpha_rho behavior seems off
4. **`eta`** — sweep {0.005, 0.010, 0.020}. Only touch after L_gene and h are set
5. **`sigma`** — add {0.02, 0.05} only if mass exploration seems stuck

**Do NOT sweep `L_score_global` unless `beta` is large.** When beta=0.02,
L_global is inert.

**Runtime cost scales as O(P · K_gene)** — gene score is the dominant phase.
K_gene=271 (L=9) is ~200s per config on 4992 spots × 9692 genes. K_gene=1027
(L=18) is ~1400s. Budget accordingly.

---

## 7. Known failure modes & fixes

| symptom | root cause | fix |
|---|---|---|
| Step-1 blur (step1_corr < 0.9) | sigma × sqrt(eta) noise overwhelms pattern, or H_START too large | `sigma = 0`, constant `h = 0.16` |
| Off-tissue particles collapse to local "fake" peaks | h too small — local noise dominates gradient direction | Increase `h` to 0.16+ |
| MRec stays near 0, mass doesn't move | `L_score_gene` too small (reach < distance to tissue) | Increase `L_score_gene` to 9+ |
| Mass-violation / NaN in history | eta × max(grad) exceeds stable step | Decrease `eta`, or reduce `L_score_gene` |
| Species mass asymmetry causes minority species to collapse into majority species tissue | `beta` too large — global score (dominated by majority) pulls minority particles | Set `beta = 0.02` |
| Purity decreases despite denoising | `L_gene` too large — gene-specific score self-amplifies contamination | Reduce `L_gene` to 9 |
| Pattern looks right in visual but per-spot classification drops | Over-concentration: mass moved to correct STRIPE but emptied "interstitial" tissue spots | Either accept trade-off (mass-transport regime) or reduce `L_gene` |

---

## 8. Reporting & paper figures

Based on the chimeric experience, these splits worked well:

- **Figure showing spatial pattern recovery**: use `L_gene = 18` (aggressive mass transport).
  Plot aggregate species expression (sum over all human genes, same for mouse) before vs
  after. Off-tissue reduction number goes in caption (e.g. 25.4% → 9.7%, 62% drop).

- **Figure showing per-gene in-tissue fraction**: paired boxplots (raw vs denoised).
  Split by all genes / species-specific genes for species-asymmetric data.
  Use Wilcoxon paired signed-rank test.

- **Benchmark table vs other methods**: use `L_gene = 9` (quantification preservation).
  Restrict everything to in-tissue spots (most imputation methods can't handle
  off-tissue). Report: classification_accuracy, species_auc, purity, cscr.
  **Do NOT report pattern_corr in benchmark tables** — it's trivially 1.0 for
  methods that change nothing and unfairly penalizes methods that denoise aggressively.

---

## 9. Quick-start checklist for a new dataset

```
1. Load data, confirm: spot count M, gene count G, meta with ground truth
2. Identify gene categories (species/cell-type) and any mass asymmetry
3. Set beta = 0.02 if asymmetric, else default
4. Run smoke test with v22 recommended config (L_gene=9)
5. Check metrics: purity, cscr, cls_acc vs raw. Expect improvements.
6. If MRec low, increase L_gene to 18 for mass-transport regime
7. Generate trajectory GIFs for visual sanity
8. If behavior is wrong, diagnostic sweep L_gene first, then h, then eta
9. Select two configs (one for each regime) for paper: L_gene=9 table, L_gene=18 figure
```

---

## 10. Paths & artifacts in this project

- Core SpaDiff code: `code/SpaDiff/utils_dev.py` (canonical), `code/SpaDiff/utils.py` (wrapper)
- Final experiment script: `scripts/spot_level_v22.py`
- Figure generation: `scripts/figure5_panels.py`, `scripts/figure5_panelC_split.py`
- Benchmark table: `scripts/benchmark_table.py`
- Manuscript updater: `scripts/update_manuscript_v22.py`
- Cleanup: `scripts/cleanup_chimeric_v22.sh`
- Denoised outputs used in paper:
  - `results/chimeric_4992_diagnostic_v22/Lg05_Le09_counts.mtx` (Table 1 source)
  - `results/chimeric_4992_diagnostic_v22/Lg05_Le18_counts.mtx` (Figure 5 B/C source)

---

## 11. Context prompt for new chat

Paste this into a new chat when starting on a new dataset:

> I'm working on denoising spatial transcriptomics data with SpaDiff
> (score-based reverse diffusion with mass-particle transport). I've
> previously tuned it on a 4992-spot human-mouse chimeric Visium dataset
> and have a comprehensive parameter guide at
> `SpaDiff_tuning_notes.md` in my project root. The final recommended
> config uses constant `eta=0.010`, `sigma=0`, `h=0.16`, `L_score_gene=9`
> (quantification-preservation) or `L_score_gene=18` (mass-transport),
> with `beta=0.02` for species-asymmetric data. Pattern_corr metrics
> are misleading (they measure conservatism, not correctness); use
> purity, cscr, classification_accuracy instead. Now I want to apply
> this to [NEW DATASET]. The data are [describe: platform, spot count,
> tissue type, known categories, any prior analysis].

