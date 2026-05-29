"""SpaDiff: Mass-particle reverse denoising for spatial transcriptomics.

__version__ = '4.0.0-final'

This module provides a clean interface to the SpaDiff denoising algorithm.
The core implementation is in utils_dev.py; this file provides a simplified
`denoise()` function for ease of use.

For experimental features and development, use utils_dev.py directly.

Algorithm Overview
------------------
Three-phase bulk-synchronous parallel (BSP) per time step:

  Phase A  Global deposition: w_global (M,) via particle->spot soft assignment.
  Phase B  Global score on spots: s_global_spot (M,2) from spot-level KNN KDE.
  Phase C  Per-gene-block parallel update with alpha mixing and alpha_rho scaling.

Dependencies: NumPy, SciPy (sparse, spatial), Python stdlib only.
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional, TextIO

import numpy as np
from scipy import sparse

# Import core implementation from utils_dev
from .utils_dev import (
    # Geometry helpers
    build_kdtree,
    query_knn,
    scale_coords_unit_box,
    K_from_layers,
    bandwidth_from_layers,
    # Particles
    counts_to_mass_particles,
    # Deposition
    soft_deposit_mass_particles_partial,
    # Score
    compute_global_score_on_spots,
    interpolate_spot_field_to_particles,
    # Alpha
    compute_alpha,
    compute_alpha_rho,
    compute_global_gene_density,
    # Boundary
    generate_ghost_spots,
    # Integerisation
    float_to_int_counts_multinomial,
    # Core driver
    run_denoise_mass_parallel,
)

__all__ = [
    # Core API
    "denoise",
    # Re-exported from utils_dev
    "build_kdtree",
    "query_knn",
    "scale_coords_unit_box",
    "K_from_layers",
    "bandwidth_from_layers",
    "counts_to_mass_particles",
    "soft_deposit_mass_particles_partial",
    "compute_global_score_on_spots",
    "interpolate_spot_field_to_particles",
    "compute_alpha",
    "compute_alpha_rho",
    "compute_global_gene_density",
    "generate_ghost_spots",
    "float_to_int_counts_multinomial",
    "run_denoise_mass_parallel",
]

Array = np.ndarray


def denoise(
    spot_coords: Array,
    counts: sparse.csr_matrix,
    # Schedule parameters
    T: int = 15,
    eta_base: float = 0.005,
    gamma: float = 0.9,
    eta_decay_mode: str = 'linear',  # 'linear' (default) or 'exponential' (legacy)
    eta_min_ratio: float = 0.10,     # eta at step T-1 relative to eta_base (linear mode only)
    H_START: float = 0.20,  # Old default: 0.12
    H_END: float = 0.03,
    decay_h: float = 0.10,  # Old default: 0.15
    sigma: float = 0.0,
    # Structural parameters
    grid_type: str = 'hex',
    L_assign: int = 1,
    L_score_global: int = 5,
    L_interp: int = 1,
    L_score_gene: int = 3,
    bandwidth_scale: float = 0.04,
    # Alpha mixing
    beta: Optional[float] = 0.02,  # Old default: 0.1
    tau_mass: float = 0.0,
    # Alpha_rho
    alpha_rho_percentile: float = 95.0,  # p95 recommended
    # v16: spot-level gene score + nonzero percentile for rho_ref
    gene_score_on_spots: bool = False,   # default False: particle-level (faster); set True for spot-level
    nonzero_rho_ref: bool = True,        # default True: percentile over nonzero spots only (v16 fix)
    # Numerical
    lam: float = 1e-3,
    eps: float = 1e-6,
    # Boundary
    use_ghost_spots: bool = True,
    K_boundary: int = 7,
    n_layers_ghost: int = 1,
    asymmetry_threshold_factor: float = 0.3,
    # Output
    return_float: bool = True,  # Old default: False
    # Species-aware denoising
    species_labels: Optional[np.ndarray] = None,
    # Execution
    n_procs: int = 4,
    random_state: int = 42,
    progress: bool = True,
    progress_prefix: str = "SpaDiff",
    file: Optional[TextIO] = None,
) -> Dict[str, object]:
    """SpaDiff mass-particle reverse denoising for spatial transcriptomics.

    This function applies a diffusion-based denoising algorithm that moves
    mass particles according to KDE score fields, progressively sharpening
    spatial gene expression patterns while preserving total mass.

    Algorithm
    ---------
    1. Convert UMI counts to mass particles at spot positions
    2. For T steps, compute KDE score fields and move particles toward
       density gradients, with bandwidth annealing from H_START to H_END
    3. Deposit final particle positions back to spots
    4. Round to integers using multinomial sampling

    Parameters
    ----------
    spot_coords : (M, 2) array
        Raw spot coordinates (any scale; will be normalized internally).
    counts : (M, G) sparse CSR matrix
        Integer UMI counts per spot (rows) and gene (columns).

    Schedule Parameters
    -------------------
    T : int, default=15
        Number of denoising steps. More steps = more refinement but slower.
    eta_base : float, default=0.005
        Base step size.
    gamma : float, default=0.9
        Decay factor per step for eta (used only when eta_decay_mode='exponential')
        and for the sigma schedule.
    eta_decay_mode : {'linear', 'exponential'}, default='linear'
        How eta decays across steps.
        - 'linear': eta_t = eta_base * linspace(1.0, eta_min_ratio, T). Simpler and
          gives substantially larger cumulative integration distance than 'exponential'
          at the same eta_base. Recommended when deep off-tissue spots need to
          actually reach tissue.
        - 'exponential' (legacy): eta_t = eta_base * (h_t/H_START)^2 * gamma^t.
          Decays quadratically with bandwidth and geometrically with time, so
          most motion happens in steps 0-5 and tails off rapidly.
    eta_min_ratio : float, default=0.10
        For linear mode: eta at step T-1 as a fraction of eta_base. Default 0.10
        means eta decays from eta_base to 10% of eta_base over T steps.
    H_START : float, default=0.20
        Initial bandwidth (in [0,1] coordinates). Large = smooth/global patterns.
        (Old default: 0.12)
    H_END : float, default=0.03
        Final bandwidth. Small = fine/local details.
    decay_h : float, default=0.10
        Exponential decay rate for bandwidth: h(t) = H_END + (H_START-H_END)*exp(-decay_h*t).
        (Old default: 0.15)
    sigma : float, default=0.0
        Noise std for stochastic updates. 0 = deterministic.

    Structural Parameters
    ---------------------
    grid_type : {'hex', 'square'}, default='hex'
        Spot grid type (affects K_from_layers calculation).
    L_assign : int, default=1
        Layers for soft particle->spot assignment. 1-2 recommended.
    L_score_global : int, default=5
        Layers for global score computation. 5-10 for smooth fields.
    L_interp : int, default=1
        Layers for interpolating spot scores to particles.
    L_score_gene : int, default=3
        Layers for gene-specific score. 2-4 recommended.
    bandwidth_scale : float, default=0.04
        Bandwidth = bandwidth_scale * L for all operations.

    Alpha Mixing
    ------------
    beta : float, default=0.02
        Threshold for alpha = rho/(rho+beta). Higher = more global score weight.
        (Old default: 0.1)
    tau_mass : float, default=0.0
        Mass threshold below which genes skip gene-specific scoring.
        Low-mass genes use only global score (faster, avoids noise).

    Alpha_rho Scaling
    -----------------
    alpha_rho_percentile : float, default=95.0
        Percentile for rho_ref in linear alpha_rho = 1 - rho/rho_ref.
        95.0 = p95 percentile (recommended, more robust than max).
    gene_score_on_spots : bool, default=False
        If True, compute gene-specific score and density at spot level
        (M points) and interpolate to particles, matching the global-score
        workflow. If False (default), compute directly at each particle
        position. Particle-level (False) is ~4.5x faster on typical data;
        spot-level (True) bounds per-gene runtime (robust to genes with
        outlier expression counts).
    nonzero_rho_ref : bool, default=True
        If True (v16+), percentile-based rho_ref for alpha_rho is computed only
        over spots with non-zero gene mass. This avoids rho_ref collapsing to 0
        for sparse genes (where most spots are zero), which would otherwise
        freeze all nonzero spots. If False (legacy), includes zeros in the
        percentile calculation.

    Numerical
    ---------
    lam : float, default=1e-3
        Regularization to avoid division by zero in score computation.
    eps : float, default=1e-6
        Numerical guard for soft normalization.

    Boundary Handling
    -----------------
    use_ghost_spots : bool, default=True
        Generate mirror ghost spots at tissue boundaries.
    K_boundary : int, default=7
        Neighbors for boundary detection.
    n_layers_ghost : int, default=1
        Number of ghost spot layers.
    asymmetry_threshold_factor : float, default=0.3
        Threshold for boundary detection.

    Output Control
    --------------
    return_float : bool, default=True
        If True, return float counts instead of rounded integers.
        (Old default: False)

    Species-Aware Denoising
    -----------------------
    species_labels : (G,) array of str or None, default=None
        Species label for each gene (e.g., 'human', 'mouse'). When provided,
        the global score is computed per-species: each gene's particles move
        according to the global density field built only from genes of the
        same species. This prevents cross-species contamination in chimeric
        experiments. If None, all genes share a single global score field.

    Execution
    ---------
    n_procs : int, default=4
        Number of parallel worker processes.
    random_state : int, default=42
        RNG seed for reproducibility.
    progress : bool, default=True
        Show progress bar.
    progress_prefix : str, default='SpaDiff'
        Prefix for progress bar.
    file : TextIO, optional
        Output stream for progress (default: sys.stderr).

    Returns
    -------
    dict with keys:
        'counts_denoised' : (M, G) sparse CSR matrix
            Denoised counts (int64 by default, float32 if return_float=True).
        'counts_float' : (M, G) float32 array
            Unrounded float counts (for mass conservation checks).
        'scale_info' : dict
            Coordinate normalization info with 'min', 'max', 'scale'.
        'stats' : dict
            Summary statistics:
            - mass_before: total mass before denoising
            - mass_after_float: total mass in float output
            - mass_after_int: total mass in integer output
            - mass_retention: fraction retained (should be ~1.0)
            - nonzero_before: number of nonzero entries before
            - nonzero_after: number of nonzero entries after
            - runtime_seconds: total processing time

    Example
    -------
    >>> from scipy.io import mmread
    >>> counts = mmread('matrix.mtx').T.tocsr()
    >>> coords = np.loadtxt('coordinates.txt')
    >>> result = denoise(coords, counts, T=15, eta_base=0.005)
    >>> denoised = result['counts_denoised']
    >>> print(f"Mass retention: {result['stats']['mass_retention']:.2%}")
    """
    start_time = time.perf_counter()

    if not sparse.isspmatrix_csr(counts):
        counts = counts.tocsr()

    M, G = counts.shape
    mass_before = float(counts.sum())
    nonzero_before = int(counts.nnz)

    # Build schedules
    t_arr = np.arange(T)
    h_sched = H_END + (H_START - H_END) * np.exp(-decay_h * t_arr)
    if eta_decay_mode == 'linear':
        # Linear decay from eta_base to eta_base * eta_min_ratio over T steps.
        # Independent of bandwidth, so eta stays meaningful even when h
        # has shrunk and the (h/H_START)^2 factor would have collapsed it.
        eta_sched = eta_base * np.linspace(1.0, eta_min_ratio, T).astype(np.float64)
    elif eta_decay_mode == 'exponential':
        # Legacy schedule: couples eta to bandwidth and adds geometric time decay
        eta_sched = eta_base * (h_sched / H_START) ** 2 * gamma ** t_arr
    else:
        raise ValueError(
            f"Unknown eta_decay_mode={eta_decay_mode!r}; "
            f"expected 'linear' or 'exponential'."
        )
    sigma_sched = np.full(T, sigma, dtype=np.float32)

    # Configure alpha_rho (linear mode: alpha = 1 - rho/rho_ref)
    if alpha_rho_percentile < 100.0:
        # Use percentile-based reference density
        alpha_rho_config = {
            'mode': 'linear',
            'use_global_density': True,
            'rho_ref_method': 'p95',
            'rho_ref_percentile': alpha_rho_percentile,
        }
    else:
        # Use max density as reference (default from script 31)
        alpha_rho_config = {
            'mode': 'linear',
            'use_global_density': True,
            'rho_ref_method': 'max',
        }
    # v16 additions: piggyback on alpha_rho_config dict (no signature changes
    # in run_denoise_mass_parallel or _gene_block_worker required).
    alpha_rho_config['gene_score_on_spots'] = bool(gene_score_on_spots)
    alpha_rho_config['nonzero_rho_ref'] = bool(nonzero_rho_ref)

    # Call core implementation
    result = run_denoise_mass_parallel(
        spot_coords=spot_coords,
        counts=counts,
        T=T,
        grid_type=grid_type,
        L_assign=L_assign,
        L_score_global=L_score_global,
        L_interp=L_interp,
        L_score_gene=L_score_gene,
        bandwidth_scale=bandwidth_scale,
        lam=lam,
        eps=eps,
        eta=eta_sched.astype(np.float32),
        sigma=sigma_sched.astype(np.float32),
        alpha_mode='beta',
        beta=beta,
        tau_mass=tau_mass,
        n_procs=n_procs,
        random_state=random_state,
        integerize=False,  # We'll do our own rounding
        progress=progress,
        progress_prefix=progress_prefix,
        file=file,
        use_ghost_spots=use_ghost_spots,
        K_boundary=K_boundary,
        n_layers_ghost=n_layers_ghost,
        asymmetry_threshold_factor=asymmetry_threshold_factor,
        alpha_rho_config=alpha_rho_config,
        h_schedule=h_sched.astype(np.float32),
        species_labels=species_labels,
    )

    # Extract denoised float counts
    denoised_out = result['denoised_spot_gene_float']
    if isinstance(denoised_out, np.ndarray):
        counts_float = denoised_out
    else:
        # Dict format - reconstruct matrix
        counts_float = np.zeros((M, G), dtype=np.float32)
        for g_idx, arr in denoised_out.items():
            counts_float[:, g_idx] = np.asarray(arr).flatten()

    # Integer rounding: simple round (preserves spatial pattern better than multinomial)
    if return_float:
        counts_denoised = sparse.csr_matrix(counts_float)
        mass_after_int = float(counts_float.sum())
    else:
        counts_int = np.round(counts_float).astype(np.int64)
        counts_int[counts_int < 0] = 0
        counts_denoised = sparse.csr_matrix(counts_int)
        mass_after_int = float(counts_int.sum())

    mass_after_float = float(counts_float.sum())
    nonzero_after = int(counts_denoised.nnz)
    runtime = time.perf_counter() - start_time

    stats = {
        'mass_before': mass_before,
        'mass_after_float': mass_after_float,
        'mass_after_int': mass_after_int,
        'mass_retention': mass_after_int / (mass_before + 1e-10),
        'nonzero_before': nonzero_before,
        'nonzero_after': nonzero_after,
        'runtime_seconds': runtime,
    }

    return {
        'counts_denoised': counts_denoised,
        'counts_float': counts_float,
        'scale_info': result['scale_info'],
        'stats': stats,
    }


# ---------------------------------------------------------------------------
# Sanity test
# ---------------------------------------------------------------------------


if __name__ == '__main__':
    from pathlib import Path
    from scipy.io import mmread
    from scipy.spatial import cKDTree
    import pandas as pd

    print("=" * 70)
    print("SpaDiff utils.py v4.0.0-final Sanity Test")
    print("=" * 70)

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data_denoised" / "chimeric" / "2_S2_manual" / "Original"

    if not DATA_PATH.exists():
        print(f"Test data not found at {DATA_PATH}")
        print("Skipping sanity test.")
        sys.exit(0)

    print(f"\nLoading test data from {DATA_PATH}...")

    # Load matrix
    matrix_path = DATA_PATH / "matrix.mtx"
    counts = mmread(matrix_path).T.tocsr()

    # Load coordinates
    barcodes = pd.read_csv(DATA_PATH / "barcodes.csv", header=None)[0].tolist()
    meta_df = pd.read_csv(DATA_PATH / "meta_data.csv")
    meta_df = meta_df.set_index('barcode').loc[barcodes].reset_index()
    coords = meta_df[['imagecol', 'imagerow']].values.astype(np.float32)

    # Load gene names
    features_df = pd.read_csv(DATA_PATH / "features.csv", header=None)
    gene_names = features_df[0].tolist()

    print(f"  Spots: {counts.shape[0]}")
    print(f"  Genes: {counts.shape[1]}")

    # Select top 50 genes by total count
    gene_totals = np.array(counts.sum(axis=0)).flatten()
    top_indices = np.argsort(gene_totals)[::-1][:50]
    counts_subset = counts[:, top_indices]
    selected_genes = [gene_names[i] for i in top_indices]

    print(f"\nRunning denoise on top 50 genes...")

    result = denoise(
        spot_coords=coords,
        counts=counts_subset,
        T=15,
        eta_base=0.005,
        gamma=0.9,
        H_START=0.12,
        H_END=0.03,
        decay_h=0.15,
        bandwidth_scale=0.04,
        # Use new defaults: lam=1e-3, beta=0.1, tau_mass=0.0, alpha_rho_percentile=100
        n_procs=4,
        random_state=42,
        progress=True,
    )

    stats = result['stats']
    print(f"\n{'='*70}")
    print("Results")
    print("=" * 70)
    print(f"  Mass retention:  {stats['mass_retention']:.4f}")
    print(f"  Nonzero before:  {stats['nonzero_before']}")
    print(f"  Nonzero after:   {stats['nonzero_after']}")
    print(f"  Runtime:         {stats['runtime_seconds']:.1f}s")

    # Compute Moran's I for a few genes
    def compute_moran(values, coords, k=6):
        values = np.asarray(values).flatten()
        n = len(values)
        tree = cKDTree(coords)
        _, indices = tree.query(coords, k=k+1)
        indices = indices[:, 1:]
        mean_val = values.mean()
        if np.std(values) < 1e-10:
            return 0.0
        z = values - mean_val
        W = 0.0
        numerator = 0.0
        for i in range(n):
            for j in indices[i]:
                W += 1.0
                numerator += z[i] * z[j]
        denominator = np.sum(z ** 2)
        if denominator < 1e-10:
            return 0.0
        return (n / W) * (numerator / denominator)

    # Normalize coords for Moran
    coords_norm = coords.copy()
    coords_norm[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    coords_norm[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

    counts_orig = np.asarray(counts_subset.todense())
    counts_deno = np.asarray(result['counts_denoised'].todense())

    print(f"\nMoran's I improvement (sample of 5 genes):")
    print(f"{'Gene':<25} {'Before':>10} {'After':>10} {'Improve':>10}")
    print("-" * 60)

    improvements = []
    for i in range(min(5, len(selected_genes))):
        orig_vals = counts_orig[:, i]
        deno_vals = counts_deno[:, i]
        moran_before = compute_moran(orig_vals, coords_norm)
        moran_after = compute_moran(deno_vals, coords_norm)
        improvement = moran_after - moran_before
        improvements.append(improvement)
        gene_name = selected_genes[i][:24]
        print(f"{gene_name:<25} {moran_before:>10.4f} {moran_after:>10.4f} {improvement:>+10.4f}")

    print(f"\nMean improvement: {np.mean(improvements):+.4f}")

    if stats['mass_retention'] > 0.95 and np.mean(improvements) > 0.3:
        print("\n[PASS] Sanity test passed!")
    else:
        print("\n[WARN] Results may need investigation")

    print("\n" + "=" * 70)
