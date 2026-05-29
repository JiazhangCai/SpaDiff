"""Mass-particle reverse denoising pipeline for spatial transcriptomics.

Architecture
------------
Three-phase bulk-synchronous parallel (BSP) per time step:

  Phase A  Global deposition: w_global (M,) via particle->spot soft assignment.
           Parallelised by particle chunk; reduced by summation in main process.

  Phase B  Global score on spots: s_global_spot (M,2) from spot-level KNN.
           Memory: (M, K_score_global) arrays only — NEVER (P, K) for particles.
           Even for large L_score_global (e.g. L=10 hex, K≈331, M=5000) this
           stays under ~30 MB.

  Phase C  Per-gene-block parallel update:
             1. Interpolate s_global_spot to gene particles via small L_interp KNN.
             2. For high-density genes only: deposit gene particles, compute
                gene-specific score on P_g particles ONLY.
             3. Mix scores with alpha, apply Euler-Maruyama update.
           Parallelised by gene block; each worker receives a contiguous
           particle slice covering exactly its gene block (O(P_block) pickle).

Particles are sorted by gene once at initialisation; CSR-style gene_starts
offsets allow O(1) slicing per gene throughout the run.

Dependencies: NumPy, SciPy, Python stdlib only.
"""

from __future__ import annotations

import math
import sys
import time
from multiprocessing import get_context
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

Array = np.ndarray

__all__ = [
    # geometry helpers
    "build_kdtree",
    "query_knn",
    "scale_coords_unit_box",
    "K_from_layers",
    "bandwidth_from_layers",
    # particles
    "counts_to_mass_particles",
    # deposition
    "soft_deposit_mass_particles_partial",
    # score
    "compute_global_score_on_spots",
    "interpolate_spot_field_to_particles",
    # alpha / update
    "compute_alpha",
    "compute_alpha_rho",
    "compute_global_gene_density",
    "compute_gene_alpha_morans",
    "update_positions",
    # integerisation
    "float_to_int_counts_multinomial",
    # drivers
    "run_denoise_mass",
    "run_denoise_mass_parallel",
    # boundary correction
    "generate_ghost_spots",
    # visualisation
    "make_gene_gif",
    "make_global_gif",
    "plot_score_field",
    "plot_gene_score_field",
]


# ---------------------------------------------------------------------------
# Progress meter
# ---------------------------------------------------------------------------


class _ProgressMeter:
    """Lightweight, in-place progress reporter using stderr.

    Parameters
    ----------
    total : int
        Total iterations.
    every : int
        Print every `every` steps.
    smoothing : float
        EMA smoothing factor for per-step time estimate.
    prefix : str
        Label printed at the start of each line.
    stream : TextIO
        Output stream (stderr by default).
    """

    def __init__(
        self,
        total: int,
        every: int,
        smoothing: float,
        prefix: str,
        stream: TextIO,
    ) -> None:
        self.total = max(1, int(total))
        self.every = max(1, int(every))
        self.smoothing = float(smoothing)
        self.prefix = prefix
        self.stream = stream
        self.start = time.perf_counter()
        self.last = self.start
        self.ema_dt: Optional[float] = None

    def update(self, step: int) -> None:
        now = time.perf_counter()
        dt = now - self.last
        self.last = now
        self.ema_dt = (
            dt
            if self.ema_dt is None
            else self.smoothing * self.ema_dt + (1.0 - self.smoothing) * dt
        )
        if (step + 1) % self.every != 0 and (step + 1) != self.total:
            return
        elapsed = now - self.start
        eta = 0.0 if self.ema_dt is None else self.ema_dt * (self.total - (step + 1))
        itps = 0.0 if not self.ema_dt else 1.0 / self.ema_dt
        self.stream.write(
            f"\r{self.prefix}: step {step + 1}/{self.total} | "
            f"elapsed {elapsed:.1f}s | eta {eta:.1f}s | {itps:.2f} it/s"
        )
        self.stream.flush()

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


# ---------------------------------------------------------------------------
# KDTree helpers
# ---------------------------------------------------------------------------


def build_kdtree(coords: Array) -> cKDTree:
    """Build a cKDTree from an (N,2) float32 coordinate array.

    Parameters
    ----------
    coords : (N,2) array of spot or particle coordinates.

    Returns
    -------
    cKDTree over coords (float32).
    """
    return cKDTree(np.asarray(coords, dtype=np.float32))


def query_knn(tree: cKDTree, queries: Array, k: int) -> Tuple[Array, Array]:
    """Query k nearest neighbors for each row of queries.

    Parameters
    ----------
    tree    : pre-built cKDTree.
    queries : (N,2) float32 query points.
    k       : number of neighbors (clamped to tree.n).

    Returns
    -------
    idx   : (N,k) int64   — neighbor indices into the tree's data.
    dists : (N,k) float32 — Euclidean distances to neighbors.
    """
    k = min(k, tree.n)
    dists, idx = tree.query(np.asarray(queries, dtype=np.float32), k=k)
    if idx.ndim == 1:
        idx = idx[:, None]
        dists = dists[:, None]
    return idx.astype(np.int64, copy=False), dists.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Coordinate scaling and geometry
# ---------------------------------------------------------------------------


def scale_coords_unit_box(
    coords: Array, eps: float = 1e-8
) -> Tuple[Array, Dict[str, Array]]:
    """Scale raw spot coordinates to the unit box [0,1]^2.

    Scaling is per-dimension:  scaled = (coords - min) / (max - min + eps).
    This makes bandwidth parameters platform-invariant across array types.

    Parameters
    ----------
    coords : (M,2) float32 raw coordinates.
    eps    : guard against zero-range axes.

    Returns
    -------
    scaled : (M,2) float32 in [0,1]^2.
    info   : dict with keys "min", "max", "scale" (per-dimension arrays).
    """
    coords = np.asarray(coords, dtype=np.float32)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    scale = mx - mn
    scaled = (coords - mn) / (scale + eps)
    return scaled.astype(np.float32), {"min": mn, "max": mx, "scale": scale}


def K_from_layers(grid_type: str, L: int) -> int:
    """Number of spots within L neighborhood layers of a central spot.

    Parameters
    ----------
    grid_type : "hex" or "square".
    L         : number of layers (0 = center only).

    Returns
    -------
    K : int
        hex    : 1 + 3*L*(L+1)
        square : (2*L+1)^2
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    if grid_type == "hex":
        return 1 + 3 * L * (L + 1)
    if grid_type == "square":
        return (2 * L + 1) ** 2
    raise ValueError(f"grid_type must be 'hex' or 'square', got {grid_type!r}")


def bandwidth_from_layers(L: int, c: float = 0.05) -> float:
    """Gaussian bandwidth in scaled [0,1] coordinates from a layer count.

    h = max(c * L, 1e-4)

    Parameters
    ----------
    L : neighborhood layers.
    c : scaling constant (default 0.05 gives h=0.05 for L=1 hex).
    """
    if L < 0:
        raise ValueError("L must be >= 0")
    return float(max(c * L, 1e-4))


# ---------------------------------------------------------------------------
# Boundary ghost spots
# ---------------------------------------------------------------------------


def generate_ghost_spots(
    spot_coords: Array,
    K_boundary: int = 7,
    n_layers_ghost: int = 1,
    asymmetry_threshold_factor: float = 0.3,
) -> Tuple[Array, Array]:
    """Detect tissue boundary spots and generate mirror ghost spots.

    Boundary spots are identified by the asymmetry of their K-nearest-neighbour
    centroid: if the centroid lies far from the spot, its neighbourhood is
    one-sided (tissue edge).  A ghost spot is placed on the exterior side to
    symmetrise the KDE neighbourhood used by the score function.

    Ghost spots are used ONLY in score and interpolation steps; they never
    receive deposited mass and never appear in the output.

    Parameters
    ----------
    spot_coords              : (M,2) float32 — scaled coordinates in [0,1]^2.
    K_boundary               : neighbours used for asymmetry estimation.
    n_layers_ghost           : number of ghost layers per boundary spot (≥1).
    asymmetry_threshold_factor : boundary threshold = factor * median_nn_dist.
                                 Lower values detect more boundary spots.

    Returns
    -------
    ghost_coords  : (N_ghost,2) float32 — ghost spot positions (may be empty).
    boundary_idx  : (N_ghost,)  int64   — index of source real spot for each ghost.
    """
    M = spot_coords.shape[0]
    if M < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

    K = min(K_boundary, M - 1)
    tree = build_kdtree(spot_coords)
    # K+1 because the first hit is the query point itself (distance 0)
    idx, dists = query_knn(tree, spot_coords, K + 1)   # (M, K+1)
    nn_idx   = idx[:, 1:]    # (M, K)  — exclude self
    nn_dists = dists[:, 1:]  # (M, K)

    # Centroid of K nearest neighbours
    nn_coords = spot_coords[nn_idx]          # (M, K, 2)
    centroid  = nn_coords.mean(axis=1)       # (M, 2)

    # Asymmetry vector: points from spot toward the interior of its neighbourhood
    asym_vec = centroid - spot_coords        # (M, 2)
    asym_mag = np.linalg.norm(asym_vec, axis=1, keepdims=False)  # (M,)

    median_nn_dist = float(np.median(nn_dists[:, 0]))
    threshold      = asymmetry_threshold_factor * median_nn_dist
    boundary_mask  = asym_mag > threshold
    bnd_idx        = np.where(boundary_mask)[0]  # real-spot indices on boundary

    if len(bnd_idx) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

    ghost_list: List[Array] = []
    src_list:   List[int]   = []
    for layer in range(1, n_layers_ghost + 1):
        for bi in bnd_idx:
            mag = float(asym_mag[bi])
            if mag < 1e-10:
                continue
            # Unit vector pointing OUTWARD (opposite to interior asymmetry)
            direction = asym_vec[bi] / mag
            ghost_pos = spot_coords[bi] - np.float32(layer * median_nn_dist) * direction
            ghost_list.append(ghost_pos.astype(np.float32))
            src_list.append(int(bi))

    if not ghost_list:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)

    ghost_coords = np.stack(ghost_list, axis=0).astype(np.float32)  # (N_ghost, 2)
    boundary_idx = np.array(src_list, dtype=np.int64)                # (N_ghost,)
    return ghost_coords, boundary_idx


# ---------------------------------------------------------------------------
# Mass particles
# ---------------------------------------------------------------------------


def counts_to_mass_particles(
    counts_csr: sparse.csr_matrix,
    spot_coords: Array,
) -> Tuple[Array, Array, Array]:
    """Convert a sparse count matrix to mass particles.

    One particle is created per nonzero (spot, gene) entry with mass equal
    to the count value.  No per-UMI expansion; this keeps P = counts.nnz.

    Parameters
    ----------
    counts_csr  : (M,G) csr_matrix with integer counts.
    spot_coords : (M,2) float32 scaled spot coordinates.

    Returns
    -------
    particle_pos0 : (P,2) float32 — initial positions (source spot coordinate).
    particle_gene : (P,) int64   — gene index for each particle.
    particle_mass : (P,) float32 — count value (mass) for each particle.
    """
    if counts_csr.nnz == 0:
        raise ValueError("counts_csr has no nonzero entries; nothing to denoise.")
    coo = counts_csr.tocoo()
    particle_pos0 = np.asarray(spot_coords, dtype=np.float32)[coo.row]
    particle_gene = coo.col.astype(np.int64)
    particle_mass = coo.data.astype(np.float32)
    return particle_pos0, particle_gene, particle_mass


# ---------------------------------------------------------------------------
# Phase A: global deposition (partial, for pool.starmap)
# ---------------------------------------------------------------------------


def soft_deposit_mass_particles_partial(
    pos_slice: Array,
    mass_slice: Array,
    spot_coords: Array,
    h_assign: float,
    K_assign: int,
    eps: float,
) -> Array:
    """Deposit a slice of particles onto the spot grid (partial contribution).

    Multiple partials are summed in the main process to form w_global (M,).
    Soft assignment uses a Gaussian kernel normalised per particle (softmax
    over the K_assign nearest spots).

    Parameters
    ----------
    pos_slice   : (N,2) float32 — particle positions for this slice.
    mass_slice  : (N,)  float32 — particle masses.
    spot_coords : (M,2) float32 — spot coordinates.
    h_assign    : float          — Gaussian bandwidth for assignment.
    K_assign    : int            — number of nearest spots to assign to.
    eps         : float          — numerical guard for normalisation.

    Returns
    -------
    partial : (M,) float32 — partial w_global from this slice.
    """
    tree = build_kdtree(spot_coords)
    idx, dists = query_knn(tree, pos_slice, K_assign)        # (N, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_assign * h_assign)
    ).astype(np.float32)                                       # (N, K)
    # Add a tiny fallback so mass is conserved even when particles drift far
    # outside the spot grid (kernel underflows to 0 in float32).  The fallback
    # distributes mass uniformly among the K nearest spots in that edge case.
    kernel += np.float32(1e-30)
    pi = kernel / (kernel.sum(axis=1, keepdims=True) + eps)   # (N, K)
    weighted = mass_slice[:, None] * pi                        # (N, K)
    M = spot_coords.shape[0]
    return np.bincount(
        idx.ravel(), weights=weighted.ravel(), minlength=M
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase B: global score on spots (memory-safe even for large L_score_global)
# ---------------------------------------------------------------------------


def compute_global_score_on_spots(
    w_global: Array,
    spot_coords: Array,
    h_global: float,
    K_score_global: int,
    lam: float,
    eps: float = 1e-8,
    ghost_coords: Optional[Array] = None,
    ghost_boundary_idx: Optional[Array] = None,
) -> Array:
    """Compute the global score field at spot locations.

    The score is the KDE score function evaluated using the deposited mass
    w_global as weights on the spot grid.  Operating on spots (size M)
    instead of particles (size P) keeps memory bounded at O(M * K_score_global)
    regardless of the number of particles.

    When ghost_coords / ghost_boundary_idx are provided, the KDE is built
    over an augmented grid (real + ghost spots) to symmetrise boundary
    neighbourhoods and remove the inward score bias at tissue edges.
    Ghost spot weights mirror their source real spot: w_ghost = w_global[boundary_idx].
    The returned score array covers only the M real spots.

    Peak memory: (M, K_score_global) float32 arrays, plus (M, K_score_global, 2)
    for offsets.  For M=5000, K=331 (L=10 hex): ~13 MB total.

    Parameters
    ----------
    w_global            : (M,) float32 — deposited mass at each spot.
    spot_coords         : (M,2) float32 — scaled spot coordinates.
    h_global            : float          — Gaussian bandwidth.
    K_score_global      : int            — number of spot-to-spot neighbors.
    lam                 : float          — density regularisation to avoid div/0.
    eps                 : float          — numerical guard.
    ghost_coords        : (N_ghost,2) float32 or None — ghost spot positions.
    ghost_boundary_idx  : (N_ghost,)  int64  or None  — source real-spot index
                          for each ghost (used to mirror weights).

    Returns
    -------
    s_global_spot : (M,2) float32 — score vector at each spot location.
    """
    M = spot_coords.shape[0]
    # Augment coords and weights with ghost spots if provided
    if ghost_coords is not None and len(ghost_coords) > 0:
        aug_coords = np.vstack([spot_coords, ghost_coords]).astype(np.float32)
        w_ghost    = w_global[ghost_boundary_idx]               # mirror weights
        w_aug      = np.concatenate([w_global, w_ghost])
    else:
        aug_coords = spot_coords
        w_aug      = w_global

    M_aug = aug_coords.shape[0]
    K = min(K_score_global, M_aug)
    tree = build_kdtree(aug_coords)
    # Query real spots against augmented tree
    idx, dists = query_knn(tree, spot_coords, K)               # (M, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_global * h_global)
    ).astype(np.float32)                                       # (M, K)
    # Offset from real spot m toward its neighbour j (may be a ghost)
    offsets = (aug_coords[idx] - spot_coords[:, None, :]).astype(np.float32)
    #                                                          # (M, K, 2)
    w = w_aug[idx]                                             # (M, K)
    R = np.sum(w * kernel, axis=1)                             # (M,) density
    num = np.sum(w[:, :, None] * kernel[:, :, None] * offsets, axis=1)  # (M,2)
    s = num / (h_global * h_global * (R[:, None] + lam))
    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Phase C helpers (executed inside the gene-block worker)
# ---------------------------------------------------------------------------


def interpolate_spot_field_to_particles(
    pos_g: Array,
    s_global_spot: Array,
    spot_tree: cKDTree,
    h_interp: float,
    K_interp: int,
    eps: float = 1e-8,
    aug_tree: Optional[cKDTree] = None,
    s_aug: Optional[Array] = None,
) -> Array:
    """Kernel-weighted interpolation of spot-level field to particle positions.

    Uses a small neighborhood (K_interp, derived from L_interp) to keep
    per-gene memory tiny: (P_g, K_interp) arrays only.

    When aug_tree / s_aug are provided (ghost-spot mode), particles query the
    augmented tree so they can "see" ghost spots on the exterior side of the
    tissue boundary, producing a more symmetric interpolation near edges.

    Parameters
    ----------
    pos_g         : (P_g,2) float32 — gene-g particle positions.
    s_global_spot : (M,2)   float32 — score field at spot locations.
    spot_tree     : pre-built cKDTree over real spot_coords (M spots).
    h_interp      : float            — Gaussian bandwidth for interpolation.
    K_interp      : int              — number of spot neighbors.
    eps           : float            — numerical guard for normalisation.
    aug_tree      : cKDTree or None  — tree over augmented coords (M+N_ghost).
    s_aug         : (M+N_ghost,2) float32 or None — augmented score field.

    Returns
    -------
    s_interp : (P_g,2) float32 — interpolated score at each gene-g particle.
    """
    use_tree = aug_tree if (aug_tree is not None) else spot_tree
    use_s    = s_aug    if (s_aug    is not None) else s_global_spot
    K = min(K_interp, use_tree.n)
    idx, dists = query_knn(use_tree, pos_g, K)                # (P_g, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_interp * h_interp)
    ).astype(np.float32)                                       # (P_g, K)
    weights = kernel / (kernel.sum(axis=1, keepdims=True) + eps)  # (P_g, K)
    s_interp = np.sum(weights[:, :, None] * use_s[idx], axis=1)
    return s_interp.astype(np.float32)                         # (P_g, 2)


def _deposit_gene_particles(
    pos_g: Array,
    mass_g: Array,
    spot_coords: Array,
    h_assign: float,
    K_assign: int,
    eps: float,
    M: int,
    spot_tree: cKDTree,
) -> Array:
    """Deposit gene-g particles to the spot grid.

    Identical to the global deposition but accepts a pre-built KDTree so the
    tree is not rebuilt for every gene in a worker.

    Parameters
    ----------
    pos_g       : (P_g,2) float32 — gene-g particle positions.
    mass_g      : (P_g,)  float32 — gene-g particle masses.
    spot_coords : (M,2)   float32 — spot coordinates (unused; kept for clarity).
    h_assign    : float            — bandwidth.
    K_assign    : int              — number of nearest spots.
    eps         : float            — numerical guard.
    M           : int              — number of spots (for bincount minlength).
    spot_tree   : pre-built cKDTree over spot_coords.

    Returns
    -------
    w_gene_g : (M,) float32 — deposited mass at each spot for gene g.
    """
    idx, dists = query_knn(spot_tree, pos_g, K_assign)        # (P_g, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_assign * h_assign)
    ).astype(np.float32)
    kernel += np.float32(1e-30)  # fallback: preserve mass for off-grid particles
    pi = kernel / (kernel.sum(axis=1, keepdims=True) + eps)
    weighted = mass_g[:, None] * pi                            # (P_g, K)
    return np.bincount(
        idx.ravel(), weights=weighted.ravel(), minlength=M
    ).astype(np.float32)


def _gene_score_on_spots(
    spot_coords: Array,
    w_gene_g: Array,
    h_gene: float,
    K_score_gene: int,
    lam: float,
    spot_tree: cKDTree,
    aug_coords: Optional[Array] = None,
    aug_w_gene_g: Optional[Array] = None,
    aug_tree: Optional[cKDTree] = None,
    eps: float = 1e-8,
) -> Tuple[Array, Array]:
    """Compute gene-specific score and density at each SPOT position.

    Mirrors compute_global_score_on_spots but uses gene-specific weights
    (w_gene_g) instead of global aggregate mass. All computation is at spot
    level O(M * K) — no per-particle density is produced. Downstream the
    scalar alpha and score vector are interpolated to particle positions,
    matching the global-score workflow.

    Parameters
    ----------
    spot_coords  : (M, 2) float32 — real spot coordinates.
    w_gene_g     : (M,)   float32 — gene-specific deposited mass at real spots.
    h_gene       : float            — Gaussian bandwidth.
    K_score_gene : int              — number of nearest neighbor spots.
    lam          : float            — density regularization.
    spot_tree    : cKDTree over real spot_coords.
    aug_coords   : (M+N_ghost, 2) or None — augmented coordinates (real + ghost).
    aug_w_gene_g : (M+N_ghost,) or None   — augmented gene mass.
    aug_tree     : cKDTree over aug_coords or None.

    Returns
    -------
    s_gene_spot   : (M, 2) float32 — gene-specific score at each real spot.
    rho_gene_spot : (M,)   float32 — gene-specific KDE density at each real spot.
    """
    use_coords = aug_coords   if (aug_coords   is not None) else spot_coords
    use_w      = aug_w_gene_g if (aug_w_gene_g is not None) else w_gene_g
    use_tree   = aug_tree     if (aug_tree     is not None) else spot_tree
    K = min(K_score_gene, use_tree.n)
    # Query real spots against augmented tree (mirrors global score routine)
    idx, dists = query_knn(use_tree, spot_coords, K)          # (M, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_gene * h_gene)
    ).astype(np.float32)                                       # (M, K)
    offsets = (use_coords[idx] - spot_coords[:, None, :]).astype(np.float32)
    #                                                          # (M, K, 2)
    w = use_w[idx]                                             # (M, K)
    rho = np.sum(w * kernel, axis=1)                           # (M,) density
    num = np.sum(w[:, :, None] * kernel[:, :, None] * offsets, axis=1)
    #                                                          # (M, 2)
    s = num / (h_gene * h_gene * (rho[:, None] + lam))
    return s.astype(np.float32), rho.astype(np.float32)


def _gene_score_on_gene_particles(
    pos_g: Array,
    w_gene_g: Array,
    spot_coords: Array,
    h_gene: float,
    K_score_gene: int,
    lam: float,
    spot_tree: cKDTree,
    aug_coords: Optional[Array] = None,
    aug_w_gene_g: Optional[Array] = None,
    aug_tree: Optional[cKDTree] = None,
) -> Tuple[Array, Array]:
    """Compute gene-specific score and density for gene-g particles ONLY.

    NEVER builds (P_all, K) arrays.  Peak memory: (P_g, K_score_gene) arrays.

    When aug_coords / aug_w_gene_g / aug_tree are provided (ghost-spot mode),
    the KDE uses the augmented neighbourhood so boundary particles receive a
    symmetric score estimate.

    Parameters
    ----------
    pos_g        : (P_g,2) float32 — gene-g particle positions.
    w_gene_g     : (M,)   float32  — gene-g deposited mass on real spots.
    spot_coords  : (M,2)  float32  — real spot coordinates.
    h_gene       : float            — Gaussian bandwidth.
    K_score_gene : int              — number of nearest spots.
    lam          : float            — density regularisation.
    spot_tree    : pre-built cKDTree over real spot_coords (M spots).
    aug_coords   : (M+N_ghost,2) float32 or None — augmented coordinates.
    aug_w_gene_g : (M+N_ghost,) float32 or None  — augmented gene mass
                   (ghost entries = w_gene_g[boundary_idx]).
    aug_tree     : cKDTree or None — tree over aug_coords.

    Returns
    -------
    s_gene : (P_g,2) float32 — gene-specific score at each gene-g particle.
    rho    : (P_g,)  float32 — gene-specific density at each particle.
    """
    use_coords = aug_coords   if (aug_coords   is not None) else spot_coords
    use_w      = aug_w_gene_g if (aug_w_gene_g is not None) else w_gene_g
    use_tree   = aug_tree     if (aug_tree     is not None) else spot_tree
    K = min(K_score_gene, use_tree.n)
    idx, dists = query_knn(use_tree, pos_g, K)                # (P_g, K)
    kernel = np.exp(
        -dists * dists / (2.0 * h_gene * h_gene)
    ).astype(np.float32)                                       # (P_g, K)
    # Offset from particle toward neighbouring spot/ghost: coord_j - x_p
    offsets = (use_coords[idx] - pos_g[:, None, :]).astype(np.float32)
    #                                                          # (P_g, K, 2)
    w = use_w[idx]                                             # (P_g, K)
    rho = np.sum(w * kernel, axis=1)                           # (P_g,)
    num = np.sum(w[:, :, None] * kernel[:, :, None] * offsets, axis=1)
    #                                                          # (P_g, 2)
    s = num / (h_gene * h_gene * (rho[:, None] + lam))
    return s.astype(np.float32), rho.astype(np.float32)


# ---------------------------------------------------------------------------
# Alpha mixing weight and Euler update
# ---------------------------------------------------------------------------


def compute_alpha(
    rho: Array,
    R: Optional[Array] = None,
    eps: float = 1e-8,
    mode: str = "beta",
    beta: Optional[float] = None,
) -> Array:
    """Compute per-particle alpha mixing weight in [0,1].

    alpha blends gene-specific score (alpha=1) with global score (alpha=0).
    High gene-specific density -> higher alpha -> gene score dominates.

    Parameters
    ----------
    rho  : (P_g,) float32 — gene-specific density at particles.
    R    : (P_g,) float32 or None — global density (needed for mode='ratio').
    eps  : float — numerical guard.
    mode : "beta"  (recommended stable):  alpha = rho / (rho + beta + eps).
           "ratio" (requires R):           alpha = rho / (rho + R + eps).
    beta : scalar mixing threshold for "beta" mode.
           If None, uses median(rho) adaptively.

    Returns
    -------
    alpha : (P_g,) float32, values in [0,1].
    """
    if mode == "ratio":
        if R is None:
            raise ValueError("R must be provided for mode='ratio'")
        a = rho / (rho + R + eps)
    elif mode == "beta":
        beta_val = beta if beta is not None else float(np.median(rho))
        a = rho / (rho + beta_val + eps)
    else:
        raise ValueError(f"Unknown alpha mode: {mode!r}")
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def compute_alpha_rho(
    rho: Array,
    rho_ref: Optional[float] = None,
    gamma: float = 2.0,
    mode: str = "sigmoid",
) -> Tuple[Array, float]:
    """Density-aware score scaling factor.

    In high-density regions, the score is suppressed to prevent over-concentration.
    alpha(rho) decreases as local density increases beyond rho_ref.

    Parameters
    ----------
    rho      : (P,) or (M,) float32 — local density at each point.
    rho_ref  : float or None — threshold density.
               For sigmoid/exp: if None, uses median(rho[rho > 0]).
               For linear: if None, uses 95th percentile of rho (more robust than max).
    gamma    : float — decay exponent (larger = sharper transition).
    mode     : "sigmoid" → 1 / (1 + (rho / rho_ref)^gamma)
               "exp"     → exp(-(rho / rho_ref)^gamma)
               "linear"  → 1 - rho / rho_max (rho_ref used as rho_max)

    Returns
    -------
    alpha    : same shape as rho, values in [0, 1].
    rho_ref_used : float — the actual rho_ref value used (for logging).
    """
    rho = np.asarray(rho, dtype=np.float32)

    if mode == "linear":
        # Linear mode: alpha = 1 - rho / rho_ref
        # Use 95th percentile as reference (more robust than max)
        # Values above p95 will have negative alpha before clip, which becomes 0
        if rho_ref is None:
            rho_ref = float(np.percentile(rho, 95)) if len(rho) > 0 else 1.0
        alpha = 1.0 - rho / (rho_ref + 1e-8)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32), rho_ref

    # Compute rho_ref if not provided (for sigmoid/exp modes)
    if rho_ref is None:
        positive_rho = rho[rho > 0]
        if len(positive_rho) > 0:
            rho_ref = float(np.median(positive_rho))
        else:
            rho_ref = 1.0  # fallback

    # Compute alpha based on mode
    ratio = rho / (rho_ref + 1e-10)

    if mode == "sigmoid":
        alpha = 1.0 / (1.0 + np.power(ratio, gamma))
    elif mode == "exp":
        alpha = np.exp(-np.power(ratio, gamma))
    else:
        raise ValueError(f"Unknown alpha_rho mode: {mode!r}")

    return np.clip(alpha, 0.0, 1.0).astype(np.float32), rho_ref


def compute_global_gene_density(
    pos: Array,
    spot_coords: Array,
    w_gene: Array,
    h: float,
) -> Array:
    """Compute global gene density at each particle position.

    Sums over ALL spots (not just K nearest neighbors) to get a true
    global density estimate. This allows particles outside the gene
    expression region to have low density (and thus high alpha),
    while particles inside have high density (low alpha).

    rho_global(x) = sum_j w_j * exp(-||x - x_j||^2 / (2 * h^2))

    For M < 2000, uses direct broadcast computation.
    For larger M, could use KDTree query_ball_point for efficiency.

    Parameters
    ----------
    pos         : (P, 2) float32 — particle positions.
    spot_coords : (M, 2) float32 — spot coordinates.
    w_gene      : (M,)   float32 — gene mass at each spot.
    h           : float          — Gaussian bandwidth.

    Returns
    -------
    rho : (P,) float32 — global gene density at each particle.
    """
    pos = np.asarray(pos, dtype=np.float32)
    spot_coords = np.asarray(spot_coords, dtype=np.float32)
    w_gene = np.asarray(w_gene, dtype=np.float32)

    P = pos.shape[0]
    M = spot_coords.shape[0]

    if M < 2000:
        # Direct broadcast: (P, 1, 2) - (1, M, 2) -> (P, M, 2)
        diff = pos[:, None, :] - spot_coords[None, :, :]  # (P, M, 2)
        dist_sq = np.sum(diff**2, axis=2)  # (P, M)
        weights = np.exp(-dist_sq / (2 * h**2))  # (P, M)
        rho = np.sum(weights * w_gene[None, :], axis=1)  # (P,)
    else:
        # Use KDTree for efficiency with large M
        from scipy.spatial import cKDTree
        tree = cKDTree(spot_coords)
        rho = np.zeros(P, dtype=np.float32)
        radius = 4 * h  # Gaussian is ~0 beyond 4 sigma

        for i in range(P):
            idx = tree.query_ball_point(pos[i], r=radius)
            if len(idx) > 0:
                d_sq = np.sum((spot_coords[idx] - pos[i])**2, axis=1)
                rho[i] = np.sum(w_gene[idx] * np.exp(-d_sq / (2 * h**2)))

    return rho.astype(np.float32)


def compute_gene_alpha_morans(
    counts: sparse.csr_matrix,
    spot_coords: Array,
    k_neighbors: int = 6,
    sigmoid_k: float = 10.0,
    sigmoid_threshold: float = 0.15,
    tau_mass: float = 50.0,
) -> Array:
    """Compute per-gene alpha based on Moran's I spatial autocorrelation.

    Genes with strong spatial pattern (high Moran's I) get alpha -> 1,
    meaning gene-specific score dominates for ALL particles of that gene.
    Genes with weak spatial pattern (low Moran's I, e.g. housekeeping)
    get alpha -> 0, meaning global score dominates.

    This is a gene-level (not per-particle) alpha, computed once before
    the denoising loop begins.

    Parameters
    ----------
    counts           : (M, G) csr_matrix — original UMI counts.
    spot_coords      : (M, 2) float32    — spot coordinates (any scale).
    k_neighbors      : int   — KNN neighbors for spatial weight matrix.
    sigmoid_k        : float — steepness of the sigmoid mapping I -> alpha.
    sigmoid_threshold: float — Moran's I value at which alpha = 0.5.
    tau_mass         : float — genes with total mass below this get alpha = 0.

    Returns
    -------
    alpha_gene : (G,) float32 — per-gene alpha in [0, 1].
    morans_I   : (G,) float32 — raw Moran's I values for diagnostics.
    """
    if not sparse.isspmatrix_csc(counts):
        counts_csc = counts.tocsc()
    else:
        counts_csc = counts
    M, G = counts.shape

    # Build KNN spatial weight matrix (row-standardised)
    tree = cKDTree(spot_coords)
    dists, idx = tree.query(spot_coords, k=k_neighbors + 1)  # +1 for self
    # Remove self (first column)
    idx = idx[:, 1:]
    dists = dists[:, 1:]

    # Binary KNN weights, row-standardised
    # W_ij = 1/k if j in KNN(i), else 0
    w_inv_k = np.float32(1.0 / k_neighbors)

    morans_I = np.full(G, np.float32(0.0))
    gene_mass = np.array(counts.sum(axis=0)).ravel()

    for g in range(G):
        if gene_mass[g] < tau_mass:
            morans_I[g] = 0.0
            continue

        x = np.array(counts_csc[:, g].toarray()).ravel().astype(np.float64)
        x_mean = x.mean()
        z = x - x_mean
        ss = np.dot(z, z)
        if ss < 1e-12:
            morans_I[g] = 0.0
            continue

        # Moran's I = (M / W) * (z' W z) / (z' z)
        # With row-standardised weights, W (sum of all weights) = M
        # So I = (z' W z) / (z' z)  where (Wz)_i = (1/k) * sum_{j in KNN(i)} z_j
        wz = np.zeros(M, dtype=np.float64)
        for i in range(M):
            wz[i] = z[idx[i]].sum() * w_inv_k
        numerator = np.dot(z, wz)
        morans_I[g] = np.float32(numerator / ss)

    # Sigmoid mapping: alpha = sigmoid(k * (I - threshold))
    alpha_gene = 1.0 / (1.0 + np.exp(-sigmoid_k * (morans_I - sigmoid_threshold)))
    # Force low-mass genes to alpha = 0
    alpha_gene[gene_mass < tau_mass] = 0.0

    return alpha_gene.astype(np.float32), morans_I.astype(np.float32)


def update_positions(
    x: Array,
    step: float,
    score: Array,
    sigma: float,
    rng: np.random.Generator,
) -> Array:
    """Euler-Maruyama position update.

    x <- x + step * score + sigma * N(0,I)

    Parameters
    ----------
    x     : (P,2) float32 — current positions.
    step  : float — step size (eta_t).
    score : (P,2) float32 — mixed score vector.
    sigma : float — noise standard deviation (0 = deterministic).
    rng   : numpy Generator for reproducible noise.

    Returns
    -------
    x_new : (P,2) float32.
    """
    x = x + step * score
    if sigma > 0.0:
        x = x + sigma * rng.standard_normal(size=x.shape).astype(np.float32)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Optional integerisation of final counts
# ---------------------------------------------------------------------------


def float_to_int_counts_multinomial(
    counts_float: Array,
    totals_float: Array,
    random_state: Optional[int] = None,
) -> Tuple[Array, Array]:
    """Round float denoised counts to integers via multinomial sampling.

    Per-spot total counts are preserved (up to rounding of the total itself).

    Parameters
    ----------
    counts_float : (M,G) float32 — denoised float counts.
    totals_float : (M,)  float32 — per-spot total deposited mass.
    random_state : int or None — RNG seed.

    Returns
    -------
    counts_int : (M,G) int64.
    totals_int : (M,)  int64 — row sums of counts_int.
    """
    rng = np.random.default_rng(random_state)
    M, G = counts_float.shape
    counts_int = np.zeros((M, G), dtype=np.int64)
    totals_int = np.zeros(M, dtype=np.int64)
    for m in range(M):
        total_int = int(np.round(totals_float[m]))
        if total_int <= 0:
            continue
        probs = counts_float[m].astype(np.float64).clip(min=0.0)
        s = probs.sum()
        if s <= 0:
            probs = np.ones(G, dtype=np.float64) / G
        else:
            probs = probs / s
            # Ensure probabilities sum exactly to 1.0 for multinomial
            probs = probs.clip(min=0.0)
            probs = probs / probs.sum()
            probs[-1] = 1.0 - probs[:-1].sum()
            if probs[-1] < 0:
                probs[-1] = 0.0
                probs = probs / probs.sum()
                probs[-1] = 1.0 - probs[:-1].sum()
        counts_int[m] = rng.multinomial(total_int, probs)
        totals_int[m] = counts_int[m].sum()
    return counts_int, totals_int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_chunks(total: int, n_chunks: int) -> List[Tuple[int, int]]:
    """Divide [0, total) into n_chunks roughly equal (start, end) ranges."""
    n_chunks = max(1, n_chunks)
    base, rem = divmod(total, n_chunks)
    chunks: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < rem else 0)
        if start < end:
            chunks.append((start, end))
        start = end
    return chunks


def _make_gene_blocks(
    G: int,
    gene_starts: Array,
    n_blocks: int,
) -> List[List[int]]:
    """Partition genes into contiguous blocks with roughly equal particle counts.

    Contiguous gene ranges ensure that the particle slice for each block is
    also contiguous in the gene-sorted array, avoiding any O(P) scatter.

    Parameters
    ----------
    G           : total number of genes.
    gene_starts : (G+1,) int64 — CSR-style offsets into the sorted particle array.
    n_blocks    : desired number of blocks (may produce fewer if G is small).

    Returns
    -------
    blocks : list of non-empty lists of consecutive gene indices.
    """
    n_blocks = max(1, n_blocks)
    total = int(gene_starts[G])
    if total == 0 or G == 0:
        return []
    target = total / n_blocks
    blocks: List[List[int]] = []
    current_start = 0
    current_count = 0
    for g in range(G):
        current_count += int(gene_starts[g + 1] - gene_starts[g])
        if current_count >= target and len(blocks) < n_blocks - 1:
            blocks.append(list(range(current_start, g + 1)))
            current_start = g + 1
            current_count = 0
    if current_start < G:
        blocks.append(list(range(current_start, G)))
    return [b for b in blocks if b]


# ---------------------------------------------------------------------------
# Gene-block worker — top-level so it is picklable by multiprocessing.spawn
# ---------------------------------------------------------------------------


def _gene_block_worker(
    gene_block: List[int],
    pos_slice: Array,
    mass_slice: Array,
    local_starts: Array,
    s_global_spot: Array,
    spot_coords: Array,
    h_assign: float,
    K_assign: int,
    h_interp: float,
    K_interp: int,
    h_gene: float,
    K_score_gene: int,
    lam: float,
    eps: float,
    low_gene_mask: Array,
    alpha_mode: str,
    beta: Optional[float],
    eta_t: float,
    sigma_t: float,
    seed: int,
    ghost_coords: Optional[Array] = None,
    boundary_idx: Optional[Array] = None,
    alpha_gene: Optional[Array] = None,
    frozen_w_gene_block: Optional[Dict[int, Array]] = None,
    alpha_rho_config: Optional[Dict] = None,
    s_global_spot_dict: Optional[Dict[int, Array]] = None,
    gene_to_species_idx: Optional[Array] = None,
) -> Tuple[Array, Dict]:
    """Process a contiguous block of genes: interpolate, score, mix, update.

    Receives only the gene block's contiguous particle slice so pickling
    overhead scales with P_block, not P_total.

    Parameters
    ----------
    gene_block    : list of consecutive gene indices for this worker.
    pos_slice     : (P_block,2) float32 — gene-sorted positions for this block.
    mass_slice    : (P_block,)  float32 — gene-sorted masses (read-only).
    local_starts  : (len(gene_block)+1,) int64 — offsets within pos_slice
                    (local_starts[i]:local_starts[i+1] = particles of gene_block[i]).
    s_global_spot : (M,2) float32 — global score field on spots (Phase B output).
    spot_coords   : (M,2) float32 — scaled spot coordinates.
    h_assign, K_assign       : deposition bandwidth and neighbor count.
    h_interp, K_interp       : interpolation bandwidth and neighbor count.
    h_gene, K_score_gene     : gene-score bandwidth and neighbor count.
    lam           : density regularisation.
    eps           : numerical guard.
    low_gene_mask : (G,) bool — True for genes that skip gene-specific score.
    alpha_mode    : "beta" or "ratio".
    beta          : mixing threshold scalar or None.
    eta_t         : Euler step size at this time step.
    sigma_t       : noise standard deviation at this time step.
    seed          : RNG seed for this worker and step.
    ghost_coords  : (N_ghost,2) float32 or None — ghost spot positions.
    boundary_idx  : (N_ghost,) int64 or None — source real-spot index per ghost.
    frozen_w_gene_block : dict[int -> (M,) float32] or None — frozen reference
                    w_gene for genes in this block (used when frozen_score=True).
    alpha_rho_config : dict or None — density-aware score scaling config.
                    Keys: 'rho_ref' (float or None), 'gamma' (float), 'mode' (str).
    s_global_spot_dict : dict[int -> (M,2) float32] or None — species-specific
                    global scores. If provided, each gene uses its species' score.
    gene_to_species_idx : (G,) int32 or None — species index for each gene.

    Returns
    -------
    new_pos_slice : (P_block,2) float32 — updated particle positions.
    alpha_rho_stats : dict — statistics about alpha_rho values if config provided.
    """
    rng = np.random.default_rng(seed)
    M = spot_coords.shape[0]
    # Deposition always uses real spots only (no ghost spots in Phase A)
    spot_tree = build_kdtree(spot_coords)

    # Build augmented tree/score for interpolation and gene score (Phase C)
    use_ghost = (ghost_coords is not None) and (len(ghost_coords) > 0)
    if use_ghost:
        aug_coords = np.vstack([spot_coords, ghost_coords]).astype(np.float32)
        aug_tree = build_kdtree(aug_coords)
        # Ghost score mirrors the real boundary spot score
        s_ghost = s_global_spot[boundary_idx]                  # (N_ghost, 2)
        s_aug = np.vstack([s_global_spot, s_ghost]).astype(np.float32)
    else:
        aug_coords = None
        aug_tree = None
        s_aug = None

    new_pos = pos_slice.copy()

    # Collect alpha_rho statistics across all genes in block
    all_alpha_rho = []

    for i, g in enumerate(gene_block):
        lo = int(local_starts[i])
        hi = int(local_starts[i + 1])
        if lo >= hi:
            continue  # gene g has no particles in this block (empty gene)

        pos_g = pos_slice[lo:hi]    # (P_g, 2) — view of slice, read-only here
        mass_g = mass_slice[lo:hi]  # (P_g,)

        # ------------------------------------------------------------------
        # Step 1: interpolate global spot field to gene particles
        # ------------------------------------------------------------------
        # Use species-specific score if available
        if s_global_spot_dict is not None and gene_to_species_idx is not None:
            sp_idx = int(gene_to_species_idx[g])
            s_global_for_gene = s_global_spot_dict[sp_idx]
            # Build species-specific augmented score for ghost spots
            if use_ghost:
                s_ghost_sp = s_global_for_gene[boundary_idx]
                s_aug_sp = np.vstack([s_global_for_gene, s_ghost_sp]).astype(np.float32)
            else:
                s_aug_sp = None
            s_global_part = interpolate_spot_field_to_particles(
                pos_g, s_global_for_gene, spot_tree, h_interp, K_interp, eps,
                aug_tree=aug_tree, s_aug=s_aug_sp,
            )
        else:
            s_global_part = interpolate_spot_field_to_particles(
                pos_g, s_global_spot, spot_tree, h_interp, K_interp, eps,
                aug_tree=aug_tree, s_aug=s_aug,
            )  # (P_g, 2)

        if low_gene_mask[g]:
            # Low-density gene: use global score only.
            # w_gene and s_gene are NEVER computed for this gene.
            s_mix = s_global_part

        else:
            # ------------------------------------------------------------------
            # Step 2a: gene deposition (only P_g particles, real spots only)
            # Use frozen reference if available, otherwise compute from current
            # ------------------------------------------------------------------
            if frozen_w_gene_block is not None and frozen_w_gene_block.get(g) is not None:
                w_gene_g = frozen_w_gene_block[g]  # (M,) frozen from t=0
            else:
                w_gene_g = _deposit_gene_particles(
                    pos_g, mass_g, spot_coords, h_assign, K_assign, eps, M, spot_tree
                )  # (M,)

            # Augmented gene weights for ghost boundary handling
            if use_ghost:
                aug_w_gene_g = np.concatenate(
                    [w_gene_g, w_gene_g[boundary_idx]]
                ).astype(np.float32)
            else:
                aug_w_gene_g = None

            # Decide whether to compute gene score at spot-level (new v16 path)
            # or at particle-level (legacy).  Configured via alpha_rho_config.
            use_spot_level = (
                alpha_rho_config is not None
                and alpha_rho_config.get('gene_score_on_spots', False)
            )

            if use_spot_level:
                # ------------------------------------------------------------------
                # Step 2b-v16: gene score + density on spots, then interpolate
                # to particles (matches global-score workflow).
                # ------------------------------------------------------------------
                s_gene_spot, rho_gene_spot = _gene_score_on_spots(
                    spot_coords, w_gene_g, h_gene, K_score_gene, lam, spot_tree,
                    aug_coords=aug_coords, aug_w_gene_g=aug_w_gene_g, aug_tree=aug_tree,
                    eps=eps,
                )  # (M, 2), (M,)

                # Ghost-augmented gene score for interpolation boundary handling
                if use_ghost:
                    s_gene_ghost = s_gene_spot[boundary_idx]
                    s_gene_aug = np.vstack(
                        [s_gene_spot, s_gene_ghost]
                    ).astype(np.float32)
                else:
                    s_gene_aug = None

                # Interpolate gene score to particles
                s_gene_p = interpolate_spot_field_to_particles(
                    pos_g, s_gene_spot, spot_tree, h_interp, K_interp, eps,
                    aug_tree=aug_tree, s_aug=s_gene_aug,
                )  # (P_g, 2)

                # ------------------------------------------------------------------
                # Step 2c-v16: alpha on spots, then interpolate to particles
                # ------------------------------------------------------------------
                if alpha_gene is not None:
                    alpha_val = float(alpha_gene[g])
                    alpha = np.full(pos_g.shape[0], alpha_val, dtype=np.float32)
                else:
                    if alpha_mode == "ratio":
                        raise ValueError(
                            "alpha_mode='ratio' requires particle-level rho; "
                            "set gene_score_on_spots=False or use alpha_mode='beta'."
                        )
                    beta_val = float(beta) if beta is not None \
                        else float(np.median(rho_gene_spot))
                    alpha_spot_mix = np.clip(
                        rho_gene_spot / (rho_gene_spot + beta_val + eps), 0.0, 1.0
                    ).astype(np.float32)  # (M,)

                    # Kernel-weighted interpolation of alpha to particles
                    K_alpha_mix = min(K_interp, spot_tree.n)
                    idx_am, dists_am = query_knn(spot_tree, pos_g, K_alpha_mix)
                    kernel_am = np.exp(
                        -dists_am * dists_am / (2.0 * h_interp * h_interp)
                    ).astype(np.float32)
                    weights_am = kernel_am / (
                        kernel_am.sum(axis=1, keepdims=True) + eps
                    )
                    alpha = np.sum(
                        weights_am * alpha_spot_mix[idx_am], axis=1
                    ).astype(np.float32)  # (P_g,)
            else:
                # ------------------------------------------------------------------
                # Step 2b-legacy: gene-specific score on P_g particles ONLY
                # ------------------------------------------------------------------
                s_gene_p, rho_p = _gene_score_on_gene_particles(
                    pos_g, w_gene_g, spot_coords, h_gene, K_score_gene, lam, spot_tree,
                    aug_coords=aug_coords, aug_w_gene_g=aug_w_gene_g, aug_tree=aug_tree,
                )  # (P_g, 2), (P_g,)

                # ------------------------------------------------------------------
                # Step 2c-legacy: alpha mixing weight at particle level
                # ------------------------------------------------------------------
                if alpha_gene is not None:
                    alpha_val = float(alpha_gene[g])
                    alpha = np.full(pos_g.shape[0], alpha_val, dtype=np.float32)
                else:
                    alpha = compute_alpha(
                        rho_p, R=None, eps=eps, mode=alpha_mode, beta=beta
                    )  # (P_g,)

            # ------------------------------------------------------------------
            # Step 2d: mix gene-specific and global scores
            # ------------------------------------------------------------------
            s_mix = alpha[:, None] * s_gene_p + (1.0 - alpha[:, None]) * s_global_part

            # ------------------------------------------------------------------
            # Step 2e: density-aware score scaling (alpha_rho)
            # ------------------------------------------------------------------
            if alpha_rho_config is not None:
                alpha_rho_mode = alpha_rho_config.get('mode', 'sigmoid')

                if alpha_rho_mode == 'linear':
                    # Linear mode: simple spot-level ratio, no KDE density estimation
                    # alpha_spot = 1 - w_gene_g / rho_ref
                    # This is O(M) instead of O(M^2) - much faster!

                    rho_ref_percentile = alpha_rho_config.get('rho_ref_percentile', 95)
                    rho_ref_method = alpha_rho_config.get('rho_ref_method', 'p95')
                    nonzero_only = alpha_rho_config.get('nonzero_rho_ref', False)

                    if rho_ref_method == 'max':
                        # max naturally ignores zeros for any gene with mass
                        rho_ref = float(np.max(w_gene_g)) + 1e-8
                    elif nonzero_only:
                        # v16: percentile over nonzero spots only — avoids
                        # rho_ref=0 for sparse genes where zeros dominate
                        nonzero_mask = w_gene_g > 0
                        if nonzero_mask.any():
                            rho_ref = float(np.percentile(
                                w_gene_g[nonzero_mask], rho_ref_percentile
                            )) + 1e-8
                        else:
                            rho_ref = 1e-8  # no mass → alpha_rho ≈ 1 everywhere
                    else:  # legacy: percentile includes zeros
                        rho_ref = float(np.percentile(w_gene_g, rho_ref_percentile)) + 1e-8

                    # Compute alpha at spot level: high mass -> low alpha -> suppressed score
                    alpha_spot = 1.0 - w_gene_g / rho_ref
                    alpha_spot = np.clip(alpha_spot, 0.0, 1.0).astype(np.float32)

                    # Interpolate spot-level alpha to particles using KNN
                    # Use same K and h as score interpolation for consistency
                    K_alpha = min(K_interp, spot_tree.n)
                    idx_alpha, dists_alpha = query_knn(spot_tree, pos_g, K_alpha)
                    kernel_alpha = np.exp(-dists_alpha**2 / (2.0 * h_interp**2)).astype(np.float32)
                    weights_alpha = kernel_alpha / (kernel_alpha.sum(axis=1, keepdims=True) + eps)
                    alpha_rho = np.sum(weights_alpha * alpha_spot[idx_alpha], axis=1).astype(np.float32)

                else:
                    # Sigmoid/exp modes: use legacy KDE-based density estimation
                    use_global = alpha_rho_config.get('use_global_density', True)

                    if use_global:
                        rho_for_alpha = compute_global_gene_density(
                            pos_g, spot_coords, w_gene_g, h_gene
                        )
                    else:
                        rho_for_alpha = rho_p

                    rho_ref = alpha_rho_config.get('rho_ref', None)
                    alpha_rho, _ = compute_alpha_rho(
                        rho_for_alpha,
                        rho_ref=rho_ref,
                        gamma=alpha_rho_config.get('gamma', 2.0),
                        mode=alpha_rho_mode,
                    )

                s_mix = s_mix * alpha_rho[:, None]
                all_alpha_rho.append(alpha_rho)

        # ------------------------------------------------------------------
        # Step 3: Euler-Maruyama update
        # eta_t now multiplies entire displacement (drift + noise)
        # so noise naturally decays with step size
        # ------------------------------------------------------------------
        displacement = s_mix
        if sigma_t > 0.0:
            displacement = displacement + sigma_t * rng.standard_normal(size=pos_g.shape).astype(np.float32)
        new_pos[lo:hi] = (pos_g + eta_t * displacement).astype(np.float32)

    # Compute alpha_rho statistics
    alpha_rho_stats = {}
    if all_alpha_rho:
        all_alpha_rho_arr = np.concatenate(all_alpha_rho)
        alpha_rho_stats = {
            'mean_alpha_rho': float(np.mean(all_alpha_rho_arr)),
            'min_alpha_rho': float(np.min(all_alpha_rho_arr)),
            'max_alpha_rho': float(np.max(all_alpha_rho_arr)),
        }

    return new_pos, alpha_rho_stats


# ---------------------------------------------------------------------------
# Main parallel driver
# ---------------------------------------------------------------------------


def run_denoise_mass_parallel(
    spot_coords: Array,
    counts: sparse.csr_matrix,
    T: int,
    grid_type: str,
    L_assign: int = 1,
    L_score_global: int = 5,
    L_interp: int = 1,
    L_score_gene: int = 3,
    bandwidth_scale: float = 0.05,
    lam: float = 1e-3,
    eps: float = 1e-6,
    eta: float | Array = 0.1,
    sigma: float | Array | None = None,
    alpha_mode: str = "beta",
    beta: Optional[float] = None,
    tau_mass: float = 50.0,
    n_procs: int = 4,
    random_state: Optional[int] = None,
    integerize: bool = False,
    max_dense_output: int = 50_000_000,
    collect_history: bool = False,
    watch_genes: Optional[List[int]] = None,
    progress: bool = True,
    progress_every: int = 1,
    progress_smoothing: float = 0.9,
    progress_prefix: str = "SpaDiff",
    file: Optional[TextIO] = None,
    use_ghost_spots: bool = True,
    K_boundary: int = 7,
    n_layers_ghost: int = 1,
    asymmetry_threshold_factor: float = 0.3,
    use_morans_alpha: bool = False,
    morans_k_neighbors: int = 6,
    morans_sigmoid_k: float = 10.0,
    morans_sigmoid_threshold: float = 0.15,
    frozen_score: bool = False,
    alpha_rho_config: Optional[Dict] = None,
    h_schedule: Optional[Array] = None,
    species_labels: Optional[Array] = None,
) -> Dict[str, object]:
    """Memory-safe BSP mass-particle reverse denoising for spatial transcriptomics.

    Three-phase per-step loop (see module docstring for full description).
    Particles are sorted by gene once at init; each gene-block worker receives
    a contiguous slice so pickling cost is O(P_block), not O(P_total).

    Parameters
    ----------
    spot_coords      : (M,2) float32 — raw spot coordinates (any scale).
    counts           : (M,G) csr_matrix — integer UMI counts.
    T                : number of denoising steps.
    grid_type        : "hex" or "square" (controls K = K_from_layers).
    L_assign         : layers for soft assignment (small; 1-2 recommended).
    L_score_global   : layers for global score on spots (can be large; 5-10).
    L_interp         : layers for interpolating spot field to particles (1-2).
    L_score_gene     : layers for gene-specific score on gene particles (2-4).
    bandwidth_scale  : c in h = c*L for all four bandwidths.
    lam              : density regularisation (avoids div/0).
    eps              : numerical guard for soft normalisation.
    eta              : step size scalar or (T,) array schedule.
    sigma            : noise std scalar or (T,) schedule; None = deterministic.
    alpha_mode       : "beta" (stable) or "ratio".
    beta             : mixing scalar for "beta" mode; None => median(rho_p).
    tau_mass         : total mass threshold below which a gene is "low-density"
                       and skips gene-specific score entirely.
    n_procs          : worker processes; 1 = serial (no Pool overhead).
    random_state     : global RNG seed.
    integerize       : if True, add "denoised_spot_gene_int" via multinomial.
    max_dense_output : if M*G exceeds this, return dict[int->Array] instead.
    collect_history  : if True, record w_global and s_global_spot at every step.
    watch_genes      : list of gene indices for which w_gene is recorded every
                       step (computed from updated positions after Phase C).
                       Ignored if collect_history is False and watch_genes is
                       provided alone — set collect_history=True too.
    progress         : show in-place progress to stderr.
    progress_every   : print interval in steps.
    progress_smoothing: EMA smoothing factor for ETA estimate.
    progress_prefix  : label for progress bar.
    file             : output stream for progress (default: sys.stderr).
    use_ghost_spots  : if True, generate ghost/mirror spots at tissue boundary
                       to symmetrise KDE neighbourhoods (recommended).
    K_boundary       : KNN size for boundary detection in generate_ghost_spots.
    n_layers_ghost   : number of ghost-spot layers to place outside boundary.
    asymmetry_threshold_factor : centroid-asymmetry threshold for boundary
                       detection (fraction of mean inter-spot distance).
    frozen_score     : if True, freeze the KDE reference at t=0. The score
                       field is computed once from initial particle positions
                       and reused at every step. This breaks the self-referential
                       positive feedback that causes mode collapse.
    alpha_rho_config : dict or None — density-aware score scaling configuration.
                       Keys: 'rho_ref' (float or None), 'gamma' (float), 'mode' (str).
                       If provided, scores are scaled by alpha(rho) to suppress
                       high-density regions and prevent over-concentration.
    h_schedule       : (T,) float32 or None — per-step bandwidth schedule for
                       coarse-to-fine annealing. If provided, h_gene and h_global
                       are overridden at each step with h_schedule[t]. This enables
                       large-to-small bandwidth annealing for progressive refinement.
    species_labels   : (G,) string array or None — species label for each gene.
                       When provided, the global score is computed per-species:
                       each gene's particles move according to the global density
                       field built only from genes of the same species. This prevents
                       cross-species contamination in chimeric experiments.

    Returns
    -------
    dict with keys:
      "denoised_spot_gene_float" : (M,G) float32 or dict[int->(M,) float32]
      "spot_total_float"         : (M,) float32
      "scale_info"               : dict with "min", "max", "scale"
      "low_gene_mask"            : (G,) bool
      "particle_pos"             : (P,2) float32 — final positions
      "particle_gene"            : (P,) int64
      "particle_mass"            : (P,) float32
      "denoised_spot_gene_int"   : (M,G) int64   (only if integerize=True)
      "spot_total_int"           : (M,)  int64   (only if integerize=True)
      "history_w_global"         : list of (M,) float32, length T+1
                                   (only if collect_history=True)
                                   index 0 = state before step 0,
                                   index T = final state after all steps.
      "history_s_global_spot"    : list of (M,2) float32, length T
                                   (only if collect_history=True)
      "history_w_gene"           : dict[int -> list of (M,) float32, length T+1]
                                   (only if watch_genes is not None)
      "history_pos"              : list of (P,2) float32, length T+1
                                   (only if collect_history=True)
      "frozen_w_global_0"        : (M,) float32 — frozen reference w_global
                                   (only if frozen_score=True)
      "frozen_w_gene_0"          : dict[int -> (M,) float32] — frozen reference
                                   per-gene weights (only if frozen_score=True)
    """
    # ------------------------------------------------------------------ setup
    if not sparse.isspmatrix_csr(counts):
        counts = counts.tocsr()
    spot_coords = np.asarray(spot_coords, dtype=np.float32)
    spot_coords, scale_info = scale_coords_unit_box(spot_coords)
    M, G = counts.shape

    # Ghost spots for boundary symmetrisation (generated once; coords are fixed)
    if use_ghost_spots and M >= K_boundary + 1:
        ghost_coords, ghost_boundary_idx = generate_ghost_spots(
            spot_coords,
            K_boundary=K_boundary,
            n_layers_ghost=n_layers_ghost,
            asymmetry_threshold_factor=asymmetry_threshold_factor,
        )
        if len(ghost_coords) == 0:
            ghost_coords = None
            ghost_boundary_idx = None
    else:
        ghost_coords = None
        ghost_boundary_idx = None

    # Build particles and sort by gene for O(1) per-gene slicing
    pos0, gene0, mass0 = counts_to_mass_particles(counts, spot_coords)
    sort_idx = np.argsort(gene0, kind="stable")
    particle_pos_s = pos0[sort_idx].copy()          # (P,2) mutable, gene-sorted
    particle_gene_s = gene0[sort_idx]               # (P,) fixed
    particle_mass_s = mass0[sort_idx]               # (P,) fixed

    # CSR-style gene offsets: gene g occupies particle_pos_s[gene_starts[g]:gene_starts[g+1]]
    gene_counts_arr = np.bincount(particle_gene_s, minlength=G)
    gene_starts = np.zeros(G + 1, dtype=np.int64)
    gene_starts[1:] = np.cumsum(gene_counts_arr)

    # Low-density gene mask: genes below tau_mass skip gene-specific score
    gene_mass = np.bincount(
        particle_gene_s, weights=particle_mass_s, minlength=G
    )
    low_gene_mask = gene_mass < tau_mass

    # Moran's I gene-level alpha (computed on original counts, pre-scaling coords)
    _alpha_gene = None
    _morans_I = None
    if use_morans_alpha:
        _alpha_gene, _morans_I = compute_gene_alpha_morans(
            counts, spot_coords,
            k_neighbors=morans_k_neighbors,
            sigmoid_k=morans_sigmoid_k,
            sigmoid_threshold=morans_sigmoid_threshold,
            tau_mass=tau_mass,
        )
        # Override low_gene_mask: genes with alpha_gene == 0 AND low mass
        # still use global score only.  But genes with alpha_gene > 0
        # should always compute gene-specific score even if originally
        # below tau_mass (Moran's I might be high for a moderately sparse gene).
        # We keep low_gene_mask only for genes that have alpha_gene == 0.
        low_gene_mask = low_gene_mask & (_alpha_gene < 1e-6)

    # Species-aware setup: map genes to species, particles to species
    _use_species = species_labels is not None
    _species_list: List[str] = []
    _gene_to_species_idx: Optional[Array] = None
    _particle_species_idx: Optional[Array] = None
    if _use_species:
        species_labels = np.asarray(species_labels)
        _species_list = list(np.unique(species_labels))
        _gene_to_species_idx = np.array(
            [_species_list.index(s) for s in species_labels], dtype=np.int32
        )
        # Map particles to species via their gene
        _particle_species_idx = _gene_to_species_idx[particle_gene_s]

    # Geometry-derived parameters
    K_assign       = K_from_layers(grid_type, L_assign)
    K_score_global = K_from_layers(grid_type, L_score_global)
    K_interp       = K_from_layers(grid_type, L_interp)
    K_score_gene   = K_from_layers(grid_type, L_score_gene)
    h_assign  = bandwidth_from_layers(L_assign,       c=bandwidth_scale)
    h_global  = bandwidth_from_layers(L_score_global, c=bandwidth_scale)
    h_interp  = bandwidth_from_layers(L_interp,       c=bandwidth_scale)
    h_gene    = bandwidth_from_layers(L_score_gene,   c=bandwidth_scale)

    # Step schedules
    eta_sched = np.asarray(eta, dtype=np.float32)
    if eta_sched.ndim == 0:
        eta_sched = np.full(T, float(eta_sched), dtype=np.float32)
    if eta_sched.shape[0] != T:
        raise ValueError("eta schedule must have length T")

    if sigma is None:
        sigma_sched = np.zeros(T, dtype=np.float32)
    else:
        sigma_sched = np.asarray(sigma, dtype=np.float32)
        if sigma_sched.ndim == 0:
            sigma_sched = np.full(T, float(sigma_sched), dtype=np.float32)
        if sigma_sched.shape[0] != T:
            raise ValueError("sigma schedule must have length T")

    # Bandwidth schedule for coarse-to-fine annealing
    h_sched: Optional[Array] = None
    if h_schedule is not None:
        h_sched = np.asarray(h_schedule, dtype=np.float32)
        if h_sched.shape[0] != T:
            raise ValueError("h_schedule must have length T")

    # Chunking for Phase A (by particle) and Phase C (by gene block)
    P = particle_pos_s.shape[0]
    ctx = get_context("spawn")
    particle_chunks = _make_chunks(P, n_procs)
    gene_blocks = _make_gene_blocks(G, gene_starts, n_procs)

    use_pool = n_procs > 1 and len(gene_blocks) > 0

    # ------------------------------------------------------------------ history setup
    # Build a KDTree once here (spot_coords never changes) so we can reuse it
    # both inside the loop for history collection and for the final deposition.
    spot_tree_fixed = build_kdtree(spot_coords)

    # Initialise history containers.  We pre-capture the state BEFORE any
    # update (t=−1 sentinel) so the returned lists are length T+1 for globals
    # and T for per-gene (recorded after each position update).
    _hist_w_global: List[Array] = []
    _hist_s_global: List[Array] = []
    _hist_w_gene: Dict[int, List[Array]] = {g: [] for g in (watch_genes or [])}
    _hist_pos: List[Array] = []
    _hist_alpha_rho: List[Dict] = []

    # Compute initial w_global (Phase A from initial positions)
    _init_partials = [
        soft_deposit_mass_particles_partial(
            particle_pos_s[s:e], particle_mass_s[s:e],
            spot_coords, h_assign, K_assign, eps,
        )
        for s, e in particle_chunks
    ]
    _w0 = np.sum(_init_partials, axis=0, dtype=np.float32)

    # Frozen score setup: compute and cache initial references
    _frozen_w_global_0: Optional[Array] = None
    _frozen_w_gene_0: Dict[int, Array] = {}
    if frozen_score:
        _frozen_w_global_0 = _w0.copy()
        # Pre-compute w_gene_0 for all non-low-density genes
        for g in range(G):
            if low_gene_mask[g]:
                continue  # low-density genes skip gene-specific score
            lo = int(gene_starts[g])
            hi = int(gene_starts[g + 1])
            if lo < hi:
                _frozen_w_gene_0[g] = _deposit_gene_particles(
                    particle_pos_s[lo:hi], particle_mass_s[lo:hi],
                    spot_coords, h_assign, K_assign, eps, M, spot_tree_fixed,
                )

    if collect_history or watch_genes:
        # NOTE: _w0 is identical to what Phase A of step 0 will compute.
        # We do NOT append it to _hist_w_global here to avoid a duplicate;
        # Phase B of step 0 will capture it.  We DO use it for the initial
        # w_gene capture so watch_genes histories start before any update.
        if collect_history:
            _hist_pos.append(particle_pos_s.copy())

        if watch_genes:
            for g in watch_genes:
                lo = int(gene_starts[g]); hi = int(gene_starts[g + 1])
                _hist_w_gene[g].append(
                    _deposit_gene_particles(
                        particle_pos_s[lo:hi], particle_mass_s[lo:hi],
                        spot_coords, h_assign, K_assign, eps, M, spot_tree_fixed,
                    ) if lo < hi else np.zeros(M, dtype=np.float32)
                )

    pm = (
        _ProgressMeter(
            T, progress_every, progress_smoothing, progress_prefix, file or sys.stderr
        )
        if progress
        else None
    )

    # ------------------------------------------------------------------ main loop
    for t in range(T):
        eta_t   = float(eta_sched[t])
        sigma_t = float(sigma_sched[t])

        # Bandwidth for this step (annealing or fixed)
        if h_sched is not None:
            h_gene_t = float(h_sched[t])
            # Scale h_global proportionally (maintain the L_score_global / L_score_gene ratio)
            h_global_t = h_gene_t * (L_score_global / L_score_gene)
        else:
            h_gene_t = h_gene
            h_global_t = h_global

        # -- Phase A: global deposition (parallel by particle chunk)
        if frozen_score:
            # Use frozen reference from t=0
            w_global = _frozen_w_global_0
        else:
            depo_args = [
                (
                    particle_pos_s[s:e],
                    particle_mass_s[s:e],
                    spot_coords,
                    h_assign,
                    K_assign,
                    eps,
                )
                for s, e in particle_chunks
            ]
            if use_pool:
                with ctx.Pool(processes=n_procs) as pool:
                    partials = pool.starmap(soft_deposit_mass_particles_partial, depo_args)
            else:
                partials = [soft_deposit_mass_particles_partial(*a) for a in depo_args]
            w_global = np.sum(partials, axis=0, dtype=np.float32)  # (M,)

        # -- Phase B: global score on spots (serial, memory O(M * K_score_global))
        # When species-aware, compute separate scores per species
        s_global_spot_dict: Optional[Dict[int, Array]] = None
        if _use_species and not frozen_score:
            # Recompute per-species w_global and s_global_spot
            s_global_spot_dict = {}
            for sp_idx, sp_name in enumerate(_species_list):
                # Mask particles of this species
                sp_mask = _particle_species_idx == sp_idx
                # Deposit only this species' particles
                sp_depo_args = [
                    (
                        particle_pos_s[s:e][sp_mask[s:e]],
                        particle_mass_s[s:e][sp_mask[s:e]],
                        spot_coords, h_assign, K_assign, eps,
                    )
                    for s, e in particle_chunks
                    if np.any(sp_mask[s:e])
                ]
                if sp_depo_args:
                    sp_partials = [soft_deposit_mass_particles_partial(*a) for a in sp_depo_args]
                    w_sp = np.sum(sp_partials, axis=0, dtype=np.float32)
                else:
                    w_sp = np.zeros(M, dtype=np.float32)
                s_global_spot_dict[sp_idx] = compute_global_score_on_spots(
                    w_sp, spot_coords, h_global_t, K_score_global, lam, eps,
                    ghost_coords=ghost_coords, ghost_boundary_idx=ghost_boundary_idx,
                )
            # Also compute combined s_global_spot for history/fallback
            s_global_spot = compute_global_score_on_spots(
                w_global, spot_coords, h_global_t, K_score_global, lam, eps,
                ghost_coords=ghost_coords, ghost_boundary_idx=ghost_boundary_idx,
            )
        else:
            s_global_spot = compute_global_score_on_spots(
                w_global, spot_coords, h_global_t, K_score_global, lam, eps,
                ghost_coords=ghost_coords, ghost_boundary_idx=ghost_boundary_idx,
            )  # (M, 2)

        if collect_history:
            # For history, always deposit from current positions for visualization
            if frozen_score:
                _curr_depo_args = [
                    (particle_pos_s[s:e], particle_mass_s[s:e],
                     spot_coords, h_assign, K_assign, eps)
                    for s, e in particle_chunks
                ]
                _curr_partials = [soft_deposit_mass_particles_partial(*a) for a in _curr_depo_args]
                _hist_w_global.append(np.sum(_curr_partials, axis=0, dtype=np.float32))
            else:
                _hist_w_global.append(w_global.copy())
            _hist_s_global.append(s_global_spot.copy())

        # -- Phase C: per-gene-block update (parallel by gene block)
        # Extract frozen w_gene for this block if frozen_score is True
        block_args = []
        for i, gene_block in enumerate(gene_blocks):
            g_min   = gene_block[0]
            g_max   = gene_block[-1]
            p_start = int(gene_starts[g_min])
            p_end   = int(gene_starts[g_max + 1])
            # Local offsets within this block's particle slice
            local_starts = (
                gene_starts[g_min : g_max + 2] - p_start
            ).astype(np.int64)
            seed = (random_state or 0) + t * 9973 + i * 31
            # Extract frozen w_gene dict for genes in this block
            frozen_w_gene_block = None
            if frozen_score:
                frozen_w_gene_block = {g: _frozen_w_gene_0.get(g) for g in gene_block}
            block_args.append(
                (
                    gene_block,
                    particle_pos_s[p_start:p_end].copy(),  # contiguous copy
                    particle_mass_s[p_start:p_end],
                    local_starts,
                    s_global_spot,
                    spot_coords,
                    h_assign,   K_assign,
                    h_interp,   K_interp,
                    h_gene_t,   K_score_gene,
                    lam, eps,
                    low_gene_mask,
                    alpha_mode, beta,
                    eta_t, sigma_t, seed,
                    ghost_coords, ghost_boundary_idx,
                    _alpha_gene,
                    frozen_w_gene_block,
                    alpha_rho_config,
                    s_global_spot_dict,
                    _gene_to_species_idx,
                )
            )

        if use_pool:
            with ctx.Pool(processes=n_procs) as pool:
                block_results = pool.starmap(_gene_block_worker, block_args)
        else:
            block_results = [_gene_block_worker(*a) for a in block_args]

        # Reassemble updated positions into gene-sorted array
        # Also collect alpha_rho statistics from all blocks
        step_alpha_rho_stats = []
        for i, (new_pos_slice, alpha_rho_stats) in enumerate(block_results):
            gene_block = gene_blocks[i]
            p_start = int(gene_starts[gene_block[0]])
            p_end   = int(gene_starts[gene_block[-1] + 1])
            particle_pos_s[p_start:p_end] = new_pos_slice
            if alpha_rho_stats:
                step_alpha_rho_stats.append(alpha_rho_stats)

        # -- History: record per-gene deposition and positions from updated state
        if collect_history:
            _hist_pos.append(particle_pos_s.copy())
            # Aggregate alpha_rho stats across all blocks for this step
            if step_alpha_rho_stats:
                agg_stats = {
                    'mean_alpha_rho': float(np.mean([s['mean_alpha_rho'] for s in step_alpha_rho_stats])),
                    'min_alpha_rho': float(np.min([s['min_alpha_rho'] for s in step_alpha_rho_stats])),
                    'max_alpha_rho': float(np.max([s['max_alpha_rho'] for s in step_alpha_rho_stats])),
                }
                _hist_alpha_rho.append(agg_stats)

        if watch_genes:
            for g in watch_genes:
                lo = int(gene_starts[g]); hi = int(gene_starts[g + 1])
                _hist_w_gene[g].append(
                    _deposit_gene_particles(
                        particle_pos_s[lo:hi], particle_mass_s[lo:hi],
                        spot_coords, h_assign, K_assign, eps, M, spot_tree_fixed,
                    ) if lo < hi else np.zeros(M, dtype=np.float32)
                )

        if pm is not None:
            pm.update(t)

    if pm is not None:
        pm.close()

    # ------------------------------------------------------------------ final deposition
    # spot_tree_fixed was built earlier and is reused here.

    # Global final deposition
    depo_args = [
        (
            particle_pos_s[s:e],
            particle_mass_s[s:e],
            spot_coords,
            h_assign,
            K_assign,
            eps,
        )
        for s, e in particle_chunks
    ]
    if use_pool:
        with ctx.Pool(processes=n_procs) as pool:
            partials = pool.starmap(soft_deposit_mass_particles_partial, depo_args)
    else:
        partials = [soft_deposit_mass_particles_partial(*a) for a in depo_args]
    w_global_final = np.sum(partials, axis=0, dtype=np.float32)  # (M,)

    # Append final global state to history (length becomes T+1)
    if collect_history:
        _hist_w_global.append(w_global_final.copy())

    # Per-gene final deposition (serial loop; spot_tree_fixed reused)
    w_gene_final: Dict[int, Array] = {}
    for g in range(G):
        lo = int(gene_starts[g])
        hi = int(gene_starts[g + 1])
        if lo >= hi:
            continue
        w_gene_final[g] = _deposit_gene_particles(
            particle_pos_s[lo:hi],
            particle_mass_s[lo:hi],
            spot_coords,
            h_assign,
            K_assign,
            eps,
            M,
            spot_tree_fixed,
        )

    # Build output (dense or sparse)
    dense_size = M * G
    if dense_size <= max_dense_output:
        denoised = np.zeros((M, G), dtype=np.float32)
        for g, arr in w_gene_final.items():
            denoised[:, g] = arr
        denoised_out: Dict[int, Array] | Array = denoised
    else:
        denoised_out = w_gene_final

    # Restore original (pre-sort) particle order for output
    unsort_idx = np.argsort(sort_idx)
    out_pos  = particle_pos_s[unsort_idx]
    out_gene = particle_gene_s[unsort_idx]
    out_mass = particle_mass_s[unsort_idx]

    result: Dict[str, object] = {
        "denoised_spot_gene_float": denoised_out,
        "spot_total_float":         w_global_final,
        "scale_info":               scale_info,
        "low_gene_mask":            low_gene_mask,
        "particle_pos":             out_pos.astype(np.float32),
        "particle_gene":            out_gene,
        "particle_mass":            out_mass,
    }

    if _alpha_gene is not None:
        result["alpha_gene"] = _alpha_gene
        result["morans_I"] = _morans_I

    if collect_history:
        result["history_w_global"]      = _hist_w_global       # list, length T+1
        result["history_s_global_spot"] = _hist_s_global        # list, length T
        result["history_pos"]           = _hist_pos             # list, length T+1
        if _hist_alpha_rho:
            result["history_alpha_rho"] = _hist_alpha_rho       # list of dicts, length T
    if watch_genes:
        result["history_w_gene"] = _hist_w_gene                 # dict[g -> list, length T+1]

    if frozen_score:
        result["frozen_w_global_0"] = _frozen_w_global_0
        result["frozen_w_gene_0"]   = _frozen_w_gene_0

    if integerize and isinstance(denoised_out, np.ndarray):
        counts_int, totals_int = float_to_int_counts_multinomial(
            denoised_out, w_global_final, random_state=random_state
        )
        result["denoised_spot_gene_int"] = counts_int
        result["spot_total_int"] = totals_int

    return result


def run_denoise_mass(
    spot_coords: Array,
    counts: sparse.csr_matrix,
    T: int,
    grid_type: str,
    L_assign: int = 1,
    L_score_global: int = 5,
    L_interp: int = 1,
    L_score_gene: int = 3,
    bandwidth_scale: float = 0.05,
    lam: float = 1e-3,
    eps: float = 1e-6,
    eta: float | Array = 0.1,
    sigma: float | Array | None = None,
    alpha_mode: str = "beta",
    beta: Optional[float] = None,
    tau_mass: float = 50.0,
    random_state: Optional[int] = None,
    integerize: bool = False,
    collect_history: bool = False,
    watch_genes: Optional[List[int]] = None,
    progress: bool = True,
    progress_every: int = 1,
    progress_smoothing: float = 0.9,
    progress_prefix: str = "SpaDiff",
    file: Optional[TextIO] = None,
) -> Dict[str, object]:
    """Serial wrapper around run_denoise_mass_parallel (n_procs=1, no Pool overhead).

    Parameters are identical to run_denoise_mass_parallel except n_procs is
    fixed at 1.  Suitable for smaller datasets or environments where
    multiprocessing is unavailable.
    """
    return run_denoise_mass_parallel(
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
        eta=eta,
        sigma=sigma,
        alpha_mode=alpha_mode,
        beta=beta,
        tau_mass=tau_mass,
        n_procs=1,
        random_state=random_state,
        integerize=integerize,
        collect_history=collect_history,
        watch_genes=watch_genes,
        progress=progress,
        progress_every=progress_every,
        progress_smoothing=progress_smoothing,
        progress_prefix=progress_prefix,
        file=file,
    )


# ---------------------------------------------------------------------------
# Toy test: mass conservation + low-gene gating
# ---------------------------------------------------------------------------


def _toy_mass_case(
    M: int = 20,
    G: int = 10,
    T: int = 2,
    tau_mass: float = 5.0,
    random_state: int = 42,
) -> Dict[str, object]:
    """Minimal self-contained toy run for correctness checks.

    Constructs a small synthetic hex-like dataset, runs run_denoise_mass, and
    asserts two invariants:

    1. Mass conservation: sum(w_global_final) ≈ sum(particle_mass) within 0.1%
       relative error.  (Soft assignment is mass-preserving by construction.)

    2. Low-gene gating: low_gene_mask contains both True and False entries,
       confirming that the tau_mass threshold is exercised.

    Parameters
    ----------
    M            : number of spots.
    G            : number of genes.
    T            : denoising steps (keep small for speed).
    tau_mass     : low-density threshold.
    random_state : RNG seed.

    Returns
    -------
    result dict from run_denoise_mass (same structure as the main driver).
    """
    rng = np.random.default_rng(random_state)

    # Simple hex-like grid
    coords = np.column_stack(
        [
            np.arange(M, dtype=np.float32),
            (np.arange(M, dtype=np.float32) % 5) * 0.5,
        ]
    )

    # Half of genes are "low-density" (2 nonzeros each), half are "high-density"
    rows, cols, data = [], [], []
    for g in range(G):
        n_nonzero = 2 if g < G // 2 else max(4, M // 4)
        spots = rng.choice(M, size=n_nonzero, replace=False)
        for m in spots:
            rows.append(int(m))
            cols.append(g)
            data.append(int(rng.integers(1, 10)))

    counts_csr = sparse.csr_matrix(
        (np.array(data, dtype=np.float32), (rows, cols)),
        shape=(M, G),
    )

    result = run_denoise_mass(
        spot_coords=coords,
        counts=counts_csr,
        T=T,
        grid_type="hex",
        L_assign=1,
        L_score_global=2,
        L_interp=1,
        L_score_gene=1,
        tau_mass=tau_mass,
        eta=0.001,          # small step to keep particles near the spot grid
        random_state=random_state,
        progress=False,
    )

    # --- Check 1: mass conservation ---
    total_in  = float(counts_csr.sum())
    total_out = float(result["spot_total_float"].sum())
    rel_err   = abs(total_out - total_in) / (total_in + 1e-8)
    assert rel_err < 1e-3, (
        f"Mass conservation failed: deposited {total_out:.4f} vs "
        f"input {total_in:.4f}  (rel_err={rel_err:.2e})"
    )

    # --- Check 2: low-gene mask is non-trivial ---
    lmask = result["low_gene_mask"]
    assert lmask.any(),  "Expected at least one low-density gene in toy example"
    assert (~lmask).any(), "Expected at least one high-density gene in toy example"

    print(
        f"[_toy_mass_case] PASSED  "
        f"input_mass={total_in:.1f}  deposited={total_out:.4f}  "
        f"rel_err={rel_err:.2e}  "
        f"low_genes={lmask.sum()}/{G}  high_genes={(~lmask).sum()}/{G}"
    )
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
# Matplotlib and imageio are imported lazily so the core algorithm remains
# usable in headless or minimal environments without these packages.


def _viz_imports():
    """Lazy import guard for visualisation dependencies."""
    try:
        import matplotlib.pyplot as _plt
        import matplotlib.colors as _mc
    except ImportError as e:
        raise ImportError("matplotlib is required for visualisation functions.") from e
    try:
        try:
            import imageio.v2 as _iio
        except ImportError:
            import imageio as _iio
    except ImportError as e:
        raise ImportError("imageio is required for GIF generation.") from e
    return _plt, _mc, _iio


def _auto_spot_size(M: int) -> float:
    """Return a visually sensible scatter marker size for M spots.

    Calibrated so spots are clearly visible but don't overlap.
    Typical range: M=500 → s≈80, M=5000 → s≈8.
    """
    return float(np.clip(40_000.0 / M, 3.0, 100.0))


def _global_clim(
    histories: List[Array],
    vmin: Optional[float],
    vmax: Optional[float],
) -> tuple:
    """Compute stable color limits across all frames in a history list."""
    lo = 0.0 if vmin is None else float(vmin)
    if vmax is None:
        hi = max((float(np.asarray(a).max()) for a in histories), default=1.0)
        hi = hi if hi > lo else lo + 1.0
    else:
        hi = float(vmax)
    return lo, hi


def _render_scatter_frame(
    spot_coords: Array,
    values: Array,
    vmin: float,
    vmax: float,
    title: str,
    cmap: str,
    s: float,
    cb_label: str,
    figsize: tuple,
    dpi: int,
) -> Array:
    """Render one scatter frame; return (H, W, 3) uint8 RGB array."""
    _plt, _mc, _iio = _viz_imports()
    from io import BytesIO

    fig, ax = _plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    sc = ax.scatter(
        spot_coords[:, 0], spot_coords[:, 1],
        c=values, s=s,
        cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("x (scaled)", fontsize=8)
    ax.set_ylabel("y (scaled)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white", bbox_inches="tight")
    _plt.close(fig)
    buf.seek(0)
    frame = np.array(_iio.imread(buf))
    return frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame


def _build_weight_gif(
    histories: List[Array],
    spot_coords: Array,
    output_path,
    title_fn,
    cb_label: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    fps: int,
    s: float,
    figsize: tuple,
    dpi: int,
) -> None:
    """Shared GIF builder for make_gene_gif and make_global_gif."""
    _plt, _mc, _iio = _viz_imports()
    from pathlib import Path as _Path

    spot_coords = np.asarray(spot_coords, dtype=np.float32)
    lo, hi = _global_clim(histories, vmin, vmax)
    T_hist  = len(histories) - 1   # last index

    # Diagnostics: print per-frame stats so users can verify frames are changing.
    arrs = [np.asarray(v, dtype=np.float32) for v in histories]
    frame_maxes = [float(a.max()) for a in arrs]
    frame_diffs = [
        float(np.abs(arrs[i + 1] - arrs[i]).max())
        for i in range(len(arrs) - 1)
    ]
    print(f"[viz] {len(arrs)} frames | color range [{lo:.4g}, {hi:.4g}]")
    print(f"[viz] per-frame max:          {[f'{x:.4g}' for x in frame_maxes]}")
    if frame_diffs:
        print(f"[viz] frame-to-frame max |Δ|: {[f'{x:.4g}' for x in frame_diffs]}")
        if max(frame_diffs) < 1e-6:
            print("[viz] WARNING: frames are nearly identical — "
                  "check bandwidth_scale (should be ~0.05, not 1.0) and eta.")

    frames = [
        _render_scatter_frame(
            spot_coords, a,
            lo, hi,
            title_fn(t, T_hist),
            cmap, s, cb_label, figsize, dpi,
        )
        for t, a in enumerate(arrs)
    ]

    out = _Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _iio.mimwrite(str(out), frames, fps=fps, loop=0)
    print(f"[viz] saved {len(frames)}-frame GIF  →  {out}")


def make_gene_gif(
    history_w_gene: List[Array],
    spot_coords: Array,
    gene_index: int,
    output_path,
    gene_name: Optional[str] = None,
    fps: int = 4,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
    s: Optional[float] = None,
    figsize: tuple = (5.5, 4.5),
    dpi: int = 100,
) -> None:
    """Animated GIF of per-gene deposited mass w_gene[g] across denoising steps.

    Spot positions are fixed; only the color (deposited mass) changes.
    Color limits are computed globally over all frames so the animation is
    temporally stable.

    Parameters
    ----------
    history_w_gene : list of (M,) float32
        w_gene[g] at each recorded step.  Obtain by passing
        ``collect_history=True, watch_genes=[g]`` to run_denoise_mass_parallel
        and reading ``result["history_w_gene"][g]``.
    spot_coords    : (M,2) float32  — fixed spot coordinates (scaled [0,1]^2).
    gene_index     : int            — gene index used in the frame title.
    output_path    : str or Path    — destination .gif file.
    gene_name      : str or None    — human-readable gene name for the title.
    fps            : frames per second.
    vmin, vmax     : color limits; None = globally auto-computed.
    cmap           : "magma" works well for sparse single-gene patterns.
    s              : marker size; None = auto-sized for M spots.
    figsize        : (width, height) in inches.
    dpi            : render resolution.
    """
    M = np.asarray(history_w_gene[0]).shape[0]
    label = gene_name if gene_name is not None else f"gene {gene_index}"
    _build_weight_gif(
        histories=history_w_gene,
        spot_coords=spot_coords,
        output_path=output_path,
        title_fn=lambda t, T: f"{label}  —  step {t} / {T}",
        cb_label="deposited mass  w_gene[g]",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fps=fps,
        s=s if s is not None else _auto_spot_size(M),
        figsize=figsize,
        dpi=dpi,
    )


def make_global_gif(
    history_w_global: List[Array],
    spot_coords: Array,
    output_path,
    fps: int = 4,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    s: Optional[float] = None,
    figsize: tuple = (5.5, 4.5),
    dpi: int = 100,
) -> None:
    """Animated GIF of global deposited mass w_global across denoising steps.

    Parameters
    ----------
    history_w_global : list of (M,) float32
        w_global at each recorded step.  Obtain via
        ``collect_history=True`` in run_denoise_mass_parallel and reading
        ``result["history_w_global"]``.  Length is T+1 (initial + T steps).
    spot_coords      : (M,2) float32 — fixed spot coordinates.
    output_path      : str or Path — destination .gif file.
    fps              : frames per second.
    vmin, vmax       : color limits; None = globally auto-computed.
    cmap             : "viridis" works well for smooth total-mass patterns.
    s                : marker size; None = auto.
    figsize          : (width, height) in inches.
    dpi              : render resolution.
    """
    M = np.asarray(history_w_global[0]).shape[0]
    _build_weight_gif(
        histories=history_w_global,
        spot_coords=spot_coords,
        output_path=output_path,
        title_fn=lambda t, T: f"global mass  w_global  —  step {t} / {T}",
        cb_label="total deposited mass  w_global",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fps=fps,
        s=s if s is not None else _auto_spot_size(M),
        figsize=figsize,
        dpi=dpi,
    )


def plot_score_field(
    spot_coords: Array,
    score_field: Array,
    normalize: bool = False,
    subsample: Optional[int] = None,
    ax=None,
    figsize: tuple = (6.0, 5.0),
    color: str = "steelblue",
    scale: Optional[float] = None,
    title: Optional[str] = None,
    seed: int = 0,
) -> tuple:
    """Quiver plot of the score vector field at fixed spot positions.

    Arrows sit at each spot (or a random subsample) and point in the score
    direction.  Arrow **length** encodes the score magnitude; all arrows share
    the same ``color``.

    Parameters
    ----------
    spot_coords  : (M,2) float32 — spot coordinates (scaled [0,1]^2).
    score_field  : (M,2) float32 — score vector per spot, e.g.
                   ``result["history_s_global_spot"][t]`` for step t.
    normalize    : if True all arrows have equal unit length (direction only).
    subsample    : int or None — randomly draw this many spots.
                   Recommended: ~500–800 for M≈5000 to avoid clutter.
    ax           : existing Axes; None = create new figure.
    figsize      : (width, height) used only when ax is None.
    color        : uniform arrow color (default "steelblue").
    scale        : quiver ``scale`` (larger → shorter arrows); None = auto.
    title        : axes title; None = auto.
    seed         : RNG seed for subsampling.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.

    Notes
    -----
    Spatial transcriptomics tips:
    - Use ``subsample=600`` for M≈5000 to keep the quiver readable.
    - ``normalize=True`` shows direction pattern without magnitude distraction.
    - ``normalize=False`` (default) lets arrow length convey the magnitude.
    """
    _plt, _mc, _iio = _viz_imports()

    spot_coords = np.asarray(spot_coords, dtype=np.float32)
    score_field = np.asarray(score_field, dtype=np.float32)
    M = spot_coords.shape[0]

    # Subsampling
    if subsample is not None and subsample < M:
        rng = np.random.default_rng(seed)
        idx = rng.choice(M, size=int(subsample), replace=False)
    else:
        idx = np.arange(M)

    xy  = spot_coords[idx]                          # (N, 2)
    uv  = score_field[idx]                          # (N, 2)
    mag = np.linalg.norm(uv, axis=1)               # (N,) original magnitude

    # Optionally normalise arrow length to unit (direction only)
    uv_plot = uv / np.where(mag > 0, mag, 1.0)[:, None] if normalize else uv

    if ax is None:
        fig, ax = _plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
    else:
        fig = ax.get_figure()

    quiver_kw: Dict = dict(
        angles="xy",
        color=color,
        width=0.003,
        headwidth=4,
        headlength=4,
    )
    if scale is not None:
        quiver_kw["scale"]       = scale
        quiver_kw["scale_units"] = "xy"

    ax.quiver(
        xy[:, 0], xy[:, 1],
        uv_plot[:, 0], uv_plot[:, 1],
        **quiver_kw,
    )

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xlabel("x (scaled)", fontsize=9)
    ax.set_ylabel("y (scaled)", fontsize=9)
    ax.tick_params(labelsize=8)

    if title is not None:
        ax.set_title(title, fontsize=10)
    else:
        mode = "direction only" if normalize else "score vectors"
        ax.set_title(f"Score field ({mode}, {len(idx)}/{M} spots)", fontsize=10)

    fig.tight_layout()
    return fig, ax


def plot_gene_score_field(
    counts,
    spot_coords: Array,
    gene_index: int,
    L_score: int = 5,
    bandwidth_scale: float = 0.05,
    grid_type: str = "hex",
    lam: float = 1e-3,
    eps: float = 1e-8,
    gene_name: Optional[str] = None,
    normalize: bool = False,
    subsample: Optional[int] = None,
    ax=None,
    figsize: tuple = (6.0, 5.0),
    color: str = "steelblue",
    scale: Optional[float] = None,
    title: Optional[str] = None,
    seed: int = 0,
) -> tuple:
    """Standalone gene-specific score field plot from a count matrix.

    Computes the score field for a single gene using only that gene's spot
    counts — no denoising run required.  The spot coordinates must already
    be scaled to [0,1]² (use ``scale_coords_unit_box`` or the ``scale_info``
    returned by ``run_denoise_mass_parallel``).

    Parameters
    ----------
    counts       : (M,) array-like **or** sparse/dense (M, G) matrix.
                   If 2-D, column ``gene_index`` is extracted.
                   Values are used directly as per-spot weights (deposited mass).
    spot_coords  : (M,2) float32 — spot coordinates **already scaled** to [0,1]².
    gene_index   : column index into ``counts`` (ignored if ``counts`` is 1-D).
    L_score      : neighborhood layers for score bandwidth; use the same value
                   as ``L_score_global`` (or ``L_score_gene``) from the run.
    bandwidth_scale : scaling constant; same value as used in the run (default 0.05).
    grid_type    : "hex" or "square".
    lam          : density regularisation (same as in the run).
    eps          : numerical guard.
    gene_name    : human-readable gene name used in the auto-generated title.
    normalize    : if True all arrows have unit length (direction only).
    subsample    : randomly subsample spots for readability (e.g. 600).
    ax           : existing Axes; None = create new figure.
    figsize      : figure size when ax is None.
    color        : uniform arrow color.
    scale        : quiver ``scale``; None = auto.
    title        : axes title override; None = auto.
    seed         : RNG seed for subsampling.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.

    Examples
    --------
    After a denoising run::

        scale_info = res['scale_info']
        coords_s   = (spot_coords - scale_info['min']) / (scale_info['scale'] + 1e-8)
        fig, ax = plot_gene_score_field(
            counts, coords_s, gene_index=0,
            L_score=5, bandwidth_scale=0.05,
            gene_name=genes[0], subsample=600,
        )

    Or completely standalone (no prior run needed)::

        fig, ax = plot_gene_score_field(
            counts, coords_s, gene_index=42,
            L_score=3, bandwidth_scale=0.05,
        )
    """
    spot_coords = np.asarray(spot_coords, dtype=np.float32)

    # Extract gene weight vector
    if hasattr(counts, "ndim") and np.asarray(counts).ndim == 1:
        w_gene = np.asarray(counts, dtype=np.float32)
    elif hasattr(counts, "toarray"):
        # sparse matrix
        col = counts.getcol(gene_index) if hasattr(counts, "getcol") else counts[:, gene_index]
        w_gene = np.asarray(col.todense()).ravel().astype(np.float32)
    else:
        arr = np.asarray(counts)
        w_gene = arr[:, gene_index].astype(np.float32) if arr.ndim == 2 else arr.astype(np.float32)

    # Geometry
    h = bandwidth_from_layers(L_score, c=bandwidth_scale)
    K = K_from_layers(grid_type, L_score)

    # Score field on spots
    s_gene_spot = compute_global_score_on_spots(
        w_gene, spot_coords, h, K, lam, eps
    )  # (M, 2)

    # Auto title
    if title is None:
        label = gene_name if gene_name is not None else f"gene {gene_index}"
        mode  = "direction only" if normalize else "score vectors"
        title = f"Gene score field: {label}  ({mode})"

    return plot_score_field(
        spot_coords=spot_coords,
        score_field=s_gene_spot,
        normalize=normalize,
        subsample=subsample,
        ax=ax,
        figsize=figsize,
        color=color,
        scale=scale,
        title=title,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Geometry sanity checks
# ---------------------------------------------------------------------------


def _sanity_geom_tests() -> bool:
    """Quick unit tests for coordinate scaling and K_from_layers."""
    coords = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
    scaled, info = scale_coords_unit_box(coords)
    assert np.allclose(scaled.min(axis=0), 0.0), "scale min"
    assert np.allclose(scaled.max(axis=0), 1.0), "scale max"
    assert K_from_layers("hex",    0) == 1,  "K hex L=0"
    assert K_from_layers("hex",    1) == 7,  "K hex L=1"
    assert K_from_layers("hex",    2) == 19, "K hex L=2"
    assert K_from_layers("square", 2) == 25, "K square L=2"
    assert math.isclose(bandwidth_from_layers(2, 0.05), 0.1), "bandwidth"
    return True


if __name__ == "__main__":
    _sanity_geom_tests()
    print("Geometry tests passed.")
    _toy_mass_case()
