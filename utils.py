import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from scipy.special import erf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = '/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/datasets'

ACTIVATION_TYPES = (
    nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU, nn.LeakyReLU,
    nn.ELU, nn.SELU, nn.Softmax, nn.LogSoftmax, nn.Hardswish,
    nn.SiLU, nn.Mish,
)

ACTIVATION_FUNCTIONS = {
    'ReLU':       lambda x: np.maximum(x, 0),
    'Sigmoid':    lambda x: 1.0 / (1.0 + np.exp(-x)),
    'Tanh':       lambda x: np.tanh(x),
    'GELU':       lambda x: x * 0.5 * (1.0 + erf(x / np.sqrt(2.0))),
    'LeakyReLU':  lambda x: np.where(x > 0, x, 0.01 * x),
    'ELU':        lambda x: np.where(x > 0, x, np.exp(x) - 1),
    'SELU':       lambda x: 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(x) - 1)),
    'SiLU':       lambda x: x / (1.0 + np.exp(-x)),
    'Mish':       lambda x: x * np.tanh(np.log(1.0 + np.exp(x))),
}

def load_act(path):
    """Load activation dict from .npy file and return it."""
    return np.load(path, allow_pickle=True).item()

################################
#     PROCRUSTES MAIN COMPUTATION
################################

def preprocess_mean_center(H, params):
    """Subtract per-neuron mean."""
    return H - H.mean(axis=0)

def preprocess_mean_center_frobenius(H, params):
    """Subtract per-neuron mean, then normalize by Frobenius norm."""
    H_c = H - H.mean(axis=0)
    return H_c 


# ============================================================
# Postprocessing methods
# ============================================================

# (none needed yet — add here if needed, e.g. low-rank projection, scaling)


# ============================================================
# Tool methods
# ============================================================
# All tools have signature: tool(act_src, act_tgt, T_maps, device, params) -> dict
# They receive the ORIGINAL (unprocessed) activations.

def _procrustes_svd(H_A, H_B, device):
    """
    Compute Procrustes SVD of cross-covariance C = H_A.T @ H_B.
    Returns U (d_A, k), S (k,), Vh (k, d_B) as numpy arrays.
    """
    H_A_t = torch.tensor(H_A, dtype=torch.float64, device=device)
    H_B_t = torch.tensor(H_B, dtype=torch.float64, device=device)
    C = H_A_t.T @ H_B_t
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()

def tool_norm_analysis(act_src, act_tgt, T_maps, device, params):
    """
    Compare Procrustes solutions from full vs subset activations.

    For each layer, computes SVD of the cross-covariance on both the full
    activation sets and the subset (act_src/act_tgt with [:N]).
    Applies preprocessing (e.g. mean centering) if configured.

    Plots per layer:
    - Heatmap of |U_full.T @ U_sub| : alignment between target directions
    - Heatmap of |Vh_full @ Vh_sub.T| : alignment between source directions
    - Singular value spectra: full vs subset overlaid

    Requires params:
    - full_source, full_target: full dataset activation filenames (without suffix)
    """
    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)
    N = params.get('N', None)
    top_k = params.get('top_k', 50)

    # Preprocessing (e.g. mean centering)
    preprocessing = params.get('_preprocessing', None)
    pre_fn = preprocessing['method'] if preprocessing and preprocessing.get('method') else None
    print("PREPROCC",pre_fn)
    pre_params = preprocessing.get('params', {}) if preprocessing else {}

    # Load full dataset activations
    act_dir = params.get('_act_dir')
    suffix = params.get('_suffix')
    full_src_name = params.get('full_source')
    full_tgt_name = params.get('full_target')

    if not (full_src_name and full_tgt_name and act_dir):
        print("  [norm_analysis] Missing full_source/full_target params, skipping")
        return {}

    full_src_path = os.path.join(act_dir, f'{full_src_name}{suffix}.npy')
    full_tgt_path = os.path.join(act_dir, f'{full_tgt_name}{suffix}.npy')
    act_src_full = np.load(full_src_path, allow_pickle=True).item()
    act_tgt_full = np.load(full_tgt_path, allow_pickle=True).item()
    layers_src_full = layer_keys(act_src_full)
    layers_tgt_full = layer_keys(act_tgt_full)

    results = {}
    layer_labels = []
    diag_U_values = []
    diag_Vh_values = []
    eff_rank_90_values = []
    eff_rank_99_values = []
    leverage_gini_values = []
    proj_coverage_values = []

    for i, (ks, kt) in enumerate(zip(layers_src, layers_tgt)):
        for phase in ['pre', 'post']:
            key = f'{ks}_{phase}'
            if T_maps.get(key) is None:
                continue

            use_post = (phase == 'post')

            # Subset activations
            H_src_sub = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt_sub = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]
            if pre_fn:
                H_src_sub = pre_fn(H_src_sub, pre_params)
                H_tgt_sub = pre_fn(H_tgt_sub, pre_params)
            U_sub, S_sub, Vh_sub = _procrustes_svd(H_tgt_sub, H_src_sub, device)

            # Full activations
            ks_f = layers_src_full[i]
            kt_f = layers_tgt_full[i]
            H_src_f = get_activations_array(act_src_full, ks_f, use_post=use_post).astype(np.float64)
            H_tgt_f = get_activations_array(act_tgt_full, kt_f, use_post=use_post).astype(np.float64)
            if pre_fn:
                H_src_f = pre_fn(H_src_f, pre_params)
                H_tgt_f = pre_fn(H_tgt_f, pre_params)
            U_full, S_full, Vh_full = _procrustes_svd(H_tgt_f, H_src_f, device)

            # Limit to top_k directions for readability
            k = min(top_k, U_full.shape[1], U_sub.shape[1], Vh_full.shape[0], Vh_sub.shape[0])

            # Direction alignment matrices
            align_U = np.abs(U_full[:, :k].T @ U_sub[:, :k])
            align_Vh = np.abs(Vh_full[:k] @ Vh_sub[:k].T)

            # Diagonality coefficient weighted by singular values
            w = S_full[:k] / (S_full[:k].sum() + 1e-12)
            diag_U = np.sum(w * np.diag(align_U)**2)
            diag_Vh = np.sum(w * np.diag(align_Vh)**2)

            # Effective rank of subset Procrustes SVD
            cumsum = np.cumsum(S_sub) / (S_sub.sum() + 1e-12)
            eff_rank_90 = int(np.searchsorted(cumsum, 0.9) + 1)
            eff_rank_99 = int(np.searchsorted(cumsum, 0.99) + 1)

            # --- Leverage scores ---
            # For each subset sample, how much it contributes to the top-k
            # full directions. leverage_i = ||U_full[:,:k].T @ h_tgt_i||^2
            # High Gini = few samples dominate the Procrustes solution
            proj_tgt = H_tgt_sub @ U_full[:, :k]   # (N, k)
            proj_src = H_src_sub @ Vh_full[:k].T    # (N, k)
            leverage_tgt = np.sum(proj_tgt**2, axis=1)  # (N,)
            leverage_src = np.sum(proj_src**2, axis=1)  # (N,)
            leverage = leverage_tgt + leverage_src
            leverage_normed = leverage / (leverage.sum() + 1e-12)
            # Gini coefficient of leverage distribution
            sorted_lev = np.sort(leverage_normed)
            n_samples = len(sorted_lev)
            cumulative = np.cumsum(sorted_lev)
            gini = 1 - 2 * np.sum(cumulative) / (n_samples * cumulative[-1] + 1e-12) + 1 / n_samples

            # --- Projection variance / coverage ---
            # For each of the top-k full directions, compute the variance
            # of subset projections. Low variance = that direction is poorly
            # excited by the subset.
            # Coverage = fraction of directions with variance > 1% of max variance
            var_per_dir_tgt = np.var(proj_tgt, axis=0)  # (k,)
            var_per_dir_src = np.var(proj_src, axis=0)  # (k,)
            var_per_dir = var_per_dir_tgt + var_per_dir_src
            var_threshold = 0.01 * var_per_dir.max()
            coverage = np.mean(var_per_dir > var_threshold)

            layer_labels.append(key)
            diag_U_values.append(diag_U)
            diag_Vh_values.append(diag_Vh)
            eff_rank_90_values.append(eff_rank_90)
            eff_rank_99_values.append(eff_rank_99)
            leverage_gini_values.append(gini)
            proj_coverage_values.append(coverage)
            results[key] = {
                'diag_U': float(diag_U), 'diag_Vh': float(diag_Vh),
                'eff_rank_90': eff_rank_90, 'eff_rank_99': eff_rank_99,
                'leverage_gini': float(gini), 'proj_coverage': float(coverage),
            }

            # Per-layer plots: heatmaps + SVs + leverage + projection variance
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'{key}', fontsize=14)

            # Row 1: alignment heatmaps + singular values
            im0 = axes[0, 0].imshow(align_U, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            axes[0, 0].set_title(f'|U_full.T @ U_sub| (target dirs)\ndiag coeff = {diag_U:.4f}')
            axes[0, 0].set_xlabel('subset direction')
            axes[0, 0].set_ylabel('full direction')
            fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

            im1 = axes[0, 1].imshow(align_Vh, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            axes[0, 1].set_title(f'|Vh_full @ Vh_sub.T| (source dirs)\ndiag coeff = {diag_Vh:.4f}')
            axes[0, 1].set_xlabel('subset direction')
            axes[0, 1].set_ylabel('full direction')
            fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

            axes[0, 2].plot(S_full[:k], label='full', linewidth=2)
            axes[0, 2].plot(S_sub[:k], label=f'subset (N={N})', linewidth=2)
            axes[0, 2].set_title('Singular values')
            axes[0, 2].set_xlabel('index')
            axes[0, 2].set_ylabel('singular value')
            axes[0, 2].legend()

            # Row 2: leverage distribution + projection variance + leverage sorted
            axes[1, 0].hist(leverage_normed, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title(f'Leverage distribution\nGini = {gini:.4f}')
            axes[1, 0].set_xlabel('leverage (normalized)')
            axes[1, 0].set_ylabel('count')

            axes[1, 1].bar(range(k), var_per_dir_tgt[:k], alpha=0.6, label='target', color='tab:blue')
            axes[1, 1].bar(range(k), var_per_dir_src[:k], alpha=0.6, label='source', color='tab:orange',
                           bottom=var_per_dir_tgt[:k])
            axes[1, 1].axhline(var_threshold, color='red', linestyle='--', label=f'1% threshold')
            axes[1, 1].set_title(f'Projection variance per direction\ncoverage = {coverage:.4f}')
            axes[1, 1].set_xlabel('full direction index')
            axes[1, 1].set_ylabel('variance')
            axes[1, 1].legend(fontsize=7)

            axes[1, 2].plot(np.sort(leverage_normed)[::-1], linewidth=2, color='steelblue')
            axes[1, 2].set_title('Leverage (sorted descending)')
            axes[1, 2].set_xlabel('sample rank')
            axes[1, 2].set_ylabel('leverage (normalized)')

            plt.tight_layout()
            plt.show()

    # Summary plots
    if layer_labels:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = range(len(layer_labels))

        # Diag coefficients
        axes[0, 0].plot(x, diag_U_values, 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel('diag coeff')
        axes[0, 0].set_title('U diagonality (target dirs) per layer')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(x, diag_Vh_values, 'o-', linewidth=2, markersize=6)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[0, 1].set_ylabel('diag coeff')
        axes[0, 1].set_title('Vh diagonality (source dirs) per layer')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        # Effective rank
        axes[1, 0].plot(x, eff_rank_90_values, 'o-', linewidth=2, markersize=6, label='90% energy')
        axes[1, 0].plot(x, eff_rank_99_values, 's-', linewidth=2, markersize=6, label='99% energy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('effective rank')
        axes[1, 0].set_title('Effective rank per layer')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Leverage Gini + Coverage
        ax_gini = axes[1, 1]
        ax_cov = ax_gini.twinx()
        ax_gini.plot(x, leverage_gini_values, 'o-', linewidth=2, markersize=6, color='tab:red', label='leverage Gini')
        ax_cov.plot(x, proj_coverage_values, 's-', linewidth=2, markersize=6, color='tab:green', label='proj coverage')
        ax_gini.set_xticks(x)
        ax_gini.set_xticklabels(layer_labels, rotation=45, ha='right')
        ax_gini.set_ylabel('Gini coefficient', color='tab:red')
        ax_cov.set_ylabel('coverage', color='tab:green')
        ax_gini.set_title('Leverage Gini & Projection coverage per layer')
        lines1, labels1 = ax_gini.get_legend_handles_labels()
        lines2, labels2 = ax_cov.get_legend_handles_labels()
        ax_gini.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax_gini.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Weighted average angular error metric (excluding Flatten layers)
    if layer_labels:
        non_flatten = [(i, l) for i, l in enumerate(layer_labels) if 'Flatten' not in l]
        if non_flatten:
            indices, labels_nf = zip(*non_flatten)
            # Weight by 1/(layer_depth+1): earlier layers weigh more
            # Extract layer index from label (e.g., 'layer_1_Linear_pre' -> 1)
            depths = []
            for l in labels_nf:
                parts = l.split('_')
                depths.append(int(parts[1]))
            layer_weights = np.array([1.0 / (d + 1) for d in depths])
            layer_weights = layer_weights / layer_weights.sum()

            diag_U_sel = np.array([diag_U_values[i] for i in indices])
            diag_Vh_sel = np.array([diag_Vh_values[i] for i in indices])
            avg_diag = np.sum(layer_weights * (diag_U_sel + diag_Vh_sel) / 2)
            print(f"  Weighted avg angular quality (excl. Flatten): {avg_diag:.4f}")

    return results
def tool_cov_spectrum_dist(act_src, act_tgt, T_maps, device, params):
    """
    Compute weighted covariance spectrum distance between source and target
    activations per layer.
    """
    normalize = params.get('normalize', True)
    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)
    N = params.get('N', None)
    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        for phase in ['pre', 'post']:
            key = f'{ks}_{phase}'
            if T_maps.get(key) is None:
                continue

            use_post = (phase == 'post')
            H_src = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]

            H_src_t = torch.tensor(H_src, dtype=torch.float64, device=device)
            H_tgt_t = torch.tensor(H_tgt, dtype=torch.float64, device=device)

            if normalize:
                H_src_t = H_src_t - H_src_t.mean(dim=0)
                H_tgt_t = H_tgt_t - H_tgt_t.mean(dim=0)
                H_src_t = H_src_t / torch.linalg.norm(H_src_t, 'fro')
                H_tgt_t = H_tgt_t / torch.linalg.norm(H_tgt_t, 'fro')

            S_src = torch.linalg.svdvals(H_src_t.T @ H_src_t).cpu().numpy()
            S_tgt = torch.linalg.svdvals(H_tgt_t.T @ H_tgt_t).cpu().numpy()

            N_s = H_src.shape[0]
            S_src = S_src[:min(N_s, H_src.shape[1])]
            S_tgt = S_tgt[:min(N_s, H_tgt.shape[1])]

            shared = min(len(S_src), len(S_tgt))
            norm_src = S_src[:shared] / S_src[0]
            norm_tgt = S_tgt[:shared] / S_tgt[0]
            results[key] = np.mean(np.abs(norm_src - norm_tgt))

    return results


def tool_sv_sum(act_src, act_tgt, T_maps, device, params):
    """
    Compute sum of singular values of the cross-covariance matrix
    between source and target activations per layer.
    """
    normalize = params.get('normalize', True)
    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)
    N = params.get('N', None)
    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        for phase in ['pre', 'post']:
            key = f'{ks}_{phase}'
            if T_maps.get(key) is None:
                continue

            use_post = (phase == 'post')
            H_src = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]

            if normalize:
                H_src = H_src - H_src.mean(axis=0)
                H_tgt = H_tgt - H_tgt.mean(axis=0)
                H_src = H_src / np.linalg.norm(H_src, 'fro')
                H_tgt = H_tgt / np.linalg.norm(H_tgt, 'fro')

            C = torch.tensor(H_tgt.T @ H_src, dtype=torch.float64, device=device)
            S = torch.linalg.svdvals(C)
            results[key] = S.sum().item()

    return results


################################
#      GVNEDI SCORE FUNCTION
################################

def g_vendi_score(G_proj):
    """
    Compute the G-Vendi diversity score (Eq. 3):

        K = G G^T / |D|          (normalised covariance)
        G-Vendi = exp( − Σ_i  λ_i^K  log λ_i^K )

    where λ_i^K are the eigenvalues of K.

    A higher score means more diverse gradients (= more diverse dataset).

    Args:
        G_proj: (n_samples, d) projected gradient matrix.

    Returns:
        score: float — the G-Vendi diversity score.
        eigenvalues: np.ndarray — sorted eigenvalues of K (for diagnostics).
    """
    n = G_proj.shape[0]
    K = (G_proj @ G_proj.T) / n                # (n, n) kernel matrix

    eigvals = np.linalg.eigvalsh(K)            # real, sorted ascending
    eigvals = eigvals[eigvals > 1e-12]         # drop near-zero eigenvalues
    eigvals = eigvals / eigvals.sum()          # normalise to distribution

    entropy = -np.sum(eigvals * np.log(eigvals))
    score = np.exp(entropy)
    return score, eigvals
    

def balanced_diversity_sampling(G_proj, labels, k_per_class, n_per_class, seed=0):
    """
    Class-balanced higher-diversity sampling.

    For each class, run higher_diversity_sampling on that class's
    projected gradients to pick n_per_class samples.  Guarantees
    exactly n_per_class samples per class in the final subset.

    Args:
        G_proj:      (N, d) projected gradient matrix for the full pool.
        labels:      (N,) class labels for the pool.
        k_per_class: K-means clusters per class.
        n_per_class: samples to pick per class.
        seed:        random seed.

    Returns:
        indices: np.ndarray of selected pool indices.
    """
    classes = np.unique(labels)
    all_indices = []

    for c in classes:
        class_mask = np.where(labels == c)[0]
        G_class = G_proj[class_mask]

        n_pick = min(n_per_class, len(class_mask))
        k = min(k_per_class, n_pick)

        if k < 2:
            rng = np.random.default_rng(seed + int(c))
            picked_local = rng.choice(len(class_mask), size=n_pick, replace=False)
        else:
            picked_local = higher_diversity_sampling(
                G_class, k=k, n_target=n_pick, seed=seed,
            )

        # Map local indices back to pool indices
        all_indices.extend(class_mask[picked_local].tolist())

    return np.array(all_indices)

def higher_diversity_sampling(D, k, n_target, seed=0):
    """
    Algorithm 1 — Higher Diversity Sampling.

    Cluster-guided balanced sampling: K-means on representations,
    then iteratively sample uniformly from random clusters.
    This up-samples sparse clusters and down-samples dense ones.

    Args:
        D:        (|D|, d) data representation matrix (e.g. projected gradients).
        k:        number of K-means clusters.
        n_target: target subset size.
        seed:     random seed.

    Returns:
        S: np.ndarray of selected indices into D.
    """
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_ids = km.fit_predict(D)

    # Build cluster → indices map
    clusters = {i: np.where(cluster_ids == i)[0] for i in range(k)}

    S = []
    batch_size = max(1, int(np.ceil(n_target / k)))
    while len(S) < n_target:
        c = rng.integers(0, k)
        members = clusters[c]
        n_pick = min(batch_size, n_target - len(S), len(members))
        if n_pick > 0:
            picked = rng.choice(members, size=n_pick, replace=False)
            S.extend(picked.tolist())

    return np.array(S[:n_target])


def random_project(G, d=1024, seed=0):
    """
    Reduce gradient dimension from |θ| to *d* via Rademacher random
    projection (Eq. 2):

        g_proj = Π^T g,   Π_ij ~ U({-1, +1})

    This is a Johnson–Lindenstrauss-type projection that approximately
    preserves inner products.

    Args:
        G:    (n_samples, |θ|) gradient matrix.
        d:    target dimension (paper uses 1024).
        seed: random seed for reproducibility.

    Returns:
        G_proj: (n_samples, d) projected gradients.
    """
    rng = np.random.default_rng(seed)
    n_params = G.shape[1]

    # Rademacher matrix: entries ∈ {-1, +1} uniformly
    Pi = rng.choice([-1, 1], size=(n_params, d)).astype(np.float32)

    G_proj = G.astype(np.float32) @ Pi         # (n_samples, d)
    return G_proj

def method_gvendi(model, N, device, params):
    """Extract activations from G-Vendi-selected class-balanced subsets.

    This method returns a list of (activations_dict, filename_suffix) tuples,
    one per subset. The caller saves each one separately.

    params:
        'dataset':       torchvision dataset class
        'mean':          tuple for normalization
        'std':           tuple for normalization
        'proxy_weights': str, path to proxy model weights
        'proxy_hidden_size_1': int
        'proxy_hidden_size_2': int
        'subset_size':   number of samples per subset
        'n_subsets':     number of subsets to generate
        'n_pool':        gradient pool size
        'k_per_class':   K-means clusters per class for diversity sampling
        'proj_dim':      random projection dimension
        'n_classes':     number of classes
        'precomputed_subsets': (optional) list of (indices, gv_score) — auto-populated after first call
    """

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params['mean'], params['std']),
    ])
    dataset = params['dataset'](root=DATA_DIR, train=True, download=True, transform=tf)
    n_classes = params.get('n_classes', 10)
    n_subsets = params.get('n_subsets', 5)
    subset_size = params['subset_size']
    n_per_class = subset_size // n_classes

    precomputed = params.get('precomputed_subsets', None)
    if precomputed is None:
        # Collect gradients and compute subsets
        n_pool = params.get('n_pool', 5000)
        proj_dim = params.get('proj_dim', 1024)
        k_per_class = params.get('k_per_class', 10)
        proxy_model = build_dense_model(params['proxy_hidden_size_1'], params['proxy_hidden_size_2'], USE_RELU)
        proxy_model.load_state_dict(torch.load(params['proxy_weights'], map_location=device))

        pool_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        print(f"  Collecting gradients for first {n_pool} samples...")
        G_all, labels_all = collect_gradients(proxy_model, pool_loader, n_pool, device)
        G_proj = random_project(G_all, d=proj_dim, seed=0)

        precomputed = []
        for sub_idx in range(n_subsets):
            seed = sub_idx * 42 + 7
            indices = balanced_diversity_sampling(
                G_proj, labels_all, k_per_class=k_per_class,
                n_per_class=n_per_class, seed=seed,
            )
            gv_score, _ = g_vendi_score(G_proj[indices])
            sub_labels = labels_all[indices]
            classes, counts = np.unique(sub_labels, return_counts=True)
            print(f"  Subset {sub_idx}: {len(indices)} samples, "
                  f"G-Vendi = {gv_score:.2f}, "
                  f"per-class: {dict(zip(classes.tolist(), counts.tolist()))}")
            precomputed.append((indices, gv_score))

        # Store back so other models can reuse
        params['precomputed_subsets'] = precomputed

    results = []
    for sub_idx, (indices, gv_score) in enumerate(precomputed):
        subset = Subset(dataset, indices.tolist())
        sub_loader = DataLoader(subset, batch_size=256, shuffle=False)
        print(f"  Extracting subset {sub_idx}: {len(indices)} samples, G-Vendi = {gv_score:.2f}, indices={indices.tolist()}")
        activations = extract_all_activations(model, sub_loader, len(indices), device)
        suffix = f'_gvendi{gv_score:.1f}_sub{sub_idx}'
        results.append((activations, suffix))

    return results



def method_full(model, N, device, params):
    """Extract activations by passing dataset samples through the model.
    params: {'dataset': dataset class, 'mean': tuple, 'std': tuple, 'seed': int (optional)}
    """
    seed = params.get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params['mean'], params['std']),
    ])
    ds = params['dataset'](root=DATA_DIR, train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=g)
    return extract_all_activations(model, loader, N, device)

def method_interpolated(model, N, device, params):
    """Extract activations then build interpolated samples.
    params: {'dataset': dataset class, 'mean': tuple, 'std': tuple,
             'n_anchors': int, 'n_interp': int, 'seed': int, 'noise_std': float}
    """
    seed = params.get('seed', 42)
    g = torch.Generator()
    g.manual_seed(seed)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(params['mean'], params['std']),
    ])
    ds = params['dataset'](root=DATA_DIR, train=True, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=g)
    activations = extract_all_activations(model, loader, N, device)
    labels = activations.pop('_labels', None)
    activations = build_interpolated_activations(
        activations, params['n_anchors'], params['n_interp'],
        seed, params.get('noise_std', 0.0),
    )
    if labels is not None:
        activations['_labels'] = labels[:params['n_anchors']]
    return activations


def method_gvendi_interpolated(model, N, device, params):
    """Run method_gvendi, then interpolate each subset's activations.

    params: same as method_gvendi, plus:
        'interp_number': int — number of interpolated pairs to generate per subset
        'seed': int (default 42) — seed for reproducible alpha and pair selection
    """

    interp_number = params['interp_number']
    seed = params.get('seed', 42)

    gvendi_results = method_gvendi(model, N, device, params)

    results = []
    for activations, suffix in gvendi_results:
        labels = activations.pop('_labels', None)
        interp_act = {}
        alphas = []
        
        for layer_name, (pre, post, act_name) in activations.items():
            rng = np.random.default_rng(seed)
            n_samples = pre.shape[0]

            idx_i = rng.integers(0, n_samples, size=interp_number)
            idx_j = rng.integers(0, n_samples, size=interp_number)
            alpha = rng.uniform(0, 1, size=(interp_number, 1))
            alphas.append(alpha)
            print(f"  {layer_name}: pairs (i,j): {list(zip(idx_i.tolist(), idx_j.tolist()))}")

            pre = pre.astype(np.float64)
            pre_interp = alpha * pre[idx_i] + (1 - alpha) * pre[idx_j]
            pre_out = np.concatenate([pre, pre_interp], axis=0)

            post_out = None
            if act_name is not None and act_name in ACTIVATION_FUNCTIONS:
                post_out = ACTIVATION_FUNCTIONS[act_name](pre_out)

            interp_act[layer_name] = (pre_out, post_out, act_name)
            post_shape = post_out.shape if post_out is not None else None
            print(f"  {layer_name}: pre={pre_out.shape}, post={post_shape} "
                  f"(original={n_samples}, interp={interp_number})")
        print(alphas)
        if labels is not None:
            interp_act['_labels'] = labels
        results.append((interp_act, suffix))

    return results


################################
#      
################################
def collect_gradients(model, loader, n_samples, device="cpu"):
    """
    For each of the first *n_samples* (x, y) pairs, compute the
    normalised loss-gradient vector g_θ(x,y) from Eq. (1):

        g_θ(x,y) = −∇ log P(y|x; θ) / ‖−∇ log P(y|x; θ)‖

    Args:
        model:     nn.Module (proxy model, kept frozen).
        loader:    DataLoader yielding (images, labels).
        n_samples: how many samples to collect.
        device:    'cuda' or 'cpu'.

    Returns:
        G: np.ndarray of shape (n_samples, |θ|)  — normalised gradients.
        labels: np.ndarray of shape (n_samples,)  — class labels.
    """
    model.eval().to(device)
    criterion = nn.CrossEntropyLoss()

    gradients = []
    all_labels = []
    collected = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if collected >= n_samples:
                break

            img = images[i:i+1].to(device)
            lbl = labels[i:i+1].to(device)

            model.zero_grad()
            for p in model.parameters():
                p.requires_grad_(True)

            logits = model(img)
            loss = criterion(logits, lbl)
            loss.backward()

            # Flatten all parameter gradients into one vector
            grad_vec = torch.cat([
                p.grad.detach().flatten() for p in model.parameters()
                if p.grad is not None
            ])

            # Normalise (Eq. 1)
            norm = grad_vec.norm()
            if norm > 0:
                grad_vec = grad_vec / norm

            gradients.append(grad_vec.cpu().numpy())
            all_labels.append(labels[i].item())
            collected += 1

        if collected >= n_samples:
            break

    G = np.stack(gradients, axis=0)            # (n_samples, |θ|)
    return G, np.array(all_labels)

def build_dense_model(hidden_size_1, hidden_size_2, use_relu=True):
    return build_more_dense_model(hidden_size_1, hidden_size_2, use_relu=True)
    
def build_small_model(hidden_size_1, hidden_size_2, use_relu=True):
    layers = [
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_size_1),
    ]
    if use_relu:
        layers.append(nn.GELU())
    layers += [
        nn.Linear(hidden_size_1, hidden_size_2)
    ]
    layers += [
        nn.Linear(hidden_size_2, 10),
    ]
    return nn.Sequential(*layers)
    
def build_more_dense_model(hidden_size_1, hidden_size_2, use_relu=True):
    layers = [nn.Flatten()]
    
    act = nn.GELU if use_relu else nn.Identity

    linear_layers = []

    # First two layers
    linear_layers.append((28 * 28, hidden_size_1))
    linear_layers.append((hidden_size_1, hidden_size_2))
    
    # Total linear layers = 7 → last is output
    # So hidden transitions = 6 → we already used 2 → need 4 more
    remaining = 7 - 1 - len(linear_layers)
    
    current_size = hidden_size_2
    
    for _ in range(remaining):
        next_size = max(16, current_size // 2)
        linear_layers.append((current_size, next_size))
        current_size = next_size

    # Build layers with activation
    for in_f, out_f in linear_layers:
        layers.append(nn.Linear(in_f, out_f))
        layers.append(act())

    # Output layer (7th Linear)
    layers.append(nn.Linear(current_size, 10))
    
    return nn.Sequential(*layers)

def extract_all_activations(model, loader, N, device):
    """
    Extract pre-activation and post-activation outputs from all non-activation layers.

    Returns a dict: {layer_name: (pre_act array (N, dim), post_act array (N, dim) or None, act_name or None)}
      - pre_act: output of the layer (before activation function)
      - post_act: output after the activation function, or None if no activation follows
    """
    layers = list(model.children())
    activations = {}
    hooks = []

    for i, layer in enumerate(layers):
        if isinstance(layer, ACTIVATION_TYPES):
            continue

        next_layer = layers[i + 1] if i + 1 < len(layers) else None
        has_act = isinstance(next_layer, ACTIVATION_TYPES)
        act_name = type(next_layer).__name__ if has_act else None

        layer_name = f"layer_{i}_{type(layer).__name__}"

        # Hook on the layer itself (pre-activation)
        def make_pre_hook(name, activation_name):
            def hook(module, input, output):
                if name not in activations:
                    activations[name] = {"pre": [], "post": [], "act_name": activation_name}
                activations[name]["pre"].append(output.detach().cpu().numpy())
            return hook

        hooks.append(layer.register_forward_hook(make_pre_hook(layer_name, act_name)))

        # Hook on the activation layer (post-activation) if it exists
        if has_act:
            def make_post_hook(name):
                def hook(module, input, output):
                    activations[name]["post"].append(output.detach().cpu().numpy())
                return hook

            hooks.append(next_layer.register_forward_hook(make_post_hook(layer_name)))

    model.eval().to(device)
    collected = 0
    all_labels = []
    n_distr = 300
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            model(images)
            all_labels.append(labels.numpy())
            collected += images.size(0)
            if collected >= N:
                break

    all_labels = np.concatenate(all_labels)[:n_distr]
    classes, counts = np.unique(all_labels, return_counts=True)
    print(f"  Class distribution (first {n_distr} samples):")
    for c, n in zip(classes, counts):
        print(f"    class {c}: {n}")
    expected = n_distr / len(classes)
    chi2 = np.sum((counts - expected) ** 2 / expected)
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=len(classes) - 1)
    print(f"  Balance check: chi2={chi2:.2f}, p={p_value:.4f} ({'balanced' if p_value > 0.05 else 'imbalanced'})")

    for h in hooks:
        h.remove()

    result = {}
    for name, data in activations.items():
        pre_arr = np.concatenate(data["pre"], axis=0)[:N].reshape(N, -1)
        if data["post"]:
            post_arr = np.concatenate(data["post"], axis=0)[:N].reshape(N, -1)
        else:
            post_arr = None
        result[name] = (pre_arr, post_arr, data["act_name"])
        post_shape = post_arr.shape if post_arr is not None else None
        print(f"  {name}: pre={pre_arr.shape}, post={post_shape}, activation={data['act_name']}")

    return result

##################
#        MODEL UTILS
#################
def get_activations_array(act_dict, layer_name, use_post):
    """
    Get the right activation array from a layer entry.
    Handles both old format (pre, act_name) and new format (pre, post, act_name).
    """
    entry = act_dict[layer_name]
    if len(entry) == 3:
        pre, post, act_name = entry
        if use_post and post is not None:
            return post
        return pre
    else:
        return entry[0]


def has_activation(act_dict, layer_name):
    """Check if a layer has an activation function after it."""
    entry = act_dict[layer_name]
    if len(entry) == 3:
        return entry[1] is not None
    else:
        return entry[1] is not None and entry[1] != 'None'


def layer_keys(act_dict):
    """Return layer names, excluding metadata keys like _labels."""
    return [k for k in act_dict.keys() if not k.startswith('_')]