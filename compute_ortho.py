import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Core computation
# ============================================================

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


def procrustes(H_A, H_B, device, rank=None):
    """
    Orthogonal Procrustes alignment from space B to space A.
    H_A: (N, d_A), H_B: (N, d_B)
    Returns T of shape (d_A, d_B) such that H_A ≈ H_B @ T.T

    If rank is given, keep only the top-k singular components of the
    cross-covariance before forming T (low-rank approximation).
    """
    H_A_t = torch.tensor(H_A, dtype=torch.float64, device=device)
    H_B_t = torch.tensor(H_B, dtype=torch.float64, device=device)
    C = H_A_t.T @ H_B_t
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    if rank is not None:
        U = U[:, :rank]
        Vh = Vh[:rank, :]
    T = U @ Vh
    return T


def compute_procrustes_maps(act_src, act_tgt, device, N=None, ranks=None,
                            preprocessing=None, postprocessing=None):
    """
    Compute Procrustes orthogonal transformation matrices: source -> target.

    preprocessing: callable(H, params) -> H_processed, applied to activations
                   before computing Procrustes. Or None for raw activations.
    postprocessing: callable(T, params) -> T_processed, applied to T matrix
                    before storing. Or None for raw T.

    Returns: T_maps dict {layer_key: T matrix or None}
    """
    pre_fn = preprocessing['method'] if preprocessing else None
    pre_params = preprocessing.get('params', {}) if preprocessing else {}
    post_fn = postprocessing['method'] if postprocessing else None
    post_params = postprocessing.get('params', {}) if postprocessing else {}

    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)
    n_layers = min(len(layers_src), len(layers_tgt))

    if ranks is None:
        ranks = [None] * n_layers

    assert len(ranks) == n_layers, (
        f"Expected {n_layers} rank values, got {len(ranks)}"
    )

    T_maps = {}
    for i, (ks, kt) in enumerate(zip(layers_src, layers_tgt)):
        r = ranks[i]

        # Pre-activation Procrustes (always computed)
        H_src_pre = get_activations_array(act_src, ks, use_post=False).astype(np.float64)[:N]
        H_tgt_pre = get_activations_array(act_tgt, kt, use_post=False).astype(np.float64)[:N]

        H_src_p = pre_fn(H_src_pre, pre_params) if pre_fn else H_src_pre
        H_tgt_p = pre_fn(H_tgt_pre, pre_params) if pre_fn else H_tgt_pre

        T_pre = procrustes(H_tgt_p, H_src_p, device, rank=r)
        T_pre_np = T_pre.cpu().numpy()
        if post_fn:
            T_pre_np = post_fn(T_pre_np, post_params)
        T_maps[f'{ks}_pre'] = T_pre_np

        rank_str = f' (rank={r})' if r is not None else ''
        print(f"  layer {i} pre:  {ks} ({H_src_pre.shape[1]}) -> {kt} ({H_tgt_pre.shape[1]})  T {T_pre_np.shape}{rank_str}")

        # Post-activation Procrustes (only if activation exists)
        if has_activation(act_src, ks) and has_activation(act_tgt, kt):
            H_src_post = get_activations_array(act_src, ks, use_post=True).astype(np.float64)[:N]
            H_tgt_post = get_activations_array(act_tgt, kt, use_post=True).astype(np.float64)[:N]

            H_src_p = pre_fn(H_src_post, pre_params) if pre_fn else H_src_post
            H_tgt_p = pre_fn(H_tgt_post, pre_params) if pre_fn else H_tgt_post

            T_post = procrustes(H_tgt_p, H_src_p, device, rank=r)
            T_post_np = T_post.cpu().numpy()
            if post_fn:
                T_post_np = post_fn(T_post_np, post_params)
            T_maps[f'{ks}_post'] = T_post_np
            print(f"  layer {i} post: {ks} ({H_src_post.shape[1]}) -> {kt} ({H_tgt_post.shape[1]})  T {T_post_np.shape}{rank_str}")
        else:
            T_maps[f'{ks}_post'] = None
            print(f"  layer {i} post: None (no activation)")

    return T_maps


# ============================================================
# Preprocessing methods
# ============================================================

# ============================================================
# Config-driven runner
# ============================================================
"""
COMPUTE_ORTHO_MAP config format:

COMPUTE_ORTHO_MAP = {
    'preprocessing': {
        'method': callable(H, params) -> H_processed,
        'params': dict,
    },
    'postprocessing': {                          # optional
        'method': callable(T, params) -> T_processed,
        'params': dict,
    },
    'tools': [                                   # list of analysis tools
        {
            'method': callable(act_src, act_tgt, T_maps, device, params) -> dict,
            'params': dict,
            'name': str,                         # display name for summary
        },
        ...
    ],
    'save_dir': str,
    'act_dir': str,
    'to_compute': [
        {
            'source': str (activation filename without suffix),
            'target': str (activation filename without suffix),
            'name': str (output name),
            'N': int or None,
            'ranks': list or None,
        },
        ...
    ],
}
"""

if COMPUTE_ORTHO_MAP is not None:
    cfg = COMPUTE_ORTHO_MAP
    if USE_RELU:
        suffix = '_relu'
    else:
        suffix = '_norelu'

    save_dir = cfg.get('save_dir', '/kaggle/working')
    act_dir = cfg.get('act_dir', os.path.join(save_dir, 'activations'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    preprocessing = cfg.get('preprocessing', None)
    postprocessing = cfg.get('postprocessing', None)
    tools = cfg.get('tools', [])

    def get_activations(filename):
        path = os.path.join(act_dir, f'{filename}{suffix}.npy')
        act = np.load(path, allow_pickle=True).item()
        print(f"  Loaded {filename} — layers: {layer_keys(act)}")
        return act

    all_tool_results = {}  # {map_name: {tool_name: {layer_key: value}}}

    for entry in cfg['to_compute']:
        source_name = entry['source']
        target_name = entry['target']
        map_name = entry['name']
        N_thesus = entry.get('N', None)
        ranks = entry.get('ranks', None)

        rank_str = f' ranks={ranks}' if ranks is not None else ' (full rank)'
        print(f"\n[{map_name}] Computing Procrustes: {source_name} -> {target_name}{rank_str}")
        act_src = get_activations(source_name)
        act_tgt = get_activations(target_name)

        T_maps = compute_procrustes_maps(
            act_src, act_tgt, device, N=N_thesus, ranks=ranks,
            preprocessing=preprocessing, postprocessing=postprocessing,
        )

        out_path = os.path.join(save_dir, f'procrustes_matrices_{map_name}_{str(N_thesus)}{suffix}.npy')
        np.save(out_path, T_maps)
        print(f"  Saved {len([k for k,v in T_maps.items() if v is not None])} matrices to {out_path}")

        # Run tools on original (unprocessed) activations + T_maps
        all_tool_results[map_name] = {}
        for tool_cfg in tools:
            if tool_cfg is None:
                continue
            tool_fn = tool_cfg['method']
            tool_params = dict(tool_cfg.get('params', {}))  # copy to avoid mutation
            tool_params['N'] = N_thesus
            tool_params['_act_dir'] = act_dir
            tool_params['_suffix'] = suffix
            tool_params['_source'] = source_name
            tool_params['_target'] = target_name
            tool_name = tool_cfg.get('name', tool_fn.__name__)
            results = tool_fn(act_src, act_tgt, T_maps, device, tool_params)
            all_tool_results[map_name][tool_name] = results
            # Print per-entry results immediately
            if not results:
                continue
            first_val = next(iter(results.values()), None)
            if isinstance(first_val, dict):
                # Dict-of-dicts format (e.g. norm_analysis)
                for k, v in results.items():
                    parts = [f"{sk}={sv:.4f}" if isinstance(sv, float) else f"{sk}={sv}" for sk, sv in v.items()]
                    print(f"  [{tool_name}] {k}: {', '.join(parts)}")
            else:
                parts = [f"{k} = {v:.6f}" for k, v in results.items()]
                print(f"  [{tool_name}] {', '.join(parts)}, avg = {np.mean(list(results.values())):.6f}")

    # Print tool summaries
    for tool_cfg in tools:
        if tool_cfg is None:
            continue
        tool_name = tool_cfg.get('name', tool_cfg['method'].__name__)
        print(f"\n{'='*60}")
        print(f"  {tool_name} summary")
        print(f"{'='*60}")
        for map_name, tool_results in all_tool_results.items():
            if tool_name in tool_results:
                results = tool_results[tool_name]
                if not results:
                    continue
                first_val = next(iter(results.values()), None)
                if isinstance(first_val, dict):
                    for key, val in results.items():
                        parts = [f"{sk}={sv:.4f}" if isinstance(sv, float) else f"{sk}={sv}" for sk, sv in val.items()]
                        print(f"  {map_name} | {key}: {', '.join(parts)}")
                else:
                    parts = []
                    vals = []
                    for key, val in results.items():
                        parts.append(f"{key} = {val:.6f}")
                        vals.append(val)
                    print(f"  {map_name}: {', '.join(parts)}, avg = {np.mean(vals):.6f}")

    # Print angular error summary across all maps
    print(f"\n{'='*60}")
    print(f"  Angular error summary (weighted avg, excl. Flatten)")
    print(f"{'='*60}")
    for map_name, tool_results in all_tool_results.items():
        for tool_name, results in tool_results.items():
            if not results:
                continue
            first_val = next(iter(results.values()), None)
            if not isinstance(first_val, dict):
                continue
            # Check if results have diag_U/diag_Vh
            if 'diag_U' not in first_val:
                continue
            # Collect per-layer diag values, excluding Flatten
            layer_labels_nf = []
            diag_vals = []
            for key, val in results.items():
                if 'Flatten' in key:
                    continue
                layer_labels_nf.append(key)
                diag_vals.append((val['diag_U'] + val['diag_Vh']) / 2)
            if layer_labels_nf:
                depths = []
                for l in layer_labels_nf:
                    parts = l.split('_')
                    depths.append(int(parts[1]))
                layer_weights = np.array([1.0 / (d + 1) for d in depths])
                layer_weights = layer_weights / layer_weights.sum()
                avg_diag = np.sum(layer_weights * np.array(diag_vals))
                print(f"  {map_name}: {avg_diag:.4f}")

    print(f"\nDone. Computed {len(cfg['to_compute'])} Procrustes maps.")
