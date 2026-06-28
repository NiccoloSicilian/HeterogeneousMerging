import os
import numpy as np

from parameter_config import USE_RELU, EXTRACT_FMAP_ORTHO

"""
EXTRACT_FMAP_ORTHO: list of (fmap_dict_file, output_name) tuples.

Loads fmap objects from compute_fmaps, extracts the T matrices,
and saves them in the same format as compute_ortho (procrustes_matrices_*).

fmap.T shape: (d_src, d_tgt) — H_src @ T ≈ H_tgt
Procrustes convention: (d_tgt, d_src) — H_tgt ≈ H_src @ T.T

So we save fmap.T transposed.

Examples:
  ('fmap_dict_fmap_wide_to_std_interp_fmnist_anchor', 'fmap_wide_to_std_interp_fashion')

Output: procrustes_matrices_{output_name}{suffix}.npy
"""

if USE_RELU:
    suffix = '_relu'
else:
    suffix = '_norelu'

save_dir = '/kaggle/working'
act_dir = os.path.join(save_dir, 'activations')

def low_rank_truncate(T, rank):
    """Low-rank approximation of orthogonal matrix T via SVD truncation."""
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    U = U[:, :rank]
    Vh = Vh[:rank, :]
    return U @ Vh


if EXTRACT_FMAP_ORTHO is not None:
    for entry in EXTRACT_FMAP_ORTHO:
        if len(entry) == 3:
            fmap_file, output_name, ranks = entry
        else:
            fmap_file, output_name = entry
            ranks = None

        path = os.path.join(act_dir, f'{fmap_file}{suffix}.npy')
        if not os.path.exists(path):
            print(f"Skipping {output_name}: {path} not found")
            continue

        fmap_dict = np.load(path, allow_pickle=True).item()
        print(f"\nLoaded {fmap_file}")

        T_maps = {}
        layer_idx = 0
        for key, fmap in fmap_dict.items():
            if fmap is not None:
                # fmap.T: (d_src, d_tgt) -> transpose to (d_tgt, d_src)
                T_np = np.array(fmap.T.cpu()).T
                if ranks is not None and layer_idx < len(ranks) and ranks[layer_idx] is not None:
                    rank = ranks[layer_idx]
                    T_np = low_rank_truncate(T_np, rank)
                    print(f"  {key}: T shape {T_np.shape} (low-rank {rank})")
                else:
                    print(f"  {key}: T shape {T_np.shape}")
                T_maps[key] = T_np
                layer_idx += 1
            else:
                T_maps[key] = None
                print(f"  {key}: None")

        out_path = os.path.join(save_dir, f'procrustes_matrices_{output_name}{suffix}.npy')
        np.save(out_path, T_maps)
        print(f"Saved to {out_path}")

    print(f"\nDone. Extracted {len(EXTRACT_FMAP_ORTHO)} fmap T matrices.")
