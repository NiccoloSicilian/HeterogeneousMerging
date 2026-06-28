import os
os.environ["NO_TORCH_DYNAMO"] = "1"
import time
import numpy as np
import torch
torch._dynamo.config.suppress_errors = True
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.linalg._eigen._svds as _svds_module
from estimators.FM import FM_T
from parameter_config import USE_RELU, COMPUTE_FMAPS

if USE_RELU:
    suffix='_relu'
else:
    suffix = '_norelu'
N =120

n_anchor = 20
anchors = torch.stack([torch.arange(0, n_anchor), torch.arange(0, n_anchor)], 1)

model_dir = '/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging'
act_dir = os.path.join(model_dir, 'activations')

# ==========================================
# Device
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch device: {device}")
if device == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

# ==========================================
# HKS patch — runs on GPU
# ==========================================
import utils.descriptors as _descriptors

def torch_HKS(evals, evects, time_list, scaled=False):
    evals_t  = torch.tensor(np.asarray(evals).flatten(),     dtype=torch.float64).to(device)
    evects_t = torch.tensor(np.asarray(evects),              dtype=torch.float64).to(device)
    t_list_t = torch.tensor(np.asarray(time_list).flatten(), dtype=torch.float64).to(device)
    coefs   = torch.exp(-torch.outer(t_list_t, evals_t))
    natural_HKS = torch.einsum('tk,nk->nt', coefs, evects_t.pow(2))
    if scaled:
        inv_scaling = coefs.sum(1)
        natural_HKS = natural_HKS * (1 / inv_scaling)[None, :]
    return natural_HKS.cpu().numpy()

_descriptors.HKS = torch_HKS

# ==========================================
# Eigensolver patch — runs on GPU
# ==========================================
def torch_svds(A, k=6, **kwargs):
    if sp.issparse(A):
        A_dense = A.toarray()
    else:
        A_dense = np.array(A)
    A_dense = (A_dense + A_dense.T) / 2.0
    A_dense += np.eye(A_dense.shape[0]) * 1e-10
    A_gpu = torch.tensor(A_dense, dtype=torch.float64).to(device)
    eigvals, eigvecs = torch.linalg.eigh(A_gpu)
    eigvals = eigvals.flip(0)[:k]
    eigvecs = eigvecs.flip(1)[:, :k]
    s  = eigvals.clamp(min=0).cpu().numpy()
    u  = eigvecs.cpu().numpy()
    vt = u.T
    return u, s, vt

spla.svds         = torch_svds
_svds_module.svds = torch_svds
import graphlearning.graph as _gl_graph
import scipy.sparse.linalg as _spl
_spl.svds = torch_svds


# ==========================================
# Compute functional maps
# ==========================================
def get_act_array(act_dict, layer_name, use_post=False):
    """Get activation array, handling both old (pre, act_name) and new (pre, post, act_name) format."""
    entry = act_dict[layer_name]
    if len(entry) == 3:
        pre, post, act_name = entry
        if use_post and post is not None:
            return post
        return pre
    return entry[0]


def layer_has_activation(act_dict, layer_name):
    """Check if layer has a post-activation (e.g. ReLU)."""
    entry = act_dict[layer_name]
    if len(entry) == 3:
        return entry[1] is not None
    return False


def compute_single_fmap(x_np, y_np, anchors, N, label,layer_index, pre_or_post,n_eigen):
    """Compute a single functional map between two activation arrays.

    x_np, y_np: activation arrays where the first n_anchor samples are
        paired anchors (matched by index from the same input data).
        The graph is built on all N samples; anchors constrain the map.
    """
    x_np = x_np[:N].astype(np.float64)
    y_np = y_np[:N].astype(np.float64)

    print(f"  {label}: ({x_np.shape}) <-> ({y_np.shape}), anchors: first {anchors.shape[0]}")
    t0 = time.time()
    '''
      layer_0_Flatten (pre)  shape=(5000, 784)  rank=500
    effective_rank=209.4  r90=28  r95=70  r99=199
  layer_1_Linear (pre)  shape=(5000, 1024)  rank=500
    effective_rank=41.6  r90=2  r95=3  r99=13
  layer_1_Linear (post)  shape=(5000, 1024)  rank=1024
    effective_rank=237.4  r90=29  r95=62  r99=189
  layer_3_Linear (pre)  shape=(5000, 512)  rank=500
    effective_rank=45.6  r90=5  r95=7  r99=15
  layer_4_Linear (pre)  shape=(5000, 10)  rank=10
    effective_rank=6.6  r90=4  r95=5  r99=7
'''
    
    
    print("Computing ", n_eigen, " base")
    try:
        fmap = FM_T(
            torch.tensor(x_np, dtype=torch.float64),
            torch.tensor(y_np, dtype=torch.float64),
            anchors,
            transformation='orthogonal',
            num_eigs=min(n_eigen, N - 1),
            graph_algo='knn',
            graph_similarity='angular',
            graph_kernel='distance',
            descriptors=('dist_geod', 'label'),
            k=12,
            n_descr=1,
            compute_gt_map=False,
            refine=True,
        )
        print(f"    FM computed in {time.time()-t0:.1f}s — similarity: {fmap.get_similarity():.4f}")
        print(f"    C shape: {np.array(fmap.C).shape}")
        return fmap
    except Exception as e:
        print(f"    failed: {e}")
        return None


def compute_fmaps(act_src, act_tgt, anchors, N, label,n_eigen):
    """Compute functional maps between two activation dicts, both pre and post activation.

    Anchors are assumed to be the first n_anchor samples in both activation dicts
    (placed there by extract_activations with the 'interpolated' dataset spec).
    """
    src_layers = list(act_src.keys())
    tgt_layers = list(act_tgt.keys())
    n_pairs = min(len(src_layers), len(tgt_layers))

    fmap_dict = {}
    print(f"\n[{label}] Computing functional maps for {n_pairs} layer pairs...")
    t_global = time.time()
    
    for i, (sk, tk) in enumerate(zip(src_layers, tgt_layers)):
        if sk.startswith('_') or tk.startswith('_'):
            continue
        print(f"\n[{i+1}/{n_pairs}] {sk} <-> {tk}")

        # Pre-activation fmap (always)
        x_pre = get_act_array(act_src, sk, use_post=False)
        y_pre = get_act_array(act_tgt, tk, use_post=False)
        fmap_dict[f'{sk}_pre'] = compute_single_fmap(x_pre, y_pre, anchors, N, f"{sk} (pre)",i, 'pre',n_eigen)

        # Post-activation fmap (only if both have activation)
        if layer_has_activation(act_src, sk) and layer_has_activation(act_tgt, tk):
            x_post = get_act_array(act_src, sk, use_post=True)
            y_post = get_act_array(act_tgt, tk, use_post=True)
            fmap_dict[f'{sk}_post'] = compute_single_fmap(x_post, y_post, anchors, N, f"{sk} (post)",i, 'post',n_eigen)
        else:
            fmap_dict[f'{sk}_post'] = None
            print(f"  {sk} (post): None (no activation)")

    print(f"\n[{label}] All layers done in {time.time()-t_global:.1f}s total")
    return fmap_dict


if COMPUTE_FMAPS is not None:
    """
    COMPUTE_FMAPS: list of (source_act, target_act, name) tuples.

    Computes functional maps source -> target per layer.
    Saves the raw fmap objects (with .C, .eigvecs1, .eigvecs2)
    for use with translator_fmap.py.

    Anchors are assumed to be the first n_anchor samples in the activation files
    (placed there by extract_activations with the 'interpolated' dataset spec).
    Both source and target must have the same anchor ordering (same input images,
    same index correspondence).

    Examples:
      ('wide_fashion_interp_activations', 'standard_fashion_interp_activations',
       'fmap_wide_to_std_fashion')

    Filename: fmap_dict_{name}{suffix}.npy
    """
    def get_activations(filename):
        path = os.path.join(act_dir, f'{filename}{suffix}.npy')
        act = np.load(path, allow_pickle=True).item()
        print(f"  Loaded {filename} — keys: {list(act.keys())}")
        return act

    for source_name, target_name, map_name,n_eigen in COMPUTE_FMAPS:
        print(f"\n[{map_name}] Computing functional maps: {source_name} -> {target_name}")
        print(f"  Anchors: first {n_anchor} samples (by index)")
        act_source = get_activations(source_name)
        act_target = get_activations(target_name)

        fmap_dict = compute_fmaps(act_source, act_target, anchors, N, map_name,n_eigen)
        map_name_eigen_n = map_name +"_"+str(n_eigen)
        # Save raw fmap objects
        fmap_path = os.path.join(act_dir, f'fmap_dict_{map_name_eigen_n}{suffix}.npy')
        np.save(fmap_path, fmap_dict)
        print(f"Saved fmap objects to {fmap_path}")

        for key, fmap in fmap_dict.items():
            if fmap is not None:
                print(f"  {key}: C shape {np.array(fmap.C).shape}, eigvecs1 {np.array(fmap.eigvecs1).shape}, eigvecs2 {np.array(fmap.eigvecs2).shape}")

    print(f"\nDone. Computed {len(COMPUTE_FMAPS)} functional maps.")
