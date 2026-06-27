
import numpy as np
import torch

import os
def load_cov_spectrum_data(load_params):
    """
    Load the four activation dicts needed by cov_spectrum_distance.

    Expects in load_params:
        'act_dir':     str — directory containing activation files
        'suffix':      str — e.g. '_relu'
        'source':      str — subset source filename (without suffix)
        'target':      str — subset target filename (without suffix)
        'full_source': str — full dataset source filename (without suffix)
        'full_target': str — full dataset target filename (without suffix)

    Returns:
        dict with 'act_src', 'act_tgt', 'act_src_full', 'act_tgt_full'
    """
    
    act_dir = load_params['act_dir']
    suffix = load_params['suffix']
    print(os.path.join(act_dir, f"{load_params['source']}{suffix}.npy"))
    return {
        'act_src':      load_act(os.path.join(act_dir, f"{load_params['source']}{suffix}.npy")),
        'act_tgt':      load_act(os.path.join(act_dir, f"{load_params['target']}{suffix}.npy")),
        'act_src_full': load_act(os.path.join(act_dir, f"{load_params['full_source']}{suffix}.npy")),
        'act_tgt_full': load_act(os.path.join(act_dir, f"{load_params['full_target']}{suffix}.npy")),
    }


def load_true_point_data(load_params):
    """
    Load pretrained and finetuned activation dicts for the same model/subset.

    Expects in load_params:
        'act_dir':      str — directory containing activation files
        'suffix':       str — e.g. '_relu'
        'pretrained':   str — pretrained activation filename (without suffix)
        'finetuned':    str — finetuned activation filename (without suffix)

    Returns:
        dict with 'act_pre', 'act_ft'
    """
    act_dir = load_params['act_dir']
    suffix = load_params['suffix']
    return {
        'act_pre': load_act(os.path.join(act_dir, f"{load_params['pretrained']}{suffix}.npy")),
        'act_ft':  load_act(os.path.join(act_dir, f"{load_params['finetuned']}{suffix}.npy")),
    }


def avg_true_point_norm(data, experiment_params):
    """
    For each sample, compute ||h_pretrained(x) - h_finetuned(x)|| per layer.
    Returns the average norm across samples for each layer.
    Low values = stable manifold points, high = task-sensitive points.
    """
    print("NOOORM")
    act_pre = data['act_pre']
    act_ft = data['act_ft']

    N = experiment_params.get('N', None)

    layers_pre = layer_keys(act_pre)
    layers_ft = layer_keys(act_ft)

    results = {}

    for kp, kf in zip(layers_pre, layers_ft):
        if 'Flatten' in kp:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_pre, kp):
                continue
            use_post = (phase == 'post')
            H_pre = get_activations_array(act_pre, kp, use_post=use_post).astype(np.float64)[:N]
            H_ft = get_activations_array(act_ft, kf, use_post=use_post).astype(np.float64)[:N]

            norms = np.linalg.norm(H_pre - H_ft, axis=1)
            results[f'{kp}_{phase}'] = float(np.var(norms))

    results['avg'] = sum(results.values()) / len(results)
    return results


def cov_spectrum_distance(data, experiment_params):

    act_src = data['act_src']
    act_tgt = data['act_tgt']

    N = experiment_params.get('N', None)
    normalize = experiment_params.get('normalize', True)
    device = experiment_params.get('device', 'cpu')
    print(N)
    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)

    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        if 'Flatten' in ks:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_src, ks):
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
            results[f'{ks}_{phase}'] = np.mean(np.abs(norm_src - norm_tgt))

    results['avg'] = sum(results.values()) / len(results)
    return results



def cov_spectrum_distance_weighted(data, experiment_params):
    """
    Weighted covariance spectrum distance: weights each spectral difference
    by the average singular value magnitude at that index.
    dist = sum(W_i * |norm_src_i - norm_tgt_i|) / sum(W_i)
    where W_i = (S_src_i + S_tgt_i) / 2
    """
    act_src = data['act_src']
    act_tgt = data['act_tgt']

    N = experiment_params.get('N', None)
    normalize = experiment_params.get('normalize', True)
    device = experiment_params.get('device', 'cpu')

    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)

    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        if 'Flatten' in ks:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_src, ks):
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
            W = (norm_src + norm_tgt) / 2
            results[f'{ks}_{phase}'] = float(np.sum(W * np.abs(norm_src - norm_tgt)) / (np.sum(W) + 1e-12))

    results['avg'] = sum(results.values()) / len(results)
    return results


#################
#           CROSS COV SPECTRUM
###############


def cross_cov_effective_rank(data, experiment_params):
    """
    Effective rank of the cross-covariance C = H_tgt.T @ H_src per layer.
    erank = exp(entropy of normalized singular values of C).
    Reports: erank_value / max_possible_rank  and flags '*' if < 50% of max.
    """
    act_src = data['act_src']
    act_tgt = data['act_tgt']

    N = experiment_params.get('N', None)

    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)

    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        if 'Flatten' in ks:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_src, ks):
                continue
            use_post = (phase == 'post')
            H_src = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]

            C = H_tgt.T @ H_src
            s = np.linalg.svd(C, compute_uv=False)
            s = s[s > 1e-12]
            p = s / s.sum()
            erank = float(np.exp(-np.sum(p * np.log(p))))
            max_rank = min(C.shape)

            results[f'{ks}_{phase}'] = erank

    results['avg'] = sum(results.values()) / len(results)
    return results

###########################
#
##############################


def cross_cov_condition_number(data, experiment_params):
    """
    Condition number (σ_max / σ_min) of the cross-covariance C = H_tgt.T @ H_src per layer.
    High = unstable Procrustes, low = stable.
    """
    act_src = data['act_src']
    act_tgt = data['act_tgt']

    N = experiment_params.get('N', None)

    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)

    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        if 'Flatten' in ks:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_src, ks):
                continue
            use_post = (phase == 'post')
            H_src = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]

            C = H_tgt.T @ H_src
            s = np.linalg.svd(C, compute_uv=False)
            results[f'{ks}_{phase}'] = float(s[0] / (s[-1] + 1e-12))

    results['avg'] = sum(results.values()) / len(results)
    return results
#########################
#
##########################


def cross_cov_tail_energy(data, experiment_params):
    """
    Fraction of singular value energy in the tail of C = H_tgt.T @ H_src.
    tail_energy = sum(S[r:]) / sum(S)
    Requires 'rank' in experiment_params (the truncation cutoff r).
    """
    act_src = data['act_src']
    act_tgt = data['act_tgt']

    N = experiment_params.get('N', None)
    r = experiment_params['rank']

    layers_src = layer_keys(act_src)
    layers_tgt = layer_keys(act_tgt)

    results = {}

    for ks, kt in zip(layers_src, layers_tgt):
        if 'Flatten' in ks:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_src, ks):
                continue
            use_post = (phase == 'post')
            H_src = get_activations_array(act_src, ks, use_post=use_post).astype(np.float64)[:N]
            H_tgt = get_activations_array(act_tgt, kt, use_post=use_post).astype(np.float64)[:N]

            C = H_tgt.T @ H_src
            s = np.linalg.svd(C, compute_uv=False)
            tail = s[r:].sum() / (s.sum() + 1e-12)
            results[f'{ks}_{phase}'] = float(tail)

    results['avg'] = sum(results.values()) / len(results)
    return results

############################
#         ANGUALR ERROR TARGET TASK VECT APPROX TASK VEc
##############################

def load_transport_data(load_params):
    """
    Load Procrustes matrices and model weights for task vector comparison.

    Expects in load_params:
        'save_dir':    str — directory containing Procrustes matrices
        'suffix':      str — e.g. '_relu'
        'ortho_file':  str — Procrustes matrix filename (without suffix)
        'wide_pre':    str — path to wide pretrained model weights
        'wide_ft':     str — path to wide finetuned model weights
        'std_pre':     str — path to standard pretrained model weights
        'std_ft':      str — path to standard finetuned model weights
    """
    suffix = load_params['suffix']
    save_dir = load_params['save_dir']

    T_maps_path = os.path.join(save_dir, f"{load_params['ortho_file']}{suffix}.npy")
    T_maps = np.load(T_maps_path, allow_pickle=True).item()

    device = 'cpu'
    wide_pre = torch.load(load_params['wide_pre'], map_location=device)
    wide_ft = torch.load(load_params['wide_ft'], map_location=device)
    std_pre = torch.load(load_params['std_pre'], map_location=device)
    std_ft = torch.load(load_params['std_ft'], map_location=device)

    src_task_vectors = {name: wide_ft[name].double() - wide_pre[name].double() for name in wide_pre}
    true_task_vectors = {name: std_ft[name].double() - std_pre[name].double() for name in std_pre}

    return {
        'T_maps': T_maps,
        'src_task_vectors': src_task_vectors,
        'true_task_vectors': true_task_vectors,
    }


def task_vector_cosine_sim(data, experiment_params):
    """
    Cosine similarity between true target task vector (τ_true = θ_std_ft - θ_std_pre)
    and transported task vector (τ_transported = T_out @ τ_src @ T_in.T) per layer.
    """
    T_maps = data['T_maps']
    src_tv = data['src_task_vectors']
    true_tv = data['true_task_vectors']

    device = experiment_params.get('device', 'cpu')

    T_tensors = {}
    for key, T_np in T_maps.items():
        if T_np is not None:
            T_tensors[key] = torch.tensor(T_np, dtype=torch.float64, device=device)

    act_layers = [k[:-4] for k in T_maps.keys() if k.endswith('_pre')]

    weight_names = [n for n in src_tv if 'weight' in n]
    bias_names = [n for n in src_tv if 'bias' in n]

    weight_to_layers = {
        weight_names[i]: (act_layers[i], act_layers[i + 1])
        for i in range(len(weight_names))
    }
    bias_to_layers = {
        bias_names[i]: act_layers[i + 1]
        for i in range(len(bias_names))
    }

    def get_T_out(layer_key):
        return T_tensors[f'{layer_key}_pre']

    def get_T_in(layer_key):
        if 'Flatten' in layer_key:
            d = T_tensors[f'{layer_key}_pre'].shape[0]
            return torch.eye(d, dtype=torch.float64, device=device)
        post_key = f'{layer_key}_post'
        if post_key in T_tensors:
            return T_tensors[post_key]
        return T_tensors[f'{layer_key}_pre']

    results = {}

    for tv_name, (in_key, out_key) in weight_to_layers.items():
        tau_src = src_tv[tv_name].to(device).double()
        tau_true = true_tv[tv_name].to(device).double()

        T_out = get_T_out(out_key)
        T_in = get_T_in(in_key)

        d_wide_out, d_wide_in = tau_src.shape
        if T_out.shape[1] != d_wide_out:
            T_out = T_out.T
        if T_in.shape[1] != d_wide_in:
            T_in = T_in.T

        tau_transported = T_out @ tau_src @ T_in.T

        cos = float(torch.nn.functional.cosine_similarity(
            tau_transported.flatten().unsqueeze(0),
            tau_true.flatten().unsqueeze(0)
        ).item())
        results[tv_name] = cos

    for tv_name, out_key in bias_to_layers.items():
        tau_src = src_tv[tv_name].to(device).double()
        tau_true = true_tv[tv_name].to(device).double()

        T_out = get_T_out(out_key)
        d_wide = tau_src.shape[0]
        if T_out.shape[1] != d_wide:
            T_out = T_out.T

        tau_transported = T_out @ tau_src

        cos = float(torch.nn.functional.cosine_similarity(
            tau_transported.flatten().unsqueeze(0),
            tau_true.flatten().unsqueeze(0)
        ).item())
        results[tv_name] = cos

    results['avg'] = sum(results.values()) / len(results)
    return results


def task_vector_weighted_cosine_sim(data, experiment_params):
    """
    SVD-weighted cosine similarity between true and transported task vectors.
    Projects both onto the SVD basis of τ_true, weights each component by
    its singular value, then computes cosine sim in that weighted space.
    This emphasizes alignment along the strongest directions of change.
    """
    T_maps = data['T_maps']
    src_tv = data['src_task_vectors']
    true_tv = data['true_task_vectors']

    device = experiment_params.get('device', 'cpu')

    T_tensors = {}
    for key, T_np in T_maps.items():
        if T_np is not None:
            T_tensors[key] = torch.tensor(T_np, dtype=torch.float64, device=device)

    act_layers = [k[:-4] for k in T_maps.keys() if k.endswith('_pre')]

    weight_names = [n for n in src_tv if 'weight' in n]
    bias_names = [n for n in src_tv if 'bias' in n]

    weight_to_layers = {
        weight_names[i]: (act_layers[i], act_layers[i + 1])
        for i in range(len(weight_names))
    }
    bias_to_layers = {
        bias_names[i]: act_layers[i + 1]
        for i in range(len(bias_names))
    }

    def get_T_out(layer_key):
        return T_tensors[f'{layer_key}_pre']

    def get_T_in(layer_key):
        if 'Flatten' in layer_key:
            d = T_tensors[f'{layer_key}_pre'].shape[0]
            return torch.eye(d, dtype=torch.float64, device=device)
        post_key = f'{layer_key}_post'
        if post_key in T_tensors:
            return T_tensors[post_key]
        return T_tensors[f'{layer_key}_pre']

    results = {}

    for tv_name, (in_key, out_key) in weight_to_layers.items():
        tau_src = src_tv[tv_name].to(device).double()
        tau_true = true_tv[tv_name].to(device).double()

        T_out = get_T_out(out_key)
        T_in = get_T_in(in_key)

        d_wide_out, d_wide_in = tau_src.shape
        if T_out.shape[1] != d_wide_out:
            T_out = T_out.T
        if T_in.shape[1] != d_wide_in:
            T_in = T_in.T

        tau_transported = T_out @ tau_src @ T_in.T

        # SVD of τ_true: U S V^T
        U, S, Vh = torch.linalg.svd(tau_true, full_matrices=False)
        # Project both into SVD basis: coefficients = U^T @ τ @ V
        coeff_true = torch.diag(S)  # S itself (diagonal)
        coeff_trans = U.T @ tau_transported @ Vh.T  # (r, r)
        # Weight by S: emphasize top singular directions
        w_true = (S * S).flatten()  # S_i * S_i
        w_trans = (S.unsqueeze(1) * coeff_trans * S.unsqueeze(0)).flatten()  # S_i * coeff_ij * S_j
        # Simpler: just weight the flattened projections by S
        proj_true = (S.unsqueeze(1) * (U.T @ tau_true @ Vh.T)).flatten()
        proj_trans = (S.unsqueeze(1) * (U.T @ tau_transported @ Vh.T)).flatten()

        cos = float(torch.nn.functional.cosine_similarity(
            proj_trans.unsqueeze(0), proj_true.unsqueeze(0)
        ).item())
        results[tv_name] = cos

    for tv_name, out_key in bias_to_layers.items():
        tau_src = src_tv[tv_name].to(device).double()
        tau_true = true_tv[tv_name].to(device).double()

        T_out = get_T_out(out_key)
        d_wide = tau_src.shape[0]
        if T_out.shape[1] != d_wide:
            T_out = T_out.T

        tau_transported = T_out @ tau_src

        # Weight by |τ_true| per component
        w = tau_true.abs()
        cos = float(torch.nn.functional.cosine_similarity(
            (w * tau_transported).unsqueeze(0),
            (w * tau_true).unsqueeze(0)
        ).item())
        results[tv_name] = cos

    results['avg'] = sum(results.values()) / len(results)
    return results



def cluster_boundary_containment(data, experiment_params):
    """
    Identify finetuned boundary points via KNN, then compute the fraction
    of pretrained activations that are near a finetuned boundary point.

    For each layer:
    1. Fit KNN on finetuned activations. A finetuned point is a boundary
       point if any of its k neighbors belongs to a different class.
    2. Collect boundary points B_ft = {h_ft_i : boundary_score_i > 0}.
    3. Compute a distance threshold: median of pairwise nearest-neighbor
       distances among boundary points themselves.
    4. For each pretrained point h_pre_j, compute
           min_b ||b_ft - h_pre_j||
       If this distance < threshold, the pretrained point is "near the
       finetuned boundary".
    5. Report the fraction of pretrained points near the boundary.

    Expects data keys: 'act_pre' (pretrained), 'act_ft' (finetuned).
    Labels from act_ft['_labels'] (or act_pre['_labels']).
    experiment_params: 'N' (int or None), 'k' (int, default 10),
                       'boundary_min_frac' (float, default 0.0).
    """
    from sklearn.neighbors import NearestNeighbors

    act_pre = data['act_pre']
    act_ft = data['act_ft']

    labels = np.array(act_ft.get('_labels', act_pre.get('_labels', [])))
    N = experiment_params.get('N', None)
    k = experiment_params.get('k', 10)
    boundary_min_frac = experiment_params.get('boundary_min_frac', 0.0)

    layers_pre = layer_keys(act_pre)
    layers_ft = layer_keys(act_ft)

    results = {}

    for kp, kf in zip(layers_pre, layers_ft):
        if 'Flatten' in kp:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_ft, kf):
                continue
            use_post = (phase == 'post')

            H_ft = get_activations_array(act_ft, kf, use_post=use_post).astype(np.float64)
            H_pre = get_activations_array(act_pre, kp, use_post=use_post).astype(np.float64)
            labs = labels.copy()

            if N is not None:
                H_ft = H_ft
                H_pre = H_pre
                labs = labs

            # Step 1: find boundary points in finetuned space via KNN
            nn_ft = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn_ft.fit(H_ft)
            _, indices = nn_ft.kneighbors(H_ft)
            neighbor_labels = labs[indices[:, 1:]]  # exclude self
            boundary_scores = (neighbor_labels != labs[:, None]).mean(axis=1)
            boundary_mask = boundary_scores > boundary_min_frac
            B_ft = H_ft[boundary_mask]
            n_boundary = int(boundary_mask.sum())

            if n_boundary > 1:
                # Step 2: threshold = median NN distance among pretrained points
                nn_pre_self = NearestNeighbors(n_neighbors=2, metric='euclidean')
                nn_pre_self.fit(H_pre)
                pre_dists, _ = nn_pre_self.kneighbors(H_pre)
                threshold = float(np.median(pre_dists[:, 1]))

                # Step 3: for each pretrained point, dist to nearest boundary point
                nn_boundary = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_boundary.fit(B_ft)
                dists_pre, _ = nn_boundary.kneighbors(H_pre)
                dists_pre = dists_pre.ravel()
                frac_near = float((dists_pre < threshold).mean())

                # Step 4: boundary coverage — for each boundary point,
                # dist to nearest pretrained point
                nn_pre = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_pre.fit(H_pre)
                dists_cov, _ = nn_pre.kneighbors(B_ft)
                dists_cov = dists_cov.ravel()
                frac_covered = float((dists_cov < threshold).mean())
            else:
                frac_near = float('nan')
                frac_covered = float('nan')
                threshold = float('nan')

            key = f'{kp}_{phase}'
            results[key] = frac_near
            results[f'{key}_cov'] = frac_covered
            print(f"  {key}: frac_near={frac_near:.4f}  "
                  f"frac_covered={frac_covered:.4f}  "
                  f"threshold={threshold:.4f}  "
                  f"n_boundary={n_boundary}/{len(labs)}  "
                  f"boundary_min_frac={boundary_min_frac}")

    near_vals = [v for k, v in results.items() if not k.endswith('_cov')]
    cov_vals = [v for k, v in results.items() if k.endswith('_cov')]
    results['avg'] = sum(near_vals) / len(near_vals)
    results['avg_cov'] = sum(cov_vals) / len(cov_vals)
    return results


def load_paired_boundary_data(load_params):
    """
    Load wide and standard pretrained + finetuned activations for paired
    boundary analysis.

    Expects in load_params:
        'act_dir':       str
        'suffix':        str
        'wide_pre':      str — wide pretrained activation filename
        'wide_ft':       str — wide finetuned activation filename
        'std_pre':       str — standard pretrained activation filename
        'std_ft':        str — standard finetuned activation filename
    """
    act_dir = load_params['act_dir']
    suffix = load_params['suffix']
    return {
        'act_wide_pre': load_act(os.path.join(act_dir, f"{load_params['wide_pre']}{suffix}.npy")),
        'act_wide_ft':  load_act(os.path.join(act_dir, f"{load_params['wide_ft']}{suffix}.npy")),
        'act_std_pre':  load_act(os.path.join(act_dir, f"{load_params['std_pre']}{suffix}.npy")),
        'act_std_ft':   load_act(os.path.join(act_dir, f"{load_params['std_ft']}{suffix}.npy")),
    }


def paired_boundary_containment(data, experiment_params):
    """
    For each sample, check if BOTH the wide and standard pretrained
    activations are near their respective finetuned boundary points.

    For each layer:
    1. Find boundary points in wide_ft and std_ft via KNN.
    2. For each sample i, compute:
         d_wide_i = min_b ||b_wide_ft - h_wide_pre_i||
         d_std_i  = min_b ||b_std_ft  - h_std_pre_i||
    3. A sample is "paired boundary" if both d_wide_i < threshold_wide
       AND d_std_i < threshold_std.
    4. Report: frac_both, frac_wide_only, frac_std_only, frac_neither.

    experiment_params: 'k' (int, default 10), 'boundary_min_frac' (float, default 0.0).
    """
    from sklearn.neighbors import NearestNeighbors

    act_wide_pre = data['act_wide_pre']
    act_wide_ft = data['act_wide_ft']
    act_std_pre = data['act_std_pre']
    act_std_ft = data['act_std_ft']

    labels = np.array(act_wide_ft.get('_labels', act_wide_pre.get('_labels', [])))
    k = experiment_params.get('k', 10)
    boundary_min_frac = experiment_params.get('boundary_min_frac', 0.0)

    layers_wp = layer_keys(act_wide_pre)
    layers_wf = layer_keys(act_wide_ft)
    layers_sp = layer_keys(act_std_pre)
    layers_sf = layer_keys(act_std_ft)

    results = {}

    for kwp, kwf, ksp, ksf in zip(layers_wp, layers_wf, layers_sp, layers_sf):
        if 'Flatten' in kwp:
            continue
        for phase in ['pre', 'post']:
            if phase == 'post' and not has_activation(act_wide_ft, kwf):
                continue
            use_post = (phase == 'post')

            H_wf = get_activations_array(act_wide_ft, kwf, use_post=use_post).astype(np.float64)
            H_wp = get_activations_array(act_wide_pre, kwp, use_post=use_post).astype(np.float64)
            H_sf = get_activations_array(act_std_ft, ksf, use_post=use_post).astype(np.float64)
            H_sp = get_activations_array(act_std_pre, ksp, use_post=use_post).astype(np.float64)
            labs = labels.copy()

            # --- Wide boundary ---
            nn_wf = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn_wf.fit(H_wf)
            _, idx_w = nn_wf.kneighbors(H_wf)
            bnd_w = (labs[idx_w[:, 1:]] != labs[:, None]).mean(axis=1) > boundary_min_frac
            B_wf = H_wf[bnd_w]

            # Wide threshold: median NN distance among wide pretrained
            nn_wp = NearestNeighbors(n_neighbors=2, metric='euclidean')
            nn_wp.fit(H_wp)
            wp_dists, _ = nn_wp.kneighbors(H_wp)
            thresh_w = float(np.median(wp_dists[:, 1]))

            # --- Standard boundary ---
            nn_sf = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn_sf.fit(H_sf)
            _, idx_s = nn_sf.kneighbors(H_sf)
            bnd_s = (labs[idx_s[:, 1:]] != labs[:, None]).mean(axis=1) > boundary_min_frac
            B_sf = H_sf[bnd_s]

            # Standard threshold: median NN distance among std pretrained
            nn_sp = NearestNeighbors(n_neighbors=2, metric='euclidean')
            nn_sp.fit(H_sp)
            sp_dists, _ = nn_sp.kneighbors(H_sp)
            thresh_s = float(np.median(sp_dists[:, 1]))

            # --- Per-sample distances to boundary ---
            if bnd_w.sum() > 1 and bnd_s.sum() > 1:
                nn_bw = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_bw.fit(B_wf)
                d_w, _ = nn_bw.kneighbors(H_wp)
                near_w = (d_w.ravel() < thresh_w)

                nn_bs = NearestNeighbors(n_neighbors=1, metric='euclidean')
                nn_bs.fit(B_sf)
                d_s, _ = nn_bs.kneighbors(H_sp)
                near_s = (d_s.ravel() < thresh_s)

                frac_both = float((near_w & near_s).mean())
                frac_wide_only = float((near_w & ~near_s).mean())
                frac_std_only = float((~near_w & near_s).mean())
                frac_neither = float((~near_w & ~near_s).mean())
            else:
                frac_both = frac_wide_only = frac_std_only = frac_neither = float('nan')

            key = f'{kwp}_{phase}'
            results[f'{key}_both'] = frac_both
            results[f'{key}_w_only'] = frac_wide_only
            results[f'{key}_s_only'] = frac_std_only
            results[f'{key}_neither'] = frac_neither
            print(f"  {key}: both={frac_both:.4f}  w_only={frac_wide_only:.4f}  "
                  f"s_only={frac_std_only:.4f}  neither={frac_neither:.4f}  "
                  f"n_bnd_w={bnd_w.sum()}  n_bnd_s={bnd_s.sum()}")

    both_vals = [v for k, v in results.items() if k.endswith('_both')]
    results['avg_both'] = sum(both_vals) / len(both_vals)
    results['avg'] = results['avg_both']
    return results

