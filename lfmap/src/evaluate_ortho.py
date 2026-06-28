import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from parameter_config import (
    TRANSPORT_ORTHO, USE_RELU,
    MNIST_STD_MODEL, MNIST_WIDE_MODEL,
    FMNIST_STD_MODEL, FMNIST_WIDE_MODEL,
)
from utils import build_dense_model, DATA_DIR



def evaluate(model, loader, device, name):
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total * 100
    print(f"  {name}: {acc:.2f}%")
    return acc


def apply_task_vectors(standard_model, results, std_params, alpha, device):
    stitched_model  = copy.deepcopy(standard_model)
    stitched_params = dict(stitched_model.named_parameters())
    for tv_name, tao_std in results.items():
        if tv_name not in std_params:
            continue
        std_w     = std_params[tv_name]
        tao_std_f = tao_std.float()
        if std_w.shape != tao_std_f.shape:
            continue
        stitched_params[tv_name].data.copy_((std_w.to(device) + alpha * tao_std_f).cpu())
    return stitched_model


def alpha_search(standard_model, results, std_params, eval_loader, device, label):
    alpha_candidates = np.arange(1.0, 8.0, 0.1).tolist()
    print(f"\n--- Alpha search [{label}] (FashionMNIST accuracy) ---")
    best_alpha    = None
    best_acc      = -1
    alpha_results = {}
    for alpha in alpha_candidates:
        model_alpha = apply_task_vectors(standard_model, results, std_params, alpha, device)
        acc = evaluate(model_alpha, eval_loader, device, f"  alpha={alpha:.2f}")
        alpha_results[alpha] = acc
        if acc > best_acc:
            best_acc   = acc
            best_alpha = alpha
    print(f"\n[{label}] Best alpha: {best_alpha}  ->  acc: {best_acc:.2f}%")
    return best_alpha, best_acc, alpha_results


def transport_with_ortho(task_vectors, weight_to_layers, bias_to_layers, T_maps, device):
    """
    Transport task vectors using pre-computed orthogonal transformation matrices.

    T_maps: dict from raw_ortho_trans.py with keys '{layer_name}_pre' and '{layer_name}_post'.
      - T_pre: Procrustes on pre-activation (what W produces) — used as T_out
      - T_post: Procrustes on post-activation (what next W receives) — used as T_in
      - If T_post is None (no activation), T_pre is used for both.

    For weights:  tau_std = T_out @ tau_wide @ T_in.T
      - T_out = T_pre of the output layer (pre-activation alignment)
      - T_in  = T_post of the input layer if activation exists, else T_pre
    For biases:   tau_std = T_out @ tau_bias
    """
    # Build T tensors, resolving pre/post for each layer
    T_tensors = {}
    for key, T_np in T_maps.items():
        if T_np is not None:
            T_tensors[key] = torch.tensor(T_np, dtype=torch.float64, device=device)

    def get_T_out(layer_key):
        """T_out = pre-activation alignment (what W directly produces)."""
        return T_tensors[f'{layer_key}_pre']

    def get_T_in(layer_key):
        """T_in = post-activation if available (what W receives as input), else pre.
        For Flatten layers, return identity (no learned params, same input space)."""
        if 'Flatten' in layer_key:
            d = T_tensors[f'{layer_key}_pre'].shape[0]
            return torch.eye(d, dtype=torch.float64, device=device)
        post_key = f'{layer_key}_post'
        if post_key in T_tensors:
            return T_tensors[post_key]
        return T_tensors[f'{layer_key}_pre']

    results = {}

    print("Transporting Weights...")
    for tv_name, (in_key, out_key) in weight_to_layers.items():
        tau_wide = task_vectors[tv_name].to(device).double()
        T_out = get_T_out(out_key)
        T_in  = get_T_in(in_key)
        print(f"    T_out key={out_key} shape={tuple(T_out.shape)}, T_in key={in_key} shape={tuple(T_in.shape)}, tau={tuple(tau_wide.shape)}")
        # T from Procrustes: H_A ≈ H_B @ T.T, T shape (d_A, d_B)
        # If computed as procrustes(H_std, H_wide): T=(d_std, d_wide), H_std ≈ H_wide @ T.T
        #   transport: tau_std = T @ tau_wide @ T_in.T  — T is already (d_std, d_wide) ✓
        # If computed as procrustes(H_wide, H_std): T=(d_wide, d_std), H_wide ≈ H_std @ T.T
        #   transport: tau_std = T.T @ tau_wide @ T_in  — need to transpose
        # Detect: tau_wide is (d_wide_out, d_wide_in)
        d_wide_out, d_wide_in = tau_wide.shape
        if T_out.shape[1] == d_wide_out:
            # T_out is (d_std, d_wide) — use as-is
            pass
        else:
            # T_out is (d_wide, d_std) — transpose to get (d_std, d_wide)
            T_out = T_out.T
        if T_in.shape[1] == d_wide_in:
            pass
        else:
            T_in = T_in.T
        tao_std = T_out @ tau_wide @ T_in.T
        results[tv_name] = tao_std.float()
        print(f"  {tv_name}: {tuple(tau_wide.shape)} -> {tuple(tao_std.shape)}")

    print("Transporting Biases...")
    for tv_name, out_key in bias_to_layers.items():
        tau_bias = task_vectors[tv_name].to(device).double()
        T_out = get_T_out(out_key)
        d_wide = tau_bias.shape[0]
        if T_out.shape[1] == d_wide:
            pass
        else:
            T_out = T_out.T
        tao_std = T_out @ tau_bias
        results[tv_name] = tao_std.float()

    print(f"Transported {len(results)} parameters.")
    return results


if TRANSPORT_ORTHO is not None:
    """
    TRANSPORT_ORTHO: list of (ortho_map_file, name) string tuples.

    ortho_map_file: Procrustes matrices from raw_ortho_trans.py (with _pre/_post keys).
                    Loaded from save_dir with suffix.

    Examples:
      ('procrustes_matrices_wide_to_std', 'ortho_wide_to_std')
    """
    use_relu = USE_RELU
    suffix   = '_relu' if use_relu else '_norelu'

    save_dir = '/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/'
    load_dir = '/leonardo_scratch/fast/IscrC_eff-SAM2/HeterogeneousMerging/models/'
    act_dir  = os.path.join(save_dir, 'activations')
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ==========================================
    # Data
    # ==========================================
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    fashion_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    N_map = 5000

    mnist_test     = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=mnist_transform)
    mnist_loader   = DataLoader(mnist_test, batch_size=256, shuffle=False)
    fashion_test   = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=fashion_transform)
    fashion_loader = DataLoader(fashion_test, batch_size=256, shuffle=False)
    fashion_subset_loader = DataLoader(
        Subset(fashion_test, list(range(N_map))),
        batch_size=256, shuffle=False)

    # ==========================================
    # Load models
    # ==========================================
    standard_model     = build_dense_model(256,  128,  use_relu)
    wide_model         = build_dense_model(1024, 512,  use_relu)
    standard_finetuned = build_dense_model(256,  128,  use_relu)
    wide_finetuned     = build_dense_model(1024, 512,  use_relu)
    
    standard_model.load_state_dict(    torch.load(os.path.join(load_dir, MNIST_STD_MODEL),           map_location=device))
    wide_model.load_state_dict(        torch.load(os.path.join(load_dir, MNIST_WIDE_MODEL),              map_location=device))
    standard_finetuned.load_state_dict(torch.load(os.path.join(load_dir, FMNIST_STD_MODEL), map_location=device))
    wide_finetuned.load_state_dict(    torch.load(os.path.join(load_dir, FMNIST_WIDE_MODEL),    map_location=device))
    print("Models loaded.")

    # ==========================================
    # Compute task vectors
    # ==========================================
    task_vectors = {}
    wide_params = dict(wide_model.named_parameters())
    ft_params   = dict(wide_finetuned.named_parameters())
    for name in wide_params:
        task_vectors[name] = ft_params[name].data - wide_params[name].data

    weight_layers = [n for n in task_vectors if 'weight' in n]
    bias_layers   = [n for n in task_vectors if 'bias'   in n]

    std_params = dict(standard_model.named_parameters())

    # ==========================================
    # True task vectors (standard finetuned - standard pretrained)
    # ==========================================
    true_task_vectors = {}
    std_ft_params = dict(standard_finetuned.named_parameters())
    for name in std_params:
        true_task_vectors[name] = std_ft_params[name].data - std_params[name].data

    # ==========================================
    # Load ortho matrices and transport
    # ==========================================
    comparison = {}

    for ortho_file, label in TRANSPORT_ORTHO:
        filename = f'{ortho_file}{suffix}.npy'
        path = os.path.join(save_dir, filename)

        if not os.path.exists(path):
            print(f"\nSkipping {label}: {path} not found")
            continue

        print(f"\n========== [{label}] ==========")
        T_maps = np.load(path, allow_pickle=True).item()
        print(f"Loaded {len(T_maps)} matrices from {filename}")

        # Build layer mappings from T_maps keys
        # T_maps has keys like 'layer_0_Flatten_pre', 'layer_0_Flatten_post', ...
        # Extract base layer names (without _pre/_post)
        act_layers = []
        for k in T_maps.keys():
            if k.endswith('_pre'):
                act_layers.append(k[:-4])  # strip '_pre'
        print(f"  Activation layers: {act_layers}")

        assert len(weight_layers) == len(act_layers) - 1, (
            f"Expected {len(act_layers)-1} weight layers, got {len(weight_layers)}"
        )

        weight_to_layers = {
            weight_layers[i]: (act_layers[i], act_layers[i + 1])
            for i in range(len(weight_layers))
        }
        bias_to_layers = {
            bias_layers[i]: act_layers[i + 1]
            for i in range(len(bias_layers))
        }

        results = transport_with_ortho(task_vectors, weight_to_layers, bias_to_layers, T_maps, device)

        best_alpha, best_acc, alpha_results = alpha_search(
            standard_model, results, std_params, fashion_loader, device, label)

        # ---- Compare alpha * transported vs true task vectors ----
        print(f"\n  --- Task vector comparison [{label}] (alpha={best_alpha:.2f}) ---")
        total_transported_norm = 0.0
        total_true_norm = 0.0
        total_diff_norm = 0.0
        # Concatenate all matched params for overall cosine similarity
        all_transported_flat = []
        all_true_flat = []
        for tv_name in results:
            tau_transported = (best_alpha * results[tv_name]).double().cpu()
            tau_true = true_task_vectors[tv_name].double().cpu()
            if tau_transported.shape != tau_true.shape:
                continue
            diff = torch.norm(tau_transported - tau_true).item()
            true_norm = torch.norm(tau_true).item()
            transported_norm = torch.norm(tau_transported).item()
            rel = diff / (true_norm + 1e-12)
            cos_sim = (tau_transported.flatten() @ tau_true.flatten()).item() / (transported_norm * true_norm + 1e-12)
            print(f"    {tv_name}: rel_diff={rel:.4f}, cos_sim={cos_sim:.4f}, "
                  f"||alpha*transported||={transported_norm:.4f}, ||true||={true_norm:.4f}")
            total_diff_norm += diff ** 2
            total_true_norm += true_norm ** 2
            total_transported_norm += transported_norm ** 2
            all_transported_flat.append(tau_transported.flatten())
            all_true_flat.append(tau_true.flatten())
        overall_rel = np.sqrt(total_diff_norm) / (np.sqrt(total_true_norm) + 1e-12)
        all_transported_flat = torch.cat(all_transported_flat)
        all_true_flat = torch.cat(all_true_flat)
        overall_cos = (all_transported_flat @ all_true_flat).item() / (torch.norm(all_transported_flat).item() * torch.norm(all_true_flat).item() + 1e-12)
        print(f"    --- Overall: rel_diff={overall_rel:.4f}, cos_sim={overall_cos:.4f}")

        stitched = apply_task_vectors(standard_model, results, std_params, best_alpha, device)
        comparison[label] = (stitched, best_alpha, best_acc, alpha_results, filename)

    # ==========================================
    # Final evaluation
    # ==========================================
    print("\n========== Final Evaluation ==========")
    print(f"\n[FashionMNIST — first {N_map} samples (used to compute maps)]")
    acc_baseline  = evaluate(standard_model,     fashion_subset_loader, device, "Standard model          ")
    evaluate(wide_model,         fashion_subset_loader, device, "Wide model              ")
    acc_finetuned = evaluate(standard_finetuned, fashion_subset_loader, device, "Standard finetuned      ")
    evaluate(wide_finetuned,     fashion_subset_loader, device, "Wide finetuned          ")
    transferable  = acc_finetuned - acc_baseline
    print(f"\n  Transferable accuracy: {acc_finetuned:.2f}% - {acc_baseline:.2f}% = {transferable:.2f}%")
    print(f"  {'method':<30} {'acc':>8} {'normalized':>12}")
    for method_name, (model, best_alpha, _, _, _) in comparison.items():
        acc = evaluate(model, fashion_subset_loader, device, f"Stitched {method_name:<25} (alpha={best_alpha:.1f})")
        normalized = (acc - acc_baseline) / (transferable + 1e-12) * 100
        print(f"  {method_name:<30} {acc:>7.2f}% {normalized:>11.1f}%")

    print(f"\n[FashionMNIST — full test set]")
    acc_baseline_full  = evaluate(standard_model,     fashion_loader, device, "Standard model          ")
    evaluate(wide_model,         fashion_loader, device, "Wide model              ")
    acc_finetuned_full = evaluate(standard_finetuned, fashion_loader, device, "Standard finetuned      ")
    evaluate(wide_finetuned,     fashion_loader, device, "Wide finetuned          ")
    transferable_full  = acc_finetuned_full - acc_baseline_full
    print(f"\n  Transferable accuracy: {acc_finetuned_full:.2f}% - {acc_baseline_full:.2f}% = {transferable_full:.2f}%")
    print(f"  {'method':<30} {'acc':>8} {'normalized':>12}")
    for method_name, (model, best_alpha, _, _, _) in comparison.items():
        acc = evaluate(model, fashion_loader, device, f"Stitched {method_name:<25} (alpha={best_alpha:.1f})")
        normalized = (acc - acc_baseline_full) / (transferable_full + 1e-12) * 100
        print(f"  {method_name:<30} {acc:>7.2f}% {normalized:>11.1f}%")
    print(f"\n[MNIST — full test set]")
    acc_baseline_full  = evaluate(standard_model,     mnist_loader, device, "Standard model          ")
    evaluate(wide_model,         mnist_loader, device, "Wide model              ")
    acc_finetuned_full = evaluate(standard_finetuned, mnist_loader, device, "Standard finetuned      ")
    evaluate(wide_finetuned,     mnist_loader, device, "Wide finetuned          ")
    transferable_full  = acc_finetuned_full - acc_baseline_full
    print(f"\n  Transferable accuracy: {acc_finetuned_full:.2f}% - {acc_baseline_full:.2f}% = {transferable_full:.2f}%")
    print(f"  {'method':<30} {'acc':>8} {'normalized':>12}")
    for method_name, (model, best_alpha, _, _, _) in comparison.items():
        acc = evaluate(model, mnist_loader, device, f"Stitched {method_name:<25} (alpha={best_alpha:.1f})")
        normalized = (acc - acc_baseline_full) / (transferable_full + 1e-12) * 100
        print(f"  {method_name:<30} {acc:>7.2f}% {normalized:>11.1f}%")

    # ==========================================
    # Alpha sweep summary
    # ==========================================
    for method_name, (_, best_alpha, best_acc, alpha_results, _) in comparison.items():
        print(f"\n--- [{method_name}] Alpha sweep ---")
        print(f"  {'alpha':>6}  {'acc':>14}")
        for alpha, acc in alpha_results.items():
            marker = " <-- best" if alpha == best_alpha else ""
            print(f"  {alpha:>6.2f}  {acc:>13.2f}%{marker}")

    # ==========================================
    # Summary table
    # ==========================================
    print("\n========== Summary Table ==========")
    print(f"  {'file_name':<45} {'performance':>12} {'alpha':>7}")
    print(f"  {'-'*45} {'-'*12} {'-'*7}")
    for method_name, (_, best_alpha, best_acc, _, filename) in comparison.items():
        print(f"  {filename:<45} {best_acc:>11.2f}% {best_alpha:>7.2f}")