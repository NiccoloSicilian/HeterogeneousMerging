import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from scipy.special import erf 


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
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            model(images)
            all_labels.append(labels.numpy())
            collected += images.size(0)
            if collected >= N:
                break

    all_labels = np.concatenate(all_labels)[:N]

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

    result['_labels'] = all_labels
    return result


def extract_activations_random_lw(model, N, device):
    """
    Extract activations layer-wise using independent positive random samples per layer.

    For each layer, generates random positive input X (N, d_in) with d_in = input dim of that layer,
    computes pre = W @ X.T + b (the linear output), and if an activation follows, post = act(pre).

    This treats each layer independently — no forward pass through the full model.

    Returns same format: {layer_name: (pre_act (N, d_out), post_act (N, d_out) or None, act_name or None)}
    """
    layers = list(model.children())
    model.eval().to(device)
    result = {}

    for i, layer in enumerate(layers):
        if isinstance(layer, ACTIVATION_TYPES):
            continue

        next_layer = layers[i + 1] if i + 1 < len(layers) else None
        has_act = isinstance(next_layer, ACTIVATION_TYPES)
        act_name = type(next_layer).__name__ if has_act else None
        layer_name = f"layer_{i}_{type(layer).__name__}"

        if isinstance(layer, nn.Flatten):
            # Flatten has no weights — use random positive input images flattened
            d_in = 28 * 28
            X = torch.abs(torch.randn(N, d_in, device=device))
            pre_arr = X.cpu().numpy()
            post_arr = None
            if has_act:
                with torch.no_grad():
                    post_arr = next_layer(X).cpu().numpy()
        elif isinstance(layer, nn.Linear):
            d_in = layer.in_features
            X = torch.abs(torch.randn(N, d_in, device=device))
            with torch.no_grad():
                pre = layer(X)  # W @ x + b
            pre_arr = pre.cpu().numpy()
            post_arr = None
            if has_act:
                with torch.no_grad():
                    post_arr = next_layer(pre).cpu().numpy()
        else:
            print(f"  Skipping unsupported layer type: {type(layer).__name__}")
            continue

        result[layer_name] = (pre_arr, post_arr, act_name)
        post_shape = post_arr.shape if post_arr is not None else None
        print(f"  {layer_name}: pre={pre_arr.shape}, post={post_shape}, activation={act_name}")

    return result

def build_interpolated_activations(activations, n_anchors, n_interp, seed, noise_std=0.0):
    """
    From extracted activations, build an augmented activation dict with
    real anchor samples followed by interpolated samples.

    For each layer:
    - Take first n_anchors real samples as anchors
    - Generate n_interp interpolated pre-activation samples (linear combos of anchors)
    - Compute post-activation by applying the actual activation function
      to the interpolated pre-activations (so they lie on the true manifold)
    - Output ordering: [n_anchors real, n_interp interpolated]

    Uses a seeded RNG so that two models called with the same seed produce
    corresponding interpolated points (same pair indices, same alphas).
    """
    print("Interpolating")
    result = {}
    for layer_name, (pre, post, act_name) in activations.items():
        rng = np.random.default_rng(seed)

        pre_base = pre[:n_anchors].astype(np.float64)

        idx_i = rng.integers(0, n_anchors, size=n_interp)
        idx_j = rng.integers(0, n_anchors, size=n_interp)
        alpha = rng.beta(1, 4, size=(n_interp, 1))
        print(f"  {layer_name}: alphas={alpha.ravel().tolist()}, avg={alpha.mean():.4f}")
        pre_interp = alpha * pre_base[idx_i] + (1 - alpha) * pre_base[idx_j]

        if noise_std > 0.0:
            per_neuron_std = pre_base.std(axis=0, keepdims=True)
            noise = rng.normal(0, 1, size=pre_interp.shape) * per_neuron_std * noise_std
            pre_interp = pre_interp + noise

        pre_out = np.concatenate([pre_base, pre_interp], axis=0)

        post_out = None
        if act_name is not None and act_name in ACTIVATION_FUNCTIONS:
            post_out = ACTIVATION_FUNCTIONS[act_name](pre_out)
        elif act_name is not None:
            print(f"  WARNING: unknown activation '{act_name}' for {layer_name}, skipping post")

        result[layer_name] = (pre_out, post_out, act_name)
        post_shape = post_out.shape if post_out is not None else None
        print(f"  {layer_name}: pre={pre_out.shape}, post={post_shape} (anchors={n_anchors}, interp={n_interp})")

    return result




def method_random_lw(model, N, device, params):
    """Extract activations layer-wise using independent random positive inputs."""
    return extract_activations_random_lw(model, N, device)


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
    ds = params['dataset'](root='/kaggle/working/data', train=True, download=True, transform=tf)
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


"""
EXTRACT_ACTIVATION config format:

EXTRACT_ACTIVATION = {
    'models': {
        'model_name': {
            'hidden_size_1': int,
            'hidden_size_2': int,
            'weights': str (path to .pth file),
        },
        ...
    },
    'N': int (max samples to extract),
    'save_dir': str,
    'to_extract': [
        {
            'using': str (model name from 'models'),
            'method': callable(model, N, device, params) -> activations dict,
            'params': dict (passed to method),
            'filename': str (output filename without suffix),
        },
        ...
    ],
}

All methods must have signature: method(model, N, device, params) -> result
  result is either:
    - a dict (layer activations, optionally '_labels')
    - a list of (dict, filename_suffix) tuples (e.g. method_gvendi returns multiple subsets)
"""

if EXTRACT_ACTIVATION is not None:
    cfg = EXTRACT_ACTIVATION
    use_relu = USE_RELU
    suffix = '_relu' if use_relu else '_norelu'
    out_dir = os.path.join(cfg['save_dir'], 'activations')
    os.makedirs(out_dir, exist_ok=True)
    N = cfg['N']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    def get_model(name):
        m = cfg['models'][name]
        model = build_dense_model(m['hidden_size_1'], m['hidden_size_2'], use_relu)
        model.load_state_dict(torch.load(m['weights'], map_location=device))
        return model

    n_saved = 0
    for entry in cfg['to_extract']:
        model_name = entry['using']
        method = entry['method']
        params = entry.get('params', {})
        base_filename = entry['filename']
        model = get_model(model_name)

        print(f"\nExtracting: {base_filename} (model={model_name})...")
        result = method(model, N, device, params)

        # method returns either a single dict or a list of (dict, suffix) tuples
        if isinstance(result, list):
            for activations, name_suffix in result:
                filename = f"{base_filename}{name_suffix}{suffix}.npy"
                np.save(os.path.join(out_dir, filename), activations)
                print(f"  Saved to {filename}")
                n_saved += 1
        else:
            filename = f"{base_filename}{suffix}.npy"
            np.save(os.path.join(out_dir, filename), result)
            print(f"  Saved to {filename}")
            n_saved += 1

    print(f"\nDone. Saved {n_saved} activation files to '{out_dir}/'")
