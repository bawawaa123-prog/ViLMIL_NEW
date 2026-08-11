"""
Inspect BiomedCLIP extracted HDF5 feature files.

Usage examples (Windows cmd):
    python tools/inspect_biomedclip_feature.py --file D:/FenLei/ViLa-MIL-main/features_biomedclip_5x/2463643-B.h5

Optional flags:
    --plot    : generate PCA scatter and norms histogram (requires matplotlib and scikit-learn)
    --save-npy: save a sample of features as .npy for quick loading

Output:
    Prints dataset keys, shapes, dtype and statistics and optionally saves plots in the same folder.
"""

import os
import argparse
import math
import numpy as np
import h5py
import json


def safe_mean_std(arr):
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def load_features(h5_path):
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise RuntimeError(f"Empty HDF5 file: {h5_path}")

        # Prefer 'features' dataset if present
        if 'features' in f:
            feats = np.array(f['features'])
        else:
            # fallback to the first dataset that looks like features
            feats = np.array(f[keys[0]])

        coords = None
        if 'coords' in f:
            try:
                coords = np.array(f['coords'])
            except Exception:
                coords = None

    return feats, coords, keys


def inspect_file(h5_path, do_plot=False, save_npy=False, sample_limit=1000):
    print(f"Inspecting: {h5_path}")
    if not os.path.exists(h5_path):
        print("File not found.")
        return 1

    feats, coords, keys = load_features(h5_path)

    print('\nHDF5 keys:', keys)
    print('\nFeature array shape:', feats.shape)
    print('dtype:', feats.dtype)

    n_patches, feat_dim = feats.shape

    # Basic stats
    per_feat_mean, per_feat_std = safe_mean_std(np.mean(feats, axis=0))  # per-dimension mean & std aggregated
    per_patch_mean, per_patch_std = safe_mean_std(np.mean(feats, axis=1))
    print('\nPer-dimension mean (aggregated): {:.6f} | std {:.6f}'.format(per_feat_mean, per_feat_std))
    print('Per-patch mean (aggregated): {:.6f} | std {:.6f}'.format(per_patch_mean, per_patch_std))

    # Norms
    norms = np.linalg.norm(feats, axis=1)
    norm_mean, norm_std = safe_mean_std(norms)
    print('\nFeature vector norms (per patch): mean {:.6f}, std {:.6f}, min {:.6f}, max {:.6f}'.format(norm_mean, norm_std, float(norms.min()) if norms.size else float('nan'), float(norms.max()) if norms.size else float('nan')))

    # Top-10 by norm
    if norms.size > 0:
        topk = min(10, norms.size)
        top_idx = np.argsort(-norms)[:topk]
        print('\nTop-{} patches by norm:'.format(topk))
        for i, idx in enumerate(top_idx):
            coord = coords[idx].tolist() if coords is not None and len(coords) > idx else None
            print(f"  {i+1}. idx={idx}, norm={norms[idx]:.6f}, coord={coord}")

    # Print a few sample vectors
    print('\nFirst 3 feature vectors (truncated to 8 values):')
    for i in range(min(3, n_patches)):
        v = feats[i]
        snippet = ', '.join([f"{x:.6f}" for x in v[:8]])
        print(f"  patch {i}: [{snippet}, ...] (len={len(v)})")

    # Summary metrics for each dimension (top 5 dims by variance)
    if feat_dim <= 50:
        print('\nPer-dimension mean/var (first 50 dims):')
        means = feats.mean(axis=0)
        vars_ = feats.var(axis=0)
        for i in range(min(50, feat_dim)):
            print(f"  dim {i}: mean={means[i]:.6f}, var={vars_[i]:.6f}")
    else:
        # print top 5 dims by variance
        vars_ = feats.var(axis=0)
        top_dims = np.argsort(-vars_)[:5]
        print('\nTop-5 dimensions by variance:')
        for d in top_dims:
            print(f"  dim {d}: mean={feats[:,d].mean():.6f}, var={vars_[d]:.6f}")

    # Optional plotting: PCA 2 components and norms histogram
    if do_plot:
        try:
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA

            # limit sample size for PCA to 2k
            sample_size = min(n_patches, 2000)
            sample_idx = np.linspace(0, n_patches - 1, sample_size).astype(int)
            X = feats[sample_idx]

            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)

            base_dir = os.path.dirname(h5_path)
            scatter_fn = os.path.join(base_dir, os.path.basename(h5_path).replace('.h5', '_pca_scatter.png'))
            hist_fn = os.path.join(base_dir, os.path.basename(h5_path).replace('.h5', '_norms_hist.png'))

            plt.figure(figsize=(8, 6))
            sc = plt.scatter(X2[:, 0], X2[:, 1], c=norms[sample_idx], cmap='viridis', s=10)
            plt.colorbar(sc, label='L2 norm')
            plt.title('PCA (2D) of {} sample features'.format(sample_size))
            plt.xlabel('PC1'); plt.ylabel('PC2')
            plt.tight_layout()
            plt.savefig(scatter_fn)
            plt.close()
            print('\nSaved PCA scatter to', scatter_fn)

            # Norm histogram
            plt.figure(figsize=(7, 4))
            plt.hist(norms, bins=50, color='C0', alpha=0.8)
            plt.title('Distribution of feature L2 norms')
            plt.xlabel('L2 norm'); plt.ylabel('count')
            plt.tight_layout()
            plt.savefig(hist_fn)
            plt.close()
            print('Saved norms histogram to', hist_fn)

        except Exception as e:
            print('\nPlotting failed (matplotlib/sklearn required):', e)
            print('Tip: pip install matplotlib scikit-learn')

    # Optional: save a sampled feature array to a .npy
    if save_npy:
        try:
            out_npy = os.path.splitext(h5_path)[0] + '_features_sample.npy'
            sample_limit = min(sample_limit, feats.shape[0])
            np.save(out_npy, feats[:sample_limit])
            print('\nSaved sample features to', out_npy)
        except Exception as e:
            print('\nFailed to save npy:', e)

    print('\nDone.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect a BiomedCLIP HDF5 feature file')
    parser.add_argument('--file', required=True, help='Path to .h5 (features + coords)')
    parser.add_argument('--plot', action='store_true', help='Generate PCA and norm histogram plots (requires matplotlib & scikit-learn)')
    parser.add_argument('--save-npy', action='store_true', help='Save a sample of features as .npy')
    parser.add_argument('--sample-limit', type=int, default=1000, help='Number of features to sample when saving .npy (default 1000)')
    args = parser.parse_args()

    exit(inspect_file(args.file, do_plot=args.plot, save_npy=args.save_npy, sample_limit=args.sample_limit))
