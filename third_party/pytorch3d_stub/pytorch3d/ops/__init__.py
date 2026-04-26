"""
pytorch3d.ops stub — pure-PyTorch CPU replacement for knn_points.

Uses chunked cdist to cap peak memory per call at ~CHUNK_MB megabytes
regardless of batch size, avoiding the original (B*M*N) spike that caused
RAM swap-thrashing on CPU-only machines.
"""

import torch
from collections import namedtuple

_KNNOutput = namedtuple("KNNOutput", ["dists", "idx", "knn"])

# Maximum MB to allocate in one cdist call (tune down if OOM, up for speed).
_CHUNK_MB = 128


def knn_points(p1, p2,
               lengths1=None, lengths2=None,
               K: int = 1,
               return_nn: bool = False,
               return_sorted: bool = True):
    """
    For each query point in p2 find the K nearest neighbours in p1.

    Chunked along the M (query) dimension so peak RAM ≈ B * CHUNK * N * 4 bytes.

    Args:
        p1: (B, N, D)  points to search in
        p2: (B, M, D)  query points
    Returns:
        KNNOutput(dists=(B,M,K), idx=(B,M,K), knn=None)
    """
    p1f = p1.float()
    p2f = p2.float()
    B, M, _ = p2f.shape
    N = p1f.shape[1]

    # How many query points fit in one chunk given the memory budget?
    bytes_per_row = N * 4  # one float32 distance per p1 point
    chunk = max(1, int(_CHUNK_MB * 1024 * 1024 // (B * bytes_per_row)))
    chunk = min(chunk, M)

    dists_out = torch.empty(B, M, K, dtype=torch.float32)
    idx_out   = torch.empty(B, M, K, dtype=torch.long)

    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        # (B, end-start, N)
        d = torch.cdist(p2f[:, start:end, :], p1f, p=2.0).pow_(2)
        topk_d, topk_i = d.topk(K, dim=-1, largest=False, sorted=return_sorted)
        dists_out[:, start:end, :] = topk_d
        idx_out  [:, start:end, :] = topk_i

    return _KNNOutput(dists=dists_out, idx=idx_out, knn=None)
