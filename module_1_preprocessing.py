# module_1_preprocessing.py
import open3d as o3d
import numpy as np
import time
from scipy.spatial import KDTree


def load_and_preprocess_mesh(file_path, output_path="airplane_preprocessed.stl"):
    print(f"\n[Module 1] 启动预处理与特征提取 ({file_path})...")
    start_time = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 网格清理与法向计算
    # ──────────────────────────────────────────────────────────────────────────
    print(" -> 正在清理底层网格并提取顶点法向...")
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(output_path, mesh)

    # 构建射线场景（供 Module 2 & 5 使用）
    mesh_t    = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 向量化 PCA 曲率计算
    #
    # 原始实现：对 50 万个点逐一调用 Open3D KDTree，Python 层循环 500 000 次，
    #           每次 PCA 均为单点计算，整体 O(N·K) 调度开销极大。
    #
    # 优化策略：
    #   ① 使用 scipy.spatial.KDTree.query() 批量（parallel）查询所有点的
    #      K 近邻索引，一次 C 层调用代替 50 万次 Python 调用；
    #   ② 以 numpy einsum 批量计算协方差矩阵，再用 np.linalg.eigvalsh
    #      一次性对全批次求特征值——纯向量化，无 Python 循环；
    #   ③ 分块（BATCH=50 000）控制内存峰值，每块约 36 MB，避免一次性展开
    #      500 000×30×3 的 ~360 MB 邻域张量。
    # ──────────────────────────────────────────────────────────────────────────
    print(" -> 正在执行向量化 PCA 曲率计算与自适应采样...")

    base_pcd = mesh.sample_points_uniformly(number_of_points=500_000)
    points   = np.asarray(base_pcd.points)   # (N, 3)
    N        = len(points)
    K        = 30

    t_kd = time.time()
    tree = KDTree(points)
    # workers=-1 启用多线程，充分利用 CPU 并行；返回 (N, K) 索引数组
    _, knn_idx = tree.query(points, k=K, workers=-1)
    print(f"    KDTree 批量近邻查询完成，耗时 {time.time() - t_kd:.1f}s")

    # 分批向量化 PCA
    BATCH      = 50_000
    curvatures = np.empty(N, dtype=np.float64)

    for start in range(0, N, BATCH):
        end  = min(start + BATCH, N)
        nbrs = points[knn_idx[start:end]]              # (B, K, 3)
        ctr  = nbrs.mean(axis=1, keepdims=True)        # (B, 1, 3)
        ctrd = nbrs - ctr                              # (B, K, 3)
        # 协方差矩阵 (B, 3, 3)，等价于 ctrd.T @ ctrd / K，逐样本
        covs = np.einsum('nki,nkj->nij', ctrd, ctrd) / K
        # eigvalsh 保证返回升序实特征值，比 eig 快且稳定
        eigs = np.linalg.eigvalsh(covs)                # (B, 3) 升序
        tr   = eigs.sum(axis=1)                        # (B,)
        curvatures[start:end] = np.where(tr > 1e-8, eigs[:, 0] / tr, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 自适应降采样
    # ──────────────────────────────────────────────────────────────────────────
    curvature_threshold = 0.015
    curved_idx = np.where(curvatures >  curvature_threshold)[0]
    flat_idx   = np.where(curvatures <= curvature_threshold)[0]

    rng        = np.random.default_rng(seed=42)   # 固定种子，结果可复现
    sel_curved = rng.choice(curved_idx, size=max(1, int(len(curved_idx) * 0.5)),
                            replace=False)
    sel_flat   = rng.choice(flat_idx,   size=max(1, int(len(flat_idx)   * 0.1)),
                            replace=False)
    sel_all    = np.concatenate([sel_curved, sel_flat])

    final_pcd = base_pcd.select_by_index(sel_all.tolist())
    pts       = np.asarray(final_pcd.points)
    norms     = np.asarray(final_pcd.normals)

    print(
        f"[Module 1] 预处理完成！特征点数: {len(pts)}"
        f"  耗时: {time.time() - start_time:.1f}s"
    )
    return mesh, pts, norms, ray_scene


if __name__ == "__main__":
    load_and_preprocess_mesh("airplane_aligned.stl")
