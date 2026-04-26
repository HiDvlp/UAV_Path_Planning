# module_1_preprocessing.py
#
# 新版预处理模块：从 .pcd 纯点云出发，使用 SHS-Net 进行有方向法向估计，
# 再通过 Screened Poisson 重建生成三角网格，供下游模块使用。
#
# 流水线：
#   1. PCD 文件读取
#   2. SHS-Net 推理 → 有方向法向 (N, 3)
#   3. Screened Poisson 重建 → 三角网格 mesh
#   4. 构建 ray_scene（与原版一致，供 Module 2 / 5 使用）
#   5. PCA 曲率计算（用于自适应降采样密度控制）
#   6. 自适应降采样 → pts, norms
#   7. 输出 mesh, pts, norms, ray_scene
#
# 依赖项（除项目原有依赖外）：
#   - SHS-Net 代码仓库：git clone https://github.com/LeoQLi/SHS-Net.git
#     将其路径添加到 PYTHONPATH 或放在项目 third_party/ 目录下
#   - PyTorch >= 1.8
#   - Pytorch3D >= 0.6（可选）：若未安装，自动使用 third_party/pytorch3d_stub/
#     中的纯 PyTorch CPU 替代实现（仅实现 knn_points，无需编译）

import os
import sys
import time
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

import torch

# ──────────────────────────────────────────────────────────────────────────────
# SHS-Net 路径配置
#
# 配置方式（按优先级）：
#   1. 环境变量 SHS_NET_ROOT 指向 SHS-Net 仓库根目录
#   2. 项目 third_party/SHS-Net/ 目录（默认 fallback）
#
# 安装：
#   cd your_project/third_party
#   git clone https://github.com/LeoQLi/SHS-Net.git
#   # 预训练权重已包含在仓库中: log/001/ckpts/ckpt_800.pt
# ──────────────────────────────────────────────────────────────────────────────
SHS_NET_ROOT = os.environ.get("SHS_NET_ROOT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "SHS-Net"
))

# SHS-Net 默认预训练权重路径
SHS_NET_CKPT = os.environ.get("SHS_NET_CKPT", os.path.join(
    SHS_NET_ROOT, "log", "001", "ckpts", "ckpt_800.pt"
))


def _ensure_shs_net_available():
    """验证 SHS-Net 代码和权重是否就位，给出可操作的错误提示。
    若 pytorch3d 未安装，自动注入纯 PyTorch 的 CPU stub 以替代。
    """
    if not os.path.isdir(SHS_NET_ROOT):
        raise FileNotFoundError(
            f"[Module 1] 找不到 SHS-Net 代码仓库: {SHS_NET_ROOT}\n"
            f"  请执行以下命令安装:\n"
            f"    cd {os.path.dirname(os.path.abspath(__file__))}/../third_party\n"
            f"    git clone https://github.com/LeoQLi/SHS-Net.git\n"
            f"  或者设置环境变量:\n"
            f"    export SHS_NET_ROOT=/path/to/SHS-Net"
        )
    net_module = os.path.join(SHS_NET_ROOT, "net", "network.py")
    if not os.path.isfile(net_module):
        raise FileNotFoundError(
            f"[Module 1] SHS-Net 目录结构不完整, 缺少 net/network.py: {SHS_NET_ROOT}\n"
            f"  请确认 SHS_NET_ROOT 指向包含 net/ 子目录的仓库根目录。"
        )

    # ── pytorch3d stub：若未安装真正的 pytorch3d，注入纯 PyTorch CPU 替代 ──
    try:
        import pytorch3d  # noqa: F401
    except ImportError:
        _STUB_ROOT = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "pytorch3d_stub"
        )
        _STUB_ROOT = os.path.normpath(_STUB_ROOT)
        if os.path.isdir(_STUB_ROOT) and _STUB_ROOT not in sys.path:
            sys.path.insert(0, _STUB_ROOT)
            print(" -> pytorch3d 未安装，已注入纯 PyTorch CPU stub (third_party/pytorch3d_stub)")
        elif not os.path.isdir(_STUB_ROOT):
            raise ImportError(
                "[Module 1] pytorch3d 未安装且找不到 CPU stub。\n"
                "  请运行: pip install pytorch3d\n"
                "  或确认 third_party/pytorch3d_stub/ 目录存在。"
            )

    if SHS_NET_ROOT not in sys.path:
        sys.path.insert(0, SHS_NET_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# 第 1 步 ─ PCD 文件读取
# ══════════════════════════════════════════════════════════════════════════════

def _load_pcd(file_path: str) -> np.ndarray:
    """
    读取 .pcd 文件，返回纯 xyz 坐标数组 (N, 3)。
    支持 Open3D 可解析的所有 PCD 格式（ASCII / Binary / Binary_compressed）。
    """
    print(f" -> 正在读取点云文件: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    if len(points) == 0:
        raise RuntimeError(f"点云文件为空或读取失败: {file_path}")
    print(f"    原始点云点数: {len(points)}")
    return points


# ══════════════════════════════════════════════════════════════════════════════
# 第 2 步 ─ SHS-Net 有方向法向推理
# ══════════════════════════════════════════════════════════════════════════════

class SHSNetEstimator:
    """
    封装 SHS-Net 推理逻辑，将原始 (N, 3) 点云转化为 (N, 3) 有方向法向。

    SHS-Net 的推理流程（基于官方 test.py）：
      对每个查询点 q_i：
        1) 提取 K 近邻 patch → 中心化 + 归一化 → PCA 旋转到局部坐标系
        2) 从整个点云中下采样全局样本点 → 同样做 PCA 变换
        3) patch 特征和 shape 特征送入 Network 前向推理 → 输出局部坐标系下的法向
        4) 用 PCA 逆变换旋转回世界坐标系
    """

    # SHS-Net 默认超参数（与官方 run.py 中 PCPNet 配置一致）
    PATCH_SIZE  = 700     # 局部 patch 近邻数
    SAMPLE_SIZE = 700     # 全局 shape 采样点数
    ENCODE_KNN  = 16      # 编码器内部 KNN 邻域
    BATCH_SIZE  = 500     # 推理批量大小（可根据 GPU 显存调整）

    def __init__(self, ckpt_path: str = SHS_NET_CKPT, device: str = "auto"):
        """
        :param ckpt_path: SHS-Net 预训练权重路径
        :param device:    "auto" 自动选择 GPU/CPU, 或指定如 "cuda:0" / "cpu"
        """
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 验证 SHS-Net 代码和权重
        _ensure_shs_net_available()
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"[Module 1] 找不到 SHS-Net 预训练权重: {ckpt_path}\n"
                f"  权重文件随仓库自带于 log/001/ckpts/ckpt_800.pt\n"
                f"  或设置环境变量: export SHS_NET_CKPT=/path/to/ckpt_800.pt"
            )

        print(f" -> 正在加载 SHS-Net 模型: {ckpt_path}")
        print(f"    推理设备: {self.device}")

        from net.network import Network  # SHS-Net 内部模块

        self.model = Network(
            num_pat=self.PATCH_SIZE,
            num_sam=self.SAMPLE_SIZE,
            encode_knn=self.ENCODE_KNN,
        ).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)
        self.model.eval()
        print(f"    模型加载完成，参数量: "
              f"{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    @staticmethod
    def _batch_pca_transform(patches: np.ndarray):
        """
        对一批 patch 批量执行 PCA 变换，消除 Python 逐点循环。

        严格复现 SHS-Net PCATrans 逻辑，但用 numpy 批量 SVD 替代逐个 torch SVD。

        :param patches: (B, K, 3) 已中心化+归一化的邻域点批次
        :return: (patches_pca, trans_batch, cp_new_batch)
                 patches_pca:  (B, K, 3)
                 trans_batch:  (B, 3, 3)
                 cp_new_batch: (B, 3)
        """
        B, K, _ = patches.shape

        # 1) 每个 patch 的均值
        pts_mean = patches.mean(axis=1)                 # (B, 3)
        centered = patches - pts_mean[:, None, :]       # (B, K, 3)

        # 2) 批量 SVD：对 centered.T 即 (B, 3, K) 做 SVD
        #    np.linalg.svd 对 (..., M, N) 数组做批量分解
        #    full_matrices=False: U (B, 3, 3), S (B, 3), Vh (B, 3, K)
        centered_T = centered.transpose(0, 2, 1)       # (B, 3, K)
        U, _, _ = np.linalg.svd(centered_T, full_matrices=False)  # U: (B, 3, 3)
        trans_batch = U.astype(np.float32)              # (B, 3, 3)

        # 3) 旋转：rotated = centered @ trans
        rotated = np.einsum('bki,bij->bkj', centered, trans_batch)  # (B, K, 3)

        # 4) 查询点在 PCA 空间中的坐标
        #    cp_new = (-pts_mean) @ trans
        cp_new_batch = np.einsum('bi,bij->bj',
                                 -pts_mean, trans_batch)  # (B, 3)

        # 5) 重新以查询点为中心
        patches_pca = rotated - cp_new_batch[:, None, :]  # (B, K, 3)

        return patches_pca, trans_batch, cp_new_batch

    def estimate_normals(self, points: np.ndarray):
        """
        对整个点云执行 SHS-Net 推理，返回有方向法向。

        同时返回近邻索引供下游曲率计算复用，避免重复构建。

        :param points: (N, 3) float32 点云坐标
        :return: (normals, knn_idx)
                 normals: (N, 3) float32 单位法向量（有方向）
                 knn_idx: (N, K) int 近邻索引（K=PATCH_SIZE，可截取前列用于曲率）
        """
        N = len(points)
        K = self.PATCH_SIZE
        print(f" -> SHS-Net 推理中 (N={N}, patch={K}, "
              f"sample={self.SAMPLE_SIZE}, batch={self.BATCH_SIZE}) ...")

        t0 = time.time()

        # ── 构建 KDTree ──
        print(f"    正在构建 KDTree ...")
        t_kd = time.time()
        kdtree = KDTree(points)

        # ── 近邻查询策略：根据内存预算选择批量/分段 ──
        # (N, K) 的 int64 + float64 ≈ N * K * 16 bytes
        mem_estimate_gb = N * K * 16 / (1024**3)
        MEM_LIMIT_GB = 2.0  # 允许近邻索引占用的最大内存

        if mem_estimate_gb <= MEM_LIMIT_GB:
            # 整体批量查询（快，内存允许）
            print(f"    批量近邻查询 (预估 {mem_estimate_gb:.1f} GB) ...")
            all_dists, all_knn_idx = kdtree.query(points, k=K, workers=-1)
            all_dists = all_dists.astype(np.float32)  # scipy 默认返回 float64
        else:
            # 分段查询（控制内存峰值）
            CHUNK = max(1, int(MEM_LIMIT_GB * (1024**3) / (K * 16)))
            print(f"    分段近邻查询 (N={N} 过大, 每段 {CHUNK} 点, "
                  f"预估总量 {mem_estimate_gb:.1f} GB) ...")
            all_dists = np.empty((N, K), dtype=np.float32)
            all_knn_idx = np.empty((N, K), dtype=np.int64)
            for s in range(0, N, CHUNK):
                e = min(s + CHUNK, N)
                all_dists[s:e], all_knn_idx[s:e] = kdtree.query(
                    points[s:e], k=K, workers=-1)

        print(f"    KDTree 近邻查询完成, 耗时 {time.time() - t_kd:.1f}s")

        normals_out = np.zeros((N, 3), dtype=np.float32)
        rng = np.random.default_rng(seed=42)

        for batch_start in range(0, N, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, N)
            B = batch_end - batch_start
            batch_slice = slice(batch_start, batch_end)

            # ── 批量构建 patches (B, K, 3) ──
            seed_pts = points[batch_start:batch_end]          # (B, 3)
            dist_maxs = all_dists[batch_slice, -1].copy()     # (B,) float32
            dist_maxs[dist_maxs < 1e-8] = 1.0

            # (B, K, 3): 每个查询点的 K 近邻坐标，中心化 + 归一化
            nbr_pts = points[all_knn_idx[batch_slice]]        # (B, K, 3)
            patches = (nbr_pts - seed_pts[:, None, :]) / dist_maxs[:, None, None]

            # ── 批量 PCA 变换（无 Python 逐点循环）──
            patches_pca, trans_batch, cp_new_batch = self._batch_pca_transform(patches)

            # ── 批量全局 shape 采样 + PCA 变换（向量化，无循环）──
            # 注意：此处使用有放回采样 (rng.integers) 而非原版的无放回 (rng.choice)，
            # 以支持批量向量化。N >> S=700 时两者统计特性几乎等价，对网络推理无影响。
            patches_mean = patches.mean(axis=1)               # (B, 3)
            sample_indices = rng.integers(0, N, size=(B, self.SAMPLE_SIZE))  # (B, S)
            sample_pts = (points[sample_indices] - seed_pts[:, None, :]) \
                         / dist_maxs[:, None, None]                          # (B, S, 3)
            sample_centered = sample_pts - patches_mean[:, None, :]          # (B, S, 3)
            sample_rot = np.einsum('bsi,bij->bsj', sample_centered, trans_batch) \
                         - cp_new_batch[:, None, :]                          # (B, S, 3)

            # ── 组装 batch tensor ──
            pcl_pat_batch = torch.from_numpy(
                patches_pca
            ).float().to(self.device)                         # (B, K, 3)
            pcl_sample_batch = torch.from_numpy(
                sample_rot
            ).float().to(self.device)                         # (B, S, 3)
            pca_trans_batch = torch.from_numpy(
                trans_batch
            ).float().to(self.device)                         # (B, 3, 3)

            # ── 前向推理 ──
            with torch.no_grad():
                n_est = self.model(
                    pcl_pat_batch,
                    pcl_sample=pcl_sample_batch,
                    mode_test=True,
                )  # (B, 3) — 局部 PCA 坐标系下的法向

            # ── PCA 逆变换：回到世界坐标系 ──
            # trans 是世界→局部的正交矩阵，逆变换 = trans^T
            n_world = torch.bmm(
                n_est.unsqueeze(1),
                pca_trans_batch.transpose(1, 2),
            ).squeeze(1)  # (B, 3)

            normals_out[batch_start:batch_end] = n_world.cpu().numpy()

            if (batch_start // self.BATCH_SIZE) % 20 == 0:
                elapsed = time.time() - t0
                progress = batch_end / N * 100
                print(f"    [{progress:5.1f}%] {batch_end}/{N} 点已处理, "
                      f"已用时 {elapsed:.1f}s")

        # 归一化为单位向量
        norms_len = np.linalg.norm(normals_out, axis=1, keepdims=True)
        norms_len = np.maximum(norms_len, np.float32(1e-8))
        normals_out = normals_out / norms_len.astype(np.float32)

        print(f" -> SHS-Net 推理完成, 总耗时: {time.time() - t0:.1f}s")
        return normals_out, all_knn_idx


# ══════════════════════════════════════════════════════════════════════════════
# 第 3 步 ─ Screened Poisson 表面重建
# ══════════════════════════════════════════════════════════════════════════════

def _poisson_reconstruct(points: np.ndarray, normals: np.ndarray,
                         depth: int = 10, density_quantile: float = 0.05
                         ) -> o3d.geometry.TriangleMesh:
    """
    使用 Screened Poisson 算法从带有方向法向的点云重建三角网格。

    :param points:  (N, 3) 点云坐标
    :param normals: (N, 3) 有方向单位法向量
    :param depth:   八叉树最大深度（越大细节越丰富，10 ≈ 1024³ 分辨率）
    :param density_quantile: 密度过滤分位数，去除密度最低的该比例顶点。
                             Poisson 重建在远离点云的包围盒边界处会产生虚假面片，
                             通常 0.05~0.10 可有效去除；过小（如 0.01）会残留冗余面。
    :return: 清理后的三角网格
    """
    print(f" -> 正在执行 Screened Poisson 表面重建 (depth={depth}) ...")
    t0 = time.time()

    # 构造带法向的点云
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Poisson 重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=False
    )

    # ── 密度过滤：去除 Poisson 在远离点云处产生的冗余面片 ──
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # ── 包围盒裁剪：额外安全网，去除超出点云范围的离群面片 ──
    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
    bbox_margin = bbox_diag * 0.05  # 包围盒对角线长度的 5%，适应不同单位尺度
    pts_min = points.min(axis=0) - bbox_margin
    pts_max = points.max(axis=0) + bbox_margin
    verts = np.asarray(mesh.vertices)
    bbox_mask = np.any((verts < pts_min) | (verts > pts_max), axis=1)
    if np.any(bbox_mask):
        mesh.remove_vertices_by_mask(bbox_mask.tolist())
        print(f"    包围盒裁剪: 移除了 {bbox_mask.sum()} 个超出范围的顶点")

    # 网格后处理
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    num_verts = len(np.asarray(mesh.vertices))
    num_tris  = len(np.asarray(mesh.triangles))
    print(f"    重建完成: {num_verts} 顶点, {num_tris} 三角面元, "
          f"耗时 {time.time() - t0:.1f}s")
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# 第 4 步 ─ 构建射线场景 (与原版一致)
# ══════════════════════════════════════════════════════════════════════════════

def _build_ray_scene(mesh: o3d.geometry.TriangleMesh):
    """
    将三角网格注册到 Open3D RaycastingScene，供 Module 2 (视线遮挡) 和
    Module 5 (碰撞检测) 使用。
    """
    print(" -> 正在构建射线投射场景 (RaycastingScene) ...")
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)
    print("    射线场景构建完成")
    return ray_scene


# ══════════════════════════════════════════════════════════════════════════════
# 第 5 步 ─ PCA 曲率计算（用于自适应降采样）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_curvatures(points: np.ndarray, K: int = 30,
                        batch_size: int = 50_000,
                        precomputed_knn_idx: np.ndarray = None) -> np.ndarray:
    """
    向量化 PCA 曲率计算：基于邻域点坐标的协方差矩阵特征值比。

    方法与原版 Module 1 完全一致：对每个点取 K 近邻，计算协方差矩阵的
    最小特征值与特征值之和的比值（λ_min / Σλ），作为局部曲率的度量。
    该值接近 0 表示平坦，接近 1/3 表示各向同性（球面/角点）。

    注意：此函数仅基于坐标几何计算曲率，未直接使用 SHS-Net 的法向输出。
    SHS-Net 法向在后续用于 Poisson 重建和视点生成，此处的曲率仅用于
    自适应降采样的密度控制。

    :param points: (N, 3)
    :param K: 近邻数
    :param batch_size: 分批处理控制内存峰值
    :param precomputed_knn_idx: (N, K') 预计算的近邻索引（K' >= K 时可截取复用）
    :return: (N,) 曲率数组
    """
    N = len(points)
    print(f" -> 正在执行向量化 PCA 曲率计算 (N={N}, K={K}) ...")
    t0 = time.time()

    if precomputed_knn_idx is not None and precomputed_knn_idx.shape[1] >= K:
        # 复用 SHS-Net 阶段预计算的近邻索引，截取前 K 列
        knn_idx = precomputed_knn_idx[:, :K]
        print(f"    复用预计算近邻索引 (从 K={precomputed_knn_idx.shape[1]} 截取前 {K} 列)")
    else:
        tree = KDTree(points)
        _, knn_idx = tree.query(points, k=K, workers=-1)
        print(f"    KDTree 近邻查询完成, 耗时 {time.time() - t0:.1f}s")

    curvatures = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        nbrs = points[knn_idx[start:end]]              # (B, K, 3)
        ctr  = nbrs.mean(axis=1, keepdims=True)        # (B, 1, 3)
        ctrd = nbrs - ctr                              # (B, K, 3)
        covs = np.einsum('nki,nkj->nij', ctrd, ctrd) / K  # (B, 3, 3)
        eigs = np.linalg.eigvalsh(covs)                # (B, 3) 升序
        tr   = eigs.sum(axis=1)                        # (B,)
        curvatures[start:end] = np.where(tr > 1e-8, eigs[:, 0] / tr, 0.0)

    print(f"    曲率计算完成, 耗时 {time.time() - t0:.1f}s")
    return curvatures


# ══════════════════════════════════════════════════════════════════════════════
# 第 6 步 ─ 自适应降采样
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_downsample(points: np.ndarray, normals: np.ndarray,
                         curvatures: np.ndarray,
                         curvature_threshold: float = 0.015,
                         curved_ratio: float = 0.5,
                         flat_ratio: float = 0.1,
                         seed: int = 42):
    """
    基于曲率的自适应降采样：高曲率区域保留更多点，平坦区域大幅稀疏。

    :param points:    (N, 3)
    :param normals:   (N, 3)
    :param curvatures:(N,)
    :param curvature_threshold: 高/低曲率分界线
    :param curved_ratio: 高曲率区域保留比例
    :param flat_ratio:   低曲率区域保留比例
    :return: (pts_down, norms_down)
    """
    print(f" -> 自适应降采样 (阈值={curvature_threshold}, "
          f"高曲率保留={curved_ratio*100:.0f}%, 低曲率保留={flat_ratio*100:.0f}%) ...")

    curved_idx = np.where(curvatures >  curvature_threshold)[0]
    flat_idx   = np.where(curvatures <= curvature_threshold)[0]

    rng = np.random.default_rng(seed=seed)

    # 处理极端情况：某一组可能为空（全平坦/全高曲率点云）
    if len(curved_idx) == 0:
        print(f"    ⚠️ 无高曲率点（全部 ≤ {curvature_threshold}），仅从平坦区域采样")
        sel_all = rng.choice(flat_idx,
                             size=max(1, int(len(flat_idx) * flat_ratio)),
                             replace=False)
    elif len(flat_idx) == 0:
        print(f"    ⚠️ 无低曲率点（全部 > {curvature_threshold}），仅从高曲率区域采样")
        sel_all = rng.choice(curved_idx,
                             size=max(1, int(len(curved_idx) * curved_ratio)),
                             replace=False)
    else:
        sel_curved = rng.choice(curved_idx,
                                size=max(1, int(len(curved_idx) * curved_ratio)),
                                replace=False)
        sel_flat   = rng.choice(flat_idx,
                                size=max(1, int(len(flat_idx) * flat_ratio)),
                                replace=False)
        sel_all = np.concatenate([sel_curved, sel_flat])

    pts_down   = points[sel_all]
    norms_down = normals[sel_all]

    print(f"    高曲率点: {len(curved_idx)} | 低曲率点: {len(flat_idx)}")
    print(f"    降采样后总特征点数: {len(sel_all)}")
    return pts_down, norms_down


# ══════════════════════════════════════════════════════════════════════════════
# 主入口函数
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_pcd(file_path: str,
                            output_mesh_path: str = "airplane_reconstructed.ply",
                            poisson_depth: int = 10,
                            shs_device: str = "auto",
                            shs_ckpt: str = SHS_NET_CKPT):
    """
    新版 Module 1 主入口：从 .pcd 纯点云出发，完成预处理与特征提取。

    :param file_path:        输入 .pcd 文件路径
    :param output_mesh_path: 重建网格的输出路径
    :param poisson_depth:    Poisson 重建八叉树深度
    :param shs_device:       SHS-Net 推理设备 ("auto" / "cuda:0" / "cpu")
    :param shs_ckpt:         SHS-Net 预训练权重路径
    :return: (mesh, pts, norms, ray_scene)
             mesh      - Open3D TriangleMesh（供 Module 2/4/5 使用）
             pts       - (M, 3) 降采样后的特征点坐标
             norms     - (M, 3) 对应法向量
             ray_scene - Open3D RaycastingScene（供 Module 2/5 使用）
    """
    print(f"\n[Module 1] 启动预处理与特征提取 ({file_path}) ...")
    start_time = time.time()

    # ── 1. PCD 文件读取 ──────────────────────────────────────────────────
    raw_points = _load_pcd(file_path)

    # ── 2. SHS-Net 有方向法向推理 ────────────────────────────────────────
    estimator = SHSNetEstimator(ckpt_path=shs_ckpt, device=shs_device)
    raw_normals, knn_idx = estimator.estimate_normals(raw_points)

    # ── 3. Screened Poisson 表面重建 ─────────────────────────────────────
    mesh = _poisson_reconstruct(raw_points, raw_normals, depth=poisson_depth)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"    重建网格已保存: {output_mesh_path}")

    # ── 4. 构建射线场景 ─────────────────────────────────────────────────
    ray_scene = _build_ray_scene(mesh)

    # ── 5. PCA 曲率计算（复用 SHS-Net 阶段的近邻索引）──────────────────
    curvatures = _compute_curvatures(raw_points, K=30,
                                     precomputed_knn_idx=knn_idx)

    # ── 6. 自适应降采样 ─────────────────────────────────────────────────
    pts, norms = _adaptive_downsample(raw_points, raw_normals, curvatures)

    elapsed = time.time() - start_time
    print(f"\n[Module 1] 预处理完成！特征点数: {len(pts)}  总耗时: {elapsed:.1f}s")
    return mesh, pts, norms, ray_scene


# ──────────────────────────────────────────────────────────────────────────────
# 兼容性别名：保留原函数名，便于 main.py 无需改动即可调用
# 如果 main.py 调用的是 load_and_preprocess_mesh()，只需将该调用替换为
# load_and_preprocess_pcd()，或在此通过别名转发。
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_mesh(file_path: str,
                             output_path: str = "airplane_preprocessed.stl"):
    """
    兼容性入口（向后兼容 main.py 中的调用签名）。

    若 file_path 指向 .pcd 文件 → 走 SHS-Net + Poisson 新流水线。
    若 file_path 指向 .stl/.obj 等网格文件 → 抛出提示，引导切换到新接口。

    注意：新流水线输出的网格格式为 .ply（Poisson 重建产物），不再是 .stl。
    下游代码（如 main.py 中的 _load_processed_mesh）需要将 STL_PROCESSED
    改为 .ply 后缀。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pcd":
        # 将输出路径后缀改为 .ply
        output_ply = os.path.splitext(output_path)[0] + ".ply"
        return load_and_preprocess_pcd(file_path, output_mesh_path=output_ply)
    else:
        raise ValueError(
            f"[Module 1] 新版预处理模块仅接受 .pcd 点云文件，"
            f"收到的文件扩展名为 '{ext}'。\n"
            f"  请将输入替换为 .pcd 文件，或在 main.py 中修改 STL_INPUT 变量。\n"
            f"  例如: STL_INPUT = 'data/airplane_scan.pcd'"
        )


if __name__ == "__main__":
    # 独立运行示例
    import argparse
    parser = argparse.ArgumentParser(description="Module 1: PCD → SHS-Net → Poisson Mesh")
    parser.add_argument("input", type=str, help="输入 .pcd 文件路径")
    parser.add_argument("--output", type=str, default="airplane_reconstructed.ply",
                        help="输出网格路径 (默认: airplane_reconstructed.ply)")
    parser.add_argument("--depth", type=int, default=10,
                        help="Poisson 重建深度 (默认: 10)")
    parser.add_argument("--device", type=str, default="auto",
                        help="SHS-Net 推理设备 (默认: auto)")
    parser.add_argument("--ckpt", type=str, default=SHS_NET_CKPT,
                        help="SHS-Net 权重路径")
    args = parser.parse_args()

    mesh, pts, norms, ray_scene = load_and_preprocess_pcd(
        args.input,
        output_mesh_path=args.output,
        poisson_depth=args.depth,
        shs_device=args.device,
        shs_ckpt=args.ckpt,
    )
    print(f"\n完成! 特征点: {len(pts)}, 网格面元: {len(np.asarray(mesh.triangles))}")