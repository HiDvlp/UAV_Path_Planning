# module_2_viewpoint.py
import numpy as np
import open3d as o3d
import time
import scipy.spatial
from .config import Config

class ViewpointGenerator:
    # ⚠️ 注意：为了获取真实的物理面积，构造函数新增了 `mesh` 参数
    def __init__(self, ray_scene, mesh, target_points, target_normals):
        self.ray_scene = ray_scene
        self.mesh = mesh
        self.points = target_points
        self.normals = target_normals
        
        # 工业级相机物理与光学参数
        self.fov_h_deg = getattr(Config, 'FOV_DEG', 90.0)
        self.aspect_ratio = getattr(Config, 'ASPECT_RATIO', 1.5)
        self.max_incidence = getattr(Config, 'MAX_INCIDENCE_ANGLE', 45.0)
        
        # 距离与角度试探参数
        self.d_offsets = getattr(Config, 'PROBE_D_OFFSETS', [0.0, 0.5, 1.0, 1.5, 2.0])
        self.thetas = getattr(Config, 'PROBE_THETAS', [0.0, 15.0, 30.0, 45.0])
        
        # --- 物理面元属性预计算 ---
        print("[Module 2] 正在提取底层 CAD 网格的绝对物理面元属性...")
        self.mesh.compute_triangle_normals()
        self.tri_normals = np.asarray(self.mesh.triangle_normals)
        vertices = np.asarray(self.mesh.vertices)
        triangles = np.asarray(self.mesh.triangles)
        # 向量化极速计算所有三角形的真实物理面积
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        self.tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        
        # --- 视锥射线阵列 (数字传感器) 预生成 ---
        # 设定传感器分辨率 (如 200x133，每张照片发射 26600 条射线)
        self.ray_res_x = 200 
        self.ray_res_y = int(self.ray_res_x / self.aspect_ratio)
        u_max = np.tan(np.radians(self.fov_h_deg / 2.0))
        v_max = u_max / self.aspect_ratio
        
        u_vals = np.linspace(-u_max, u_max, self.ray_res_x)
        v_vals = np.linspace(-v_max, v_max, self.ray_res_y)
        uu, vv = np.meshgrid(u_vals, v_vals)
        # 归一化相机坐标系下的射线方向 [N_rays, 3]
        dir_cam = np.stack([uu.flatten(), vv.flatten(), np.ones_like(uu.flatten())], axis=-1)
        self.dir_cam = dir_cam / np.linalg.norm(dir_cam, axis=-1, keepdims=True)
        
        self.cos_max_inc = np.cos(np.radians(self.max_incidence))

    def _generate_polar_dirs(self, normals, thetas_deg, phis_deg):
        # (保持不变的极坐标扩散向量化代码)
        Z = normals
        W = np.array([0.0, 0.0, 1.0])
        T1 = np.cross(W, Z)
        norm_T1 = np.linalg.norm(T1, axis=1, keepdims=True)
        bad_idx = (norm_T1[:, 0] < 1e-3)
        if np.any(bad_idx):
            T1[bad_idx] = np.cross(np.array([1.0, 0.0, 0.0]), Z[bad_idx])
            norm_T1[bad_idx] = np.linalg.norm(T1[bad_idx], axis=1, keepdims=True)
        T1 /= norm_T1
        T2 = np.cross(Z, T1)

        dirs = []
        for theta in thetas_deg:
            th = np.radians(theta)
            for phi in phis_deg:
                ph = np.radians(phi)
                D = Z * np.cos(th) + np.sin(th) * (T1 * np.cos(ph) + T2 * np.sin(ph))
                dirs.append(D)
        return np.stack(dirs, axis=1)

    def _check_vp_validity(self, vps, targets, safe_radii):
        # (保持不变的三重物理安检)
        mask_z = vps[:, 2] >= Config.MIN_SAFE_Z
        dist_tensor = self.ray_scene.compute_distance(o3d.core.Tensor(vps, dtype=o3d.core.Dtype.Float32))
        mask_radius = dist_tensor.numpy() >= safe_radii
        dirs = targets - vps
        dists = np.linalg.norm(dirs, axis=1)
        valid_dirs = dists > 1e-5
        
        rays = np.zeros((len(vps), 6), dtype=np.float32)
        rays[:, :3] = vps
        dirs_norm = np.zeros_like(dirs)
        dirs_norm[valid_dirs] = dirs[valid_dirs] / dists[valid_dirs][:, None]
        rays[:, 3:] = dirs_norm
        
        ans = self.ray_scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans['t_hit'].numpy()
        mask_los = t_hit >= (dists - 0.1)
        
        return mask_z & mask_radius & mask_los

    def generate_candidates(self):
        print(f"\n[Module 2] 启动距离与极坐标交替试探算法 (1对1优选池)...")
        start_t = time.time()
        
        N = len(self.points)
        is_underbelly = self.normals[:, 2] < -0.1
        base_R = np.where(is_underbelly, Config.UNDERBELLY_CAMERA_DIST, Config.CAMERA_DISTANCE)
        safe_R = np.where(is_underbelly, Config.UNDERBELLY_SAFE_RADIUS, Config.SAFE_RADIUS)
        
        is_solved = np.zeros(N, dtype=bool)
        best_vps = np.zeros((N, 3))
        
        # ==========================================
        # 🌟 阶段 1：距离与角度交替试探生成
        # ==========================================
        for d_off in self.d_offsets:
            if np.all(is_solved): break
            print(f" -> [距离探测] 尝试拉远 {d_off} 米...")
            for theta in self.thetas:
                idx_unsolved = np.where(~is_solved)[0]
                if len(idx_unsolved) == 0: break
                
                curr_R = base_R[idx_unsolved] + d_off
                curr_targets = self.points[idx_unsolved]
                curr_normals = self.normals[idx_unsolved]
                curr_safe_R = safe_R[idx_unsolved]
                
                phis = [0.0] if theta == 0.0 else [0, 45, 90, 135, 180, 225, 270, 315]
                num_dirs = len(phis)
                
                dirs = self._generate_polar_dirs(curr_normals, [theta], phis)
                vps = curr_targets[:, None, :] + dirs * curr_R[:, None, None]
                
                vps_flat = vps.reshape(-1, 3)
                targets_flat = np.repeat(curr_targets, num_dirs, axis=0)
                safe_flat = np.repeat(curr_safe_R, num_dirs, axis=0)
                
                mask_flat = self._check_vp_validity(vps_flat, targets_flat, safe_flat)
                mask_2d = mask_flat.reshape(-1, num_dirs)
                
                has_valid = np.any(mask_2d, axis=1)
                first_valid_idx = np.argmax(mask_2d, axis=1) 
                
                solved_local = np.where(has_valid)[0]
                solved_global = idx_unsolved[solved_local]
                
                best_vps[solved_global] = vps[solved_local, first_valid_idx[solved_local], :]
                is_solved[solved_global] = True

        valid_vps_pool = best_vps[is_solved]
        pcd_vps = o3d.geometry.PointCloud()
        pcd_vps.points = o3d.utility.Vector3dVector(valid_vps_pool)
        pcd_vps = pcd_vps.voxel_down_sample(voxel_size=0.5)
        all_vps = np.asarray(pcd_vps.points)
        print(f" -> 提纯后候选视点池: {len(all_vps)} 个。")

        # ==========================================
        # 📷 阶段 2：向量化批量虚拟相机射线阵列评价
        # ==========================================
        print(f" -> 正在启动高并发视锥射线阵列，计算真实三角面元覆盖与物理质量矩阵...")
        eval_start_t = time.time()

        K          = len(all_vps)
        R          = len(self.dir_cam)
        INVALID_ID = 4294967295

        # ── ① 批量 KDTree 查询（一次替代 K 次单点查询）────────────────
        tree = scipy.spatial.KDTree(self.points)
        _, nearest_idxs = tree.query(all_vps)        # (K,)
        lookat_pts = self.points[nearest_idxs]        # (K, 3)

        # ── ② 向量化计算所有视点相机坐标系 ────────────────────────────
        Z_cams  = lookat_pts - all_vps               # (K, 3)
        norms_z = np.linalg.norm(Z_cams, axis=1)    # (K,)
        vp_valid = norms_z >= 1e-5
        Z_cams[vp_valid] /= norms_z[vp_valid, None]

        # 同时算出两种 X 轴备选，再按近竖直与否选择
        X_from_W  = np.cross([0.0, 0.0, 1.0], Z_cams)   # (K, 3)
        X_from_e1 = np.cross([1.0, 0.0, 0.0], Z_cams)   # (K, 3)
        near_vert = np.abs(Z_cams[:, 2]) > 0.999
        X_raw     = np.where(near_vert[:, None], X_from_e1, X_from_W)   # (K, 3)
        X_cams    = X_raw / np.maximum(np.linalg.norm(X_raw, axis=1, keepdims=True), 1e-8)
        Y_cams    = np.cross(Z_cams, X_cams)             # (K, 3)

        # ── ③ 分批射线投射（每批 CHUNK 个视点，控制内存峰值）──────────
        CHUNK      = 64
        ideal_dist = Config.CAMERA_DISTANCE
        sigma2     = Config.DIST_SCORE_SIGMA2
        coverage_dict_raw = {}

        for chunk_start in range(0, K, CHUNK):
            chunk_end   = min(chunk_start + CHUNK, K)
            C           = chunk_end - chunk_start
            chunk_valid = vp_valid[chunk_start:chunk_end]
            if not np.any(chunk_valid):
                continue

            c_vps = all_vps[chunk_start:chunk_end]    # (C, 3)
            c_X   = X_cams[chunk_start:chunk_end]     # (C, 3)
            c_Y   = Y_cams[chunk_start:chunk_end]
            c_Z   = Z_cams[chunk_start:chunk_end]

            # 世界坐标系射线方向 (C, R, 3)
            D_world = (self.dir_cam[None, :, 0:1] * c_X[:, None, :] +
                       self.dir_cam[None, :, 1:2] * c_Y[:, None, :] +
                       self.dir_cam[None, :, 2:3] * c_Z[:, None, :])

            # 拼成 (C*R, 6) 射线张量并批量投射
            origins = np.repeat(c_vps, R, axis=0)     # (C*R, 3)
            rays    = np.concatenate(
                [origins, D_world.reshape(-1, 3)], axis=1
            ).astype(np.float32)

            ans           = self.ray_scene.cast_rays(o3d.core.Tensor(rays))
            hit_ids_all   = ans['primitive_ids'].numpy()   # (C*R,)
            hit_dists_all = ans['t_hit'].numpy()           # (C*R,)

            for j in range(C):
                if not chunk_valid[j]:
                    continue
                vp_idx = chunk_start + j
                s, e   = j * R, (j + 1) * R
                hit_ids   = hit_ids_all[s:e]
                hit_dists = hit_dists_all[s:e]
                dirs_j    = D_world[j]                    # (R, 3)

                valid_mask = hit_ids != INVALID_ID
                if not np.any(valid_mask):
                    continue

                valid_ids   = hit_ids[valid_mask]
                valid_dists = hit_dists[valid_mask]
                valid_dirs  = dirs_j[valid_mask]

                unique_tris, uniq_idx = np.unique(valid_ids, return_index=True)
                dists     = valid_dists[uniq_idx]
                dirs      = valid_dirs[uniq_idx]

                tri_norms = self.tri_normals[unique_tris]
                cos_inc   = np.sum(-dirs * tri_norms, axis=1)
                inc_mask  = cos_inc >= self.cos_max_inc
                if not np.any(inc_mask):
                    continue

                final_tris  = unique_tris[inc_mask]
                final_cos   = cos_inc[inc_mask]
                final_dists = dists[inc_mask]
                areas       = self.tri_areas[final_tris]

                dist_score = np.exp(-((final_dists - ideal_dist) ** 2) / sigma2)
                q_scores   = areas * final_cos * dist_score
                coverage_dict_raw[vp_idx] = {
                    int(tid): float(sc) for tid, sc in zip(final_tris, q_scores)
                }

            if chunk_start % (CHUNK * 8) == 0:
                print(f"    [{chunk_end / K * 100:5.1f}%] {chunk_end}/{K} 视点已处理")

        # ── ④ 过滤空视点并重新连续编号 ─────────────────────────────────
        final_valid_vps    = []
        final_coverage_dict = {}
        for old_idx in range(K):
            tri_scores = coverage_dict_raw.get(old_idx)
            if tri_scores:
                new_idx = len(final_valid_vps)
                final_valid_vps.append(all_vps[old_idx])
                final_coverage_dict[new_idx] = tri_scores

        final_valid_vps = (np.array(final_valid_vps)
                           if final_valid_vps else np.empty((0, 3)))

        # ── ⑤ 真实物理表面积覆盖率统计 ─────────────────────────────────
        all_covered_tris = set()
        for tri_scores in final_coverage_dict.values():
            all_covered_tris.update(tri_scores.keys())

        total_mesh_area = np.sum(self.tri_areas)
        covered_area    = (np.sum(self.tri_areas[list(all_covered_tris)])
                           if all_covered_tris else 0.0)
        coverage_rate   = (covered_area / total_mesh_area * 100
                           if total_mesh_area > 0 else 0.0)

        print(f"✔️ 数字相机扫描完毕！耗时: {time.time() - eval_start_t:.2f} 秒")
        print(f"✔️ 提纯后有效视点数: {len(final_valid_vps)}")
        print(f"📊 飞机总表面积: {total_mesh_area:.2f} m² | 实际覆盖面积: {covered_area:.2f} m²")
        print(f"🌟 真实物理表面覆盖率: {coverage_rate:.2f}%")
        print(f"[Module 2] 总耗时: {time.time() - start_t:.2f} 秒")

        return final_valid_vps, final_coverage_dict