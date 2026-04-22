# module_2_viewpoint.py
import numpy as np
import open3d as o3d
import time
import scipy.spatial
from config import Config

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
        # 📷 阶段 2：数字虚拟相机射线阵列评价 (革命性升级)
        # ==========================================
        print(f" -> 正在启动高并发视锥射线阵列，计算真实三角面元覆盖与物理质量矩阵...")
        eval_start_t = time.time()
        coverage_dict = {}
        tree = scipy.spatial.KDTree(self.points)
        
        W = np.array([0.0, 0.0, 1.0]) # 保证 Roll = 0
        ideal_dist = Config.CAMERA_DISTANCE # 理想焦距，用于 GSD 衰减
        
        # O3D 射线未命中时的无效 ID 是 2^32 - 1
        INVALID_ID = 4294967295

        for i, Vc in enumerate(all_vps):
            # 获取光轴方向
            _, nearest_idx = tree.query(Vc)
            LookAt = self.points[nearest_idx]
            Z_cam = LookAt - Vc
            norm_Z = np.linalg.norm(Z_cam)
            if norm_Z < 1e-5: continue
            Z_cam /= norm_Z 
            
            if np.abs(Z_cam[2]) > 0.999:
                X_cam = np.array([1.0, 0.0, 0.0])
            else:
                X_cam = np.cross(W, Z_cam)
                X_cam /= np.linalg.norm(X_cam)
            Y_cam = np.cross(Z_cam, X_cam)
            
            # 将归一化射线转换到世界坐标系 (万箭齐发)
            # dir_world = dir_cam_x * X + dir_cam_y * Y + dir_cam_z * Z
            D_world = (self.dir_cam[:, 0:1] * X_cam + 
                       self.dir_cam[:, 1:2] * Y_cam + 
                       self.dir_cam[:, 2:3] * Z_cam)
            
            # 构建并批量发射射线
            rays = np.zeros((len(D_world), 6), dtype=np.float32)
            rays[:, :3] = Vc
            rays[:, 3:] = D_world
            
            ans = self.ray_scene.cast_rays(o3d.core.Tensor(rays))
            hit_ids = ans['primitive_ids'].numpy()
            hit_dists = ans['t_hit'].numpy()
            
            # 过滤掉打向天空的无效射线
            valid_mask = hit_ids != INVALID_ID
            if not np.any(valid_mask): continue
            
            valid_ids = hit_ids[valid_mask]
            valid_dists = hit_dists[valid_mask]
            valid_dirs = D_world[valid_mask]
            
            # 提取被击中的【唯一】三角面元，并获取它们对应的那条光线的数据
            unique_tris, unique_indices = np.unique(valid_ids, return_index=True)
            dists = valid_dists[unique_indices]
            dirs = valid_dirs[unique_indices]
            
            # 获取这些面元的真实法向，计算入射角余弦值
            tri_norms = self.tri_normals[unique_tris]
            cos_inc = np.sum(-dirs * tri_norms, axis=1) # 点积
            
            # 入射角红线过滤 (超过 MAX_INCIDENCE_ANGLE 的直接判 0 分不记录)
            inc_mask = cos_inc >= self.cos_max_inc
            if not np.any(inc_mask): continue
            
            final_tris = unique_tris[inc_mask]
            final_cos = cos_inc[inc_mask]
            final_dists = dists[inc_mask]
            
            # 获取这些面元的真实物理面积
            areas = self.tri_areas[final_tris]
            
            # 核心：计算 Q(i,j) 质量分数
            # 1. 角度得分：cos(alpha)
            # 2. 距离得分：高斯衰减 exp(-(d - D_opt)^2 / 8.0)
            dist_score = np.exp(-((final_dists - ideal_dist)**2) / 8.0)
            
            q_scores = areas * final_cos * dist_score
            
            # 组装字典 {tri_id: q_score}
            # 使用字典推导式，这不仅符合 Module 3 的迭代习惯，还能完美存储连续质量权重
            coverage_dict[i] = {tri_id: float(score) for tri_id, score in zip(final_tris, q_scores)}

        # 剔除未能提供任何有效覆盖的空视点
        final_valid_vps = []
        final_coverage_dict = {}
        for old_idx, tri_scores in coverage_dict.items():
            if len(tri_scores) > 0:
                new_idx = len(final_valid_vps)
                final_valid_vps.append(all_vps[old_idx])
                final_coverage_dict[new_idx] = tri_scores
            
        final_valid_vps = np.array(final_valid_vps)
        
        # ==========================================
        # 📊 真实物理表面积覆盖率统计
        # ==========================================
        all_covered_tris = set()
        for tri_scores in final_coverage_dict.values():
            all_covered_tris.update(tri_scores.keys())
            
        total_mesh_area = np.sum(self.tri_areas)
        covered_area = np.sum(self.tri_areas[list(all_covered_tris)])
        coverage_rate = (covered_area / total_mesh_area) * 100
        
        print(f"✔️ 数字相机扫描完毕！耗时: {time.time() - eval_start_t:.2f} 秒")
        print(f"✔️ 提纯后有效视点数: {len(final_valid_vps)}")
        print(f"📊 飞机总表面积: {total_mesh_area:.2f} m² | 实际覆盖面积: {covered_area:.2f} m²")
        print(f"🌟 真实物理表面覆盖率: {coverage_rate:.2f}%")
        print(f"[Module 2] 总耗时: {time.time() - start_t:.2f} 秒")

        return final_valid_vps, final_coverage_dict