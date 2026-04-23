# module_5_trajectory_optimization.py
#
# 三层架构：
#   Layer 1 - CollisionChecker    : 基于 Open3D 的距离场，判定任意点/线段是否满足安全半径
#   Layer 2 - VoxelAStarPlanner   : 体素化 A* 在安全空间内规划无碰撞折线路径
#   Layer 3 - UAVTrajectoryPlanner: 串联以上两层，生成带时间戳的完整 setpoint 序列
#
# 设计原则：
#   - 视点位置 (x,y,z) 和偏航角 (yaw) 为硬约束，在 setpoint 中精确出现
#   - 视点处强制悬停 HOVER_TIME 秒，速度=0（"停走停"模式，对 PX4 最友好）
#   - 转场段 yaw 保持出发视点的 yaw 值（视点间无 yaw 要求）
#   - 全程距离飞机蒙皮 >= FLIGHT_SAFE_RADIUS

import numpy as np
import open3d as o3d
import heapq
import time
import os
import csv

from .config import Config


# ============================================================
# Layer 1: CollisionChecker
# ============================================================

class CollisionChecker:
    """
    基于 Open3D RaycastingScene 的碰撞检测器。
    使用 compute_distance() 获取精确的点-蒙皮最近距离，
    无需射线投射，对大批量查询极为高效。
    """

    def __init__(self, mesh):
        print("\n[Module 5 - CollisionChecker] 正在构建碰撞检测场...")
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(mesh_t)
        self.safe_radius = Config.FLIGHT_SAFE_RADIUS
        print(f"  -> 碰撞场构建完成。全程飞行安全半径: {self.safe_radius} m")

    def distance_to_surface(self, pts: np.ndarray) -> np.ndarray:
        """
        批量计算点集到网格蒙皮的最近距离。
        :param pts: (N, 3) float 数组
        :return:    (N,)  距离数组（单位：米）
        """
        t = o3d.core.Tensor(pts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        return self.scene.compute_distance(t).numpy()

    def is_safe(self, pt: np.ndarray) -> bool:
        """单点安全判定：距蒙皮 >= FLIGHT_SAFE_RADIUS 返回 True"""
        return float(self.distance_to_surface(pt.reshape(1, 3))[0]) >= self.safe_radius

    def is_safe_segment(self, p1: np.ndarray, p2: np.ndarray, n_samples: int = 12) -> bool:
        """
        线段安全判定：在 p1->p2 上均匀采样 n_samples 个点，
        全部满足安全半径则返回 True。
        """
        ts = np.linspace(0.0, 1.0, n_samples)
        pts = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        return bool(np.all(self.distance_to_surface(pts.astype(np.float32)) >= self.safe_radius))


# ============================================================
# Layer 2: VoxelAStarPlanner
# ============================================================

class VoxelAStarPlanner:
    """
    体素化 A* 路径规划器。

    工作流程：
      1. 在飞机包围盒（含安全余量）内建立均匀体素网格
      2. 批量查询每个体素中心到蒙皮的距离，标记安全/不安全
      3. 对于任意两个视点，先尝试直飞（快速路），失败则启动 A* 搜索
      4. 对原始折线路径做贪心可视性剪枝，减少不必要的中间节点
    """

    def __init__(self, collision_checker: CollisionChecker, mesh):
        self.checker = collision_checker
        self.voxel_size = Config.VOXEL_SIZE

        # 包围盒 + 安全余量作为搜索空间
        aabb = mesh.get_axis_aligned_bounding_box()
        margin = self.checker.safe_radius * 2.0
        self.origin = np.asarray(aabb.min_bound) - margin
        max_bound  = np.asarray(aabb.max_bound) + margin

        extent = max_bound - self.origin
        self.grid_shape = (np.ceil(extent / self.voxel_size).astype(int) + 1).tolist()

        print(f"\n[Module 5 - VoxelAStarPlanner] 正在构建体素安全图...")
        print(f"  -> 体素分辨率: {self.voxel_size} m | 网格尺寸: {self.grid_shape}"
              f" = {int(np.prod(self.grid_shape))} 个体素")

        self._build_safe_grid()

        # 26-连通性邻域偏移（含斜向）
        self._neighbor_offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        self._neighbor_costs = [
            np.sqrt(dx*dx + dy*dy + dz*dz) * self.voxel_size
            for dx, dy, dz in self._neighbor_offsets
        ]

    # ── 安全体素图构建 ──────────────────────────────────────────

    def _build_safe_grid(self):
        """
        向量化批量计算所有体素中心的蒙皮距离，建立 bool 安全图。
        分批处理防止内存溢出（约 50 000 点/批）。
        """
        start_t = time.time()
        nx, ny, nz = self.grid_shape

        xi = np.arange(nx)
        yi = np.arange(ny)
        zi = np.arange(nz)
        gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')

        centers = np.stack([
            self.origin[0] + gx.ravel() * self.voxel_size,
            self.origin[1] + gy.ravel() * self.voxel_size,
            self.origin[2] + gz.ravel() * self.voxel_size,
        ], axis=1).astype(np.float32)

        BATCH = 50_000
        dist_chunks = []
        for i in range(0, len(centers), BATCH):
            dist_chunks.append(self.checker.distance_to_surface(centers[i:i + BATCH]))
        dists = np.concatenate(dist_chunks)

        safe_flat = dists >= self.checker.safe_radius
        self.safe_grid = safe_flat.reshape(self.grid_shape)

        # 强制地面以下体素不安全
        z_centers = self.origin[2] + zi * self.voxel_size
        unsafe_z = z_centers < Config.MIN_SAFE_Z
        if np.any(unsafe_z):
            self.safe_grid[:, :, unsafe_z] = False

        safe_n = int(np.sum(self.safe_grid))
        total_n = int(np.prod(self.grid_shape))
        print(f"  -> 安全体素图构建完成！"
              f"安全/总计 = {safe_n}/{total_n} ({100*safe_n/total_n:.1f}%)"
              f" | 耗时: {time.time()-start_t:.1f}s")

    # ── 坐标转换 ────────────────────────────────────────────────

    def _pt_to_idx(self, pt: np.ndarray) -> tuple:
        """世界坐标 -> 体素索引（边界裁剪）"""
        raw = np.round((pt - self.origin) / self.voxel_size).astype(int)
        clipped = np.clip(raw, 0, np.array(self.grid_shape) - 1)
        return tuple(clipped.tolist())

    def _idx_to_pt(self, idx: tuple) -> np.ndarray:
        """体素索引 -> 体素中心世界坐标"""
        return self.origin + np.array(idx, dtype=float) * self.voxel_size

    # ── A* 核心 ─────────────────────────────────────────────────

    def _heuristic(self, a: tuple, b: tuple) -> float:
        return (abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])) * self.voxel_size

    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """
        规划 start -> goal 的无碰撞路径。
        优先尝试直飞（O(n) 线段检测），失败则启动完整 A*。

        :return: (M, 3) 路径点数组，首尾精确等于 start / goal
        """
        # 快速路：直飞检测
        if self.checker.is_safe_segment(start, goal, n_samples=20):
            return np.array([start, goal])

        s_idx = self._pt_to_idx(start)
        g_idx = self._pt_to_idx(goal)

        if not self.safe_grid[s_idx]:
            s_idx = self._nearest_safe(s_idx)
        if not self.safe_grid[g_idx]:
            g_idx = self._nearest_safe(g_idx)

        # A* 搜索
        counter   = 0
        open_heap = []
        heapq.heappush(open_heap, (0.0, counter, s_idx))

        g_score   = {s_idx: 0.0}
        came_from = {}
        closed    = set()
        found     = False

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == g_idx:
                found = True
                break

            cx, cy, cz = current
            nx_bound, ny_bound, nz_bound = self.grid_shape

            for (dx, dy, dz), geom_cost in zip(self._neighbor_offsets, self._neighbor_costs):
                nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
                if not (0 <= nx_ < nx_bound and
                        0 <= ny_ < ny_bound and
                        0 <= nz_ < nz_bound):
                    continue
                nb = (nx_, ny_, nz_)
                if nb in closed or not self.safe_grid[nb]:
                    continue

                # 高度代价（论文公式5）：偏好低空 + 偏好平稳高度
                nb_z      = self.origin[2] + nz_ * self.voxel_size
                alt_cost  = (Config.WEIGHT_ALT_MEAN * nb_z +
                             Config.WEIGHT_ALT_VAR  * abs(dz) * self.voxel_size)
                step_cost = geom_cost + alt_cost

                tent_g = g_score[current] + step_cost
                if nb not in g_score or tent_g < g_score[nb]:
                    g_score[nb]   = tent_g
                    came_from[nb] = current
                    f = tent_g + self._heuristic(nb, g_idx)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, nb))

        if not found:
            print(f"  ⚠️  A* 未能找到路径（{start} -> {goal}），退化为直线。"
                  f"请检查 FLIGHT_SAFE_RADIUS 或 VOXEL_SIZE。")
            return np.array([start, goal])

        # 回溯
        path_idx = []
        cur = g_idx
        while cur in came_from:
            path_idx.append(cur)
            cur = came_from[cur]
        path_idx.append(s_idx)
        path_idx.reverse()

        path_pts       = np.array([self._idx_to_pt(i) for i in path_idx])
        path_pts[0]    = start
        path_pts[-1]   = goal

        return self._prune(path_pts)

    def _nearest_safe(self, idx: tuple, radius: int = 6) -> tuple:
        """在体素邻域内寻找最近安全体素（向量化替代三重 Python 循环）。"""
        x, y, z   = idx
        nx, ny, nz = self.grid_shape

        xs = np.arange(max(0, x - radius), min(nx, x + radius + 1))
        ys = np.arange(max(0, y - radius), min(ny, y + radius + 1))
        zs = np.arange(max(0, z - radius), min(nz, z + radius + 1))

        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (M, 3)

        safe_mask   = self.safe_grid[coords[:, 0], coords[:, 1], coords[:, 2]]
        safe_coords = coords[safe_mask]

        if len(safe_coords) == 0:
            return idx

        diffs = safe_coords - np.array([x, y, z])
        best  = safe_coords[(diffs * diffs).sum(axis=1).argmin()]
        return tuple(best.tolist())

    def _prune(self, path: np.ndarray) -> np.ndarray:
        """
        贪心可视性剪枝：从当前节点出发尽量向后跳，
        若两节点间可直飞则跳过所有中间节点。
        """
        if len(path) <= 2:
            return path
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self.checker.is_safe_segment(path[i], path[j], n_samples=15):
                    break
                j -= 1
            pruned.append(path[j])
            i = j
        return np.array(pruned)


# ============================================================
# 平滑工具：Catmull-Rom 样条
# ============================================================

def _catmull_rom_segment(p0, p1, p2, p3, n: int = 20) -> np.ndarray:
    """在 p1->p2 段生成 n 个 Catmull-Rom 插值点（不含终点）"""
    ts  = np.linspace(0.0, 1.0, n, endpoint=False)
    t2  = ts * ts
    t3  = t2 * ts
    pts = 0.5 * (
        2.0 * p1
        + (-p0 + p2)                 * ts[:, None]
        + (2*p0 - 5*p1 + 4*p2 - p3) * t2[:, None]
        + (-p0 + 3*p1 - 3*p2 + p3)  * t3[:, None]
    )
    return pts


def _smooth_catmull_rom(path: np.ndarray, pts_per_seg: int = 20) -> np.ndarray:
    """
    对折线路径整体施加 Catmull-Rom 样条平滑。
    首尾各添加反向幻影控制点以保证边界切线方向正确。
    """
    if len(path) < 2:
        return path
    if len(path) == 2:
        ts = np.linspace(0.0, 1.0, pts_per_seg)
        return path[0] + ts[:, None] * (path[1] - path[0])

    phantom_start = path[0]  + (path[0]  - path[1])
    phantom_end   = path[-1] + (path[-1] - path[-2])
    ext           = np.vstack([phantom_start, path, phantom_end])

    pieces = []
    for i in range(1, len(ext) - 2):
        pieces.append(_catmull_rom_segment(
            ext[i-1], ext[i], ext[i+1], ext[i+2], n=pts_per_seg
        ))
    pieces.append(path[-1:])          # 精确终点
    return np.vstack(pieces)


def _altitude_cost(trajectory: list) -> tuple:
    """
    按论文公式(5)计算单机完整轨迹的高度代价及分量：
        J_altitude = h1 * ȳ + h2 * Var(y)
    其中 ȳ 为所有 setpoint 的平均高度，Var(y) 为高度方差。

    :return: (J_altitude, z_mean, z_var)
    """
    zs     = np.array([row[3] for row in trajectory], dtype=float)
    z_mean = float(np.mean(zs))
    z_var  = float(np.mean((zs - z_mean) ** 2))
    j_alt  = Config.WEIGHT_ALT_MEAN * z_mean + Config.WEIGHT_ALT_VAR * z_var
    return j_alt, z_mean, z_var


def build_smooth_path(
    astar_planner: VoxelAStarPlanner,
    route_pts: np.ndarray,
) -> np.ndarray:
    """
    对完整航线做逐段 A* 规划 + Catmull-Rom 平滑，返回拼接后的空间几何路径。
    :param route_pts: (N, 3) 完整航点序列（含起飞点首尾）
    :return: (M, 3) 平滑路径点数组
    """
    segments = []
    for i in range(len(route_pts) - 1):
        raw_path    = astar_planner.plan(route_pts[i], route_pts[i + 1])
        smooth_path = _smooth_catmull_rom(raw_path, pts_per_seg=20)
        if i < len(route_pts) - 2:
            smooth_path = smooth_path[:-1]
        segments.append(smooth_path)
    return np.vstack(segments) if segments else np.asarray(route_pts)


# ============================================================
# Layer 3: UAVTrajectoryPlanner
# ============================================================

class UAVTrajectoryPlanner:
    """
    单架无人机完整轨迹生成器。

    输出格式（每行一个 setpoint）：
      timestamp_s, x, y, z, yaw_rad, type

    type 取值：
      TAKEOFF        —— 起飞点悬停
      WAYPOINT_START —— 视点悬停起始帧（触发拍照）
      WAYPOINT       —— 视点悬停中间帧
      WAYPOINT_END   —— 视点悬停结束帧
      TRANSIT        —— 视点间转场飞行
      LAND           —— 返回起飞点后的落地等待帧
    """

    def __init__(self,
                 collision_checker: CollisionChecker,
                 astar_planner:     VoxelAStarPlanner):
        self.checker   = collision_checker
        self.planner   = astar_planner
        self.speed     = Config.TRANSIT_SPEED     # 转场速度 (m/s)
        self.hover_t   = Config.HOVER_TIME        # 视点悬停时长 (s)
        self.freq      = Config.CSV_FREQ          # 输出频率 (Hz)
        self.dt        = 1.0 / self.freq

    # ── 主入口 ──────────────────────────────────────────────────

    def build_trajectory(self,
                         uav_id:     int,
                         route_pts:  np.ndarray,
                         route_yaws: np.ndarray) -> list:
        """
        构建单架无人机的完整时间戳 setpoint 序列。

        :param uav_id:     无人机编号（仅用于日志）
        :param route_pts:  (N, 3)  完整航点序列（含起飞点首尾）
        :param route_yaws: (N,)    对应偏航角（弧度）
                           约定：起飞点 yaw=0，返回点 yaw=0，视点处为精确值
        :return: list of (timestamp_s, x, y, z, yaw_rad, type)
        """
        assert len(route_pts) == len(route_yaws), \
            "route_pts 与 route_yaws 长度必须一致"

        n_wps = len(route_pts)
        print(f"\n[UAV {uav_id}] 开始构建轨迹 (共 {n_wps} 个航点，{n_wps-1} 段)...")

        trajectory   = []
        current_time = 0.0

        for seg in range(n_wps - 1):
            p_start  = route_pts[seg]
            p_goal   = route_pts[seg + 1]
            y_start  = route_yaws[seg]
            y_goal   = route_yaws[seg + 1]

            # ── ① 当前点悬停 ───────────────────────────────────
            if seg == 0:
                hover_secs = 1.0          # 起飞点短暂稳定悬停
                base_type  = "TAKEOFF"
            elif seg == n_wps - 2:
                # 返回起飞点前的最后一个真实视点：完整悬停
                hover_secs = self.hover_t
                base_type  = "WAYPOINT"
            else:
                hover_secs = self.hover_t
                base_type  = "WAYPOINT"

            current_time = self._append_hover(
                trajectory, current_time,
                p_start, y_start, hover_secs, base_type
            )

            # ── ② A* + Catmull-Rom 转场 ────────────────────────
            print(f"  [Seg {seg+1:3d}/{n_wps-1}] "
                  f"{np.round(p_start,1)} -> {np.round(p_goal,1)} ...")
            raw_path    = self.planner.plan(p_start, p_goal)
            smooth_path = _smooth_catmull_rom(raw_path, pts_per_seg=20)

            current_time = self._append_transit(
                trajectory, current_time,
                smooth_path, y_start          # 转场段保持出发视点 yaw
            )

        # ── ③ 最后一个航点（返回起飞点）的悬停/落地等待 ─────────
        current_time = self._append_hover(
            trajectory, current_time,
            route_pts[-1], route_yaws[-1],
            hover_secs=2.0, base_type="LAND"
        )

        j_alt, z_mean, z_var = _altitude_cost(trajectory)
        print(f"  [UAV {uav_id}] 轨迹构建完毕！"
              f"总时长: {current_time:.1f}s | Setpoint 数: {len(trajectory)}")
        print(f"  [UAV {uav_id}] J_altitude = {j_alt:.4f}  "
              f"(均值 ȳ={z_mean:.3f}m  方差 Var={z_var:.3f}m²)")
        return trajectory

    # ── 工具：悬停段生成 ────────────────────────────────────────

    def _append_hover(self,
                      traj:       list,
                      t0:         float,
                      pt:         np.ndarray,
                      yaw:        float,
                      hover_secs: float,
                      base_type:  str) -> float:
        """
        在当前时刻起，向 traj 追加一段静止悬停的 setpoints。
        首帧标记 {base_type}_START，末帧标记 {base_type}_END，
        中间帧标记 {base_type}。
        """
        n = max(1, int(round(hover_secs / self.dt)))
        for k in range(n):
            if   k == 0:       row_type = f"{base_type}_START"
            elif k == n - 1:   row_type = f"{base_type}_END"
            else:              row_type = base_type

            traj.append((
                round(t0, 4),
                round(float(pt[0]), 4),
                round(float(pt[1]), 4),
                round(float(pt[2]), 4),
                round(float(yaw),   6),
                row_type
            ))
            t0 += self.dt
        return t0

    # ── 工具：转场段生成 ────────────────────────────────────────

    def _append_transit(self,
                        traj:        list,
                        t0:          float,
                        smooth_path: np.ndarray,
                        yaw:         float) -> float:
        """
        将平滑路径按 TRANSIT_SPEED 等弧长重采样为离散 setpoints，
        追加到 traj 中（全部标记为 TRANSIT）。
        yaw 在整段转场中保持恒定（等于出发视点的 yaw）。
        """
        if len(smooth_path) < 2:
            return t0

        diffs   = np.diff(smooth_path, axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cumlen  = np.concatenate([[0.0], np.cumsum(seg_len)])
        total   = float(cumlen[-1])

        if total < 1e-4:
            return t0

        n_steps = max(1, int(total / self.speed / self.dt))
        s_vals  = np.linspace(0.0, total, n_steps)

        # 向量化弧长参数化插值：批量计算所有采样点坐标
        idxs    = np.searchsorted(cumlen, s_vals, side='right') - 1
        idxs    = np.clip(idxs, 0, len(smooth_path) - 2)
        segs    = seg_len[idxs]
        t_loc   = np.where(segs < 1e-8, 0.0, (s_vals - cumlen[idxs]) / segs)
        pts_all = smooth_path[idxs] + t_loc[:, None] * diffs[idxs]   # (n_steps, 3)
        ts_all  = t0 + np.arange(n_steps) * self.dt
        yaw_r   = round(float(yaw), 6)

        for k in range(n_steps):
            traj.append((
                round(float(ts_all[k]), 4),
                round(float(pts_all[k, 0]), 4),
                round(float(pts_all[k, 1]), 4),
                round(float(pts_all[k, 2]), 4),
                yaw_r,
                "TRANSIT",
            ))

        return float(ts_all[-1]) + self.dt


# ============================================================
# 导出工具
# ============================================================

def export_trajectory_csv(uav_id:    int,
                          trajectory: list,
                          output_dir: str = "trajectories") -> str:
    """
    将单架无人机的 setpoint 序列导出为独立 CSV 文件。

    列说明：
      timestamp_s  —— 从 0 开始的绝对时间戳（秒）
      x, y, z      —— 位置（米，本地笛卡尔系）
      yaw_rad      —— 偏航角（弧度，NED：正北=0，顺时针为正）
      type         —— setpoint 类型（见类说明）
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"uav_{uav_id}_trajectory.csv")

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "x", "y", "z", "yaw_rad", "type"])
        w.writerows(trajectory)

    print(f"  💾 UAV {uav_id} 轨迹已导出: {path}  ({len(trajectory)} 行)")
    return path


def export_trajectories_ply(all_trajectories: list,
                            output_path: str = "5_final_trajectories.ply"):
    """
    将全部无人机轨迹导出为带颜色的点云 PLY，供 CloudCompare / MeshLab 可视化。
    WAYPOINT 类型的点用更亮的颜色+更大密度标注，以便识别视点位置。
    """
    PALETTE = [
        [0.0, 1.0, 1.0],   # UAV 1: 荧光青
        [1.0, 0.2, 0.8],   # UAV 2: 亮洋红
        [1.0, 0.9, 0.0],   # UAV 3: 明黄
        [1.0, 0.5, 0.0],   # UAV 4: 亮橙
    ]
    WP_BOOST = [1.5, 1.5, 1.5]  # 视点标记放大系数（颜色更亮，此处仅示意）

    all_pts  = []
    all_cols = []

    for uid, traj in enumerate(all_trajectories):
        base_c = PALETTE[uid % len(PALETTE)]
        for row in traj:
            pt = [row[1], row[2], row[3]]
            tp = row[5]
            # 视点悬停帧使用白色高亮
            c  = [1.0, 1.0, 1.0] if "WAYPOINT" in tp else base_c
            all_pts.append(pt)
            all_cols.append(c)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(all_cols))
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"\n💾 全轨迹可视化点云已导出: {output_path}")