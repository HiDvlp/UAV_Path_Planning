import os
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from .config import Config

class MultiUAVPlanner:
    def __init__(self, viewpoints, takeoff_points):
        """
        :param viewpoints: (N, 3) 视点坐标
        :param takeoff_points: (K, 3) 基站起飞点坐标
        """
        self.viewpoints = np.asarray(viewpoints)
        self.takeoff_points = np.asarray(takeoff_points)
        self.num_uavs = len(takeoff_points)

        # 计算每个视点的偏航角（指向原点方向）
        self.yaws = np.arctan2(-self.viewpoints[:, 1], -self.viewpoints[:, 0])

    # ──────────────────────────────────────────────
    # 原有方法（KMeans + 独立TSP）
    # ──────────────────────────────────────────────

    def _cluster_tasks(self):
        print(f"[Module 4] 正在执行动力学感知聚类 (K={self.num_uavs})...")
        kmeans = KMeans(n_clusters=self.num_uavs, init=self.takeoff_points,
                        n_init=1, random_state=42)
        labels = kmeans.fit_predict(self.viewpoints)
        return [np.where(labels == i)[0] for i in range(self.num_uavs)]

    def _solve_tsp(self, takeoff_pt, v_indices):
        """针对给定起飞点和视点子集，构建 4D 代价矩阵并求解 TSP。
        返回 v_indices 中元素组成的有序列表（不含起飞点）。
        """
        subset_pts  = self.viewpoints[v_indices]
        subset_yaws = self.yaws[v_indices]

        all_pts  = np.vstack([takeoff_pt, subset_pts])
        all_yaws = np.concatenate([[0.0], subset_yaws])
        num_nodes = len(all_pts)

        print(f" -> 构建 4D 代价矩阵 (节点数: {num_nodes})...")
        pi = all_pts[:, None, :]
        pj = all_pts[None, :, :]
        d_eucl   = np.linalg.norm(pi - pj, axis=-1)
        d_z      = np.abs(pj[..., 2] - pi[..., 2]) * Config.WEIGHT_CLIMB
        yaw_diff = all_yaws[None, :] - all_yaws[:, None]
        d_yaw    = np.abs((yaw_diff + np.pi) % (2 * np.pi) - np.pi) * Config.WEIGHT_TURN
        dist_int = ((d_eucl + d_z + d_yaw) * 1000).astype(int)
        np.fill_diagonal(dist_int, 0)

        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return int(dist_int[manager.IndexToNode(from_index),
                                manager.IndexToNode(to_index)])

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_params.time_limit.seconds = 120

        solution = routing.SolveWithParameters(search_params)

        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(v_indices[node - 1])
            index = solution.Value(routing.NextVar(index))
        return route

    def _plan_kmeans(self, mesh, output_dir="."):
        """原有 KMeans + 独立TSP 路径，保留作为对照回退。"""
        subsets = self._cluster_tasks()
        line_points = []
        all_uav_routes = []

        print("\n[KMeans] 正在进行拓扑排序与直线可视化...")
        for i, v_indices in enumerate(subsets):
            print(f" -> 规划 UAV {i+1} 的路径...")
            route_indices = self._solve_tsp(self.takeoff_points[i], v_indices)

            full_route_pts = np.vstack([
                self.takeoff_points[i],
                self.viewpoints[route_indices],
                self.takeoff_points[i]
            ])
            all_uav_routes.append((full_route_pts, self.yaws[route_indices]))
            line_points.append(full_route_pts)

        ply_path = os.path.join(output_dir, "4_topo_lines.ply")
        self._visualize_and_export(mesh, line_points, ply_path)
        return all_uav_routes

    # ──────────────────────────────────────────────
    # MUCS-BSAE 新方法
    # ──────────────────────────────────────────────

    def _global_tsp_order(self):
        """对所有 M 个视点求解一次全局 TSP，返回视点索引的有序列表。"""
        depot_pt    = np.mean(self.takeoff_points, axis=0)
        all_indices = np.arange(len(self.viewpoints))
        print(f"[BSAE] 全局TSP排序 ({len(all_indices)} 视点)，预计 80–200s...")
        try:
            return self._solve_tsp(depot_pt, all_indices)
        except Exception as e:
            print(f"[BSAE] WARNING: 全局TSP异常({e})，使用自然顺序回退")
            return list(all_indices)

    def _bsae_segment(self, global_order, uav_idx, start_wp_idx, t_budget):
        """从 global_order[start_wp_idx:] 开始，贪心为 UAV uav_idx 分配视点。

        同时受时间预算 t_budget 和电池能量约束。
        返回 (assigned_indices, next_start_idx)。
        """
        takeoff_pt = self.takeoff_points[uav_idx]
        SEL   = Config.ENERGY_PER_METER_FLIGHT
        STEL  = Config.ENERGY_PER_SECOND_HOVER
        BATT  = Config.UAV_BATTERY_CAPACITY
        SPEED = Config.CRUISE_SPEED
        HOVER = Config.HOVER_TIME

        assigned    = []
        t_elapsed   = 0.0
        E_accum     = 0.0
        current_pos = takeoff_pt.copy()

        for i in range(start_wp_idx, len(global_order)):
            wp_pos   = self.viewpoints[global_order[i]]
            d_flight = np.linalg.norm(wp_pos - current_pos)
            d_return = np.linalg.norm(wp_pos - takeoff_pt)

            t_flight = d_flight / SPEED
            t_return = d_return / SPEED

            # 前瞻检查：加入该视点后能否还能飞回基地？
            if (t_elapsed + t_flight + HOVER + t_return > t_budget or
                    E_accum + (t_flight + t_return) * SEL + HOVER * STEL > BATT):
                return assigned, i

            assigned.append(global_order[i])
            t_elapsed += t_flight + HOVER
            E_accum   += t_flight * SEL + HOVER * STEL
            current_pos = wp_pos

        return assigned, len(global_order)

    def _bsae_allocate(self, global_order):
        """二分搜索最小可行时间预算，将全局有序视点分配给各 UAV。

        返回 list[list[int]]，长度 == num_uavs。
        """
        M, K = len(global_order), self.num_uavs

        # T_max：沿全局TSP路径单机完成全部视点所需时间
        pts   = np.vstack([self.takeoff_points[0],
                           self.viewpoints[global_order],
                           self.takeoff_points[0]])
        T_max = int(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
                    / Config.CRUISE_SPEED) + M * int(Config.HOVER_TIME) + 1

        def try_budget(t):
            segs, nxt = [], 0
            for ui in range(K):
                seg, nxt = self._bsae_segment(global_order, ui, nxt, float(t))
                segs.append(seg)
                if nxt >= M:
                    segs += [[] for _ in range(ui + 1, K)]
                    return True, segs
            return sum(len(s) for s in segs) == M, segs

        ok, best = try_budget(T_max)
        if not ok:
            print("[BSAE] WARNING: 电池约束过紧，均分回退（不含能量约束）")
            chunk = M // K
            return [global_order[i * chunk:(i + 1) * chunk if i < K - 1 else M]
                    for i in range(K)]

        t_lo, t_hi = 0, T_max
        while t_lo < t_hi - 1:
            t_mid = (t_lo + t_hi) // 2
            ok, segs = try_budget(t_mid)
            if ok:
                t_hi, best = t_mid, segs
            else:
                t_lo = t_mid

        print(f"[BSAE] 最小可行时间预算: {t_hi}s")
        return best

    def _imucs_relay_optimize(self, segment_indices, uav_takeoff):
        """IMUCS-BSAE 接力点优化：将离当前UAV基地最近的未访问视点置于段首，
        再对剩余视点重新求解 TSP，消除无效迂回。
        """
        if len(segment_indices) <= 1:
            return list(segment_indices)

        seg_arr  = np.array(segment_indices)
        seg_pts  = self.viewpoints[seg_arr]
        dists    = np.linalg.norm(seg_pts - uav_takeoff[None, :], axis=1)
        relay_local = int(np.argmin(dists))
        relay_vp    = segment_indices[relay_local]

        remaining = [idx for idx in segment_indices if idx != relay_vp]
        if not remaining:
            return [relay_vp]

        relay_pos     = self.viewpoints[relay_vp]
        ordered_rest  = self._solve_tsp(relay_pos, np.array(remaining, dtype=int))
        return [relay_vp] + ordered_rest

    def _plan_bsae(self, mesh, output_dir="."):
        """MUCS-BSAE + IMUCS-BSAE 主流程。"""
        print("\n[Module 4 - MUCS-BSAE] 全局TSP + 二分搜索负载均衡 + 接力点优化")

        # 阶段1：全局TSP排序
        global_order = self._global_tsp_order()

        # 阶段2：二分搜索分配
        raw_segments = self._bsae_allocate(global_order)

        # 阶段3：各段接力点优化 + 路线组装
        line_points    = []
        all_uav_routes = []

        for uav_idx, segment in enumerate(raw_segments):
            takeoff_pt = self.takeoff_points[uav_idx]

            reordered = (self._imucs_relay_optimize(segment, takeoff_pt)
                         if segment else [])

            if not reordered:
                route_pts  = np.vstack([takeoff_pt, takeoff_pt])   # (2, 3)
                route_yaws = np.array([], dtype=float)              # (0,)
            else:
                route_pts  = np.vstack([takeoff_pt,
                                        self.viewpoints[np.array(reordered)],
                                        takeoff_pt])                # (N_i+2, 3)
                route_yaws = self.yaws[np.array(reordered)]        # (N_i,)

            all_uav_routes.append((route_pts, route_yaws))
            line_points.append(route_pts)

            dist_total = np.sum(np.linalg.norm(np.diff(route_pts, axis=0), axis=1))
            print(f"  UAV {uav_idx+1}: {len(reordered)} 视点, 路径={dist_total:.1f}m")

        ply_path = os.path.join(output_dir, "4_topo_lines.ply")
        self._visualize_and_export(mesh, line_points, ply_path)
        return all_uav_routes

    # ──────────────────────────────────────────────
    # 入口
    # ──────────────────────────────────────────────

    def plan(self, mesh, output_dir="."):
        if Config.USE_BSAE_ALLOCATION:
            return self._plan_bsae(mesh, output_dir)
        return self._plan_kmeans(mesh, output_dir)

    # ──────────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────────

    def _visualize_and_export(self, mesh, paths, filename):
        """将多架无人机的拓扑路径转化为高密度彩色点云并导出为 PLY。"""
        COLORS = [
            [0.0, 1.0, 1.0],   # UAV 1: 荧光青
            [1.0, 0.2, 0.8],   # UAV 2: 亮洋红
            [1.0, 0.9, 0.0],   # UAV 3: 明黄
            [1.0, 0.5, 0.0],   # UAV 4: 亮橙
        ]
        all_pts_list  = []
        all_cols_list = []

        for i, path in enumerate(paths):
            pts_np = np.asarray(path)
            if len(pts_np) < 2:
                continue
            c = np.array(COLORS[i % len(COLORS)], dtype=np.float64)

            for j in range(len(pts_np) - 1):
                p0, p1 = pts_np[j], pts_np[j + 1]
                n  = max(int(np.linalg.norm(p1 - p0) / 0.05), 50)
                ts = np.linspace(0.0, 1.0, n)[:, None]
                all_pts_list.append(p0 + ts * (p1 - p0))
                all_cols_list.append(np.tile(c, (n, 1)))

        if not all_pts_list:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts_list))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols_list))
        o3d.io.write_point_cloud(filename, pcd)
        print(f" 已导出: {filename}")
