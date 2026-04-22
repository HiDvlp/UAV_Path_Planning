import os
import numpy as np
import open3d as o3d
import time
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
        
        # 1. 计算每个视点的偏航角约束 (假设相机看向坐标系中心或根据法向，此处简化为看向原点)
        # 实际应从 Module 3 传入法向，这里计算指向原点的 Yaw
        self.yaws = np.arctan2(-self.viewpoints[:, 1], -self.viewpoints[:, 0])

    def _cluster_tasks(self):
        print(f"[Module 4] 正在执行动力学感知聚类 (K={self.num_uavs})...")
        # 以起飞点为初始质心，使每架无人机优先负责离自己最近的区域
        kmeans = KMeans(n_clusters=self.num_uavs, init=self.takeoff_points,
                        n_init=1, random_state=42)
        labels = kmeans.fit_predict(self.viewpoints)
        return [np.where(labels == i)[0] for i in range(self.num_uavs)]

    def _solve_tsp(self, takeoff_pt, v_indices):
        """针对单机子集，构建 4D 代价矩阵并求解 TSP"""
        subset_pts = self.viewpoints[v_indices]
        subset_yaws = self.yaws[v_indices]
        
        # 组合：[Takeoff] + [Viewpoints]
        all_pts = np.vstack([takeoff_pt, subset_pts])
        all_yaws = np.concatenate([[0.0], subset_yaws]) # 假设起飞时朝向 0
        num_nodes = len(all_pts)

        # 向量化构建 4D 代价矩阵（广播替代双重 Python 循环）
        print(f" -> 构建 4D 代价矩阵 (节点数: {num_nodes})...")
        pi = all_pts[:, None, :]                                           # (M, 1, 3)
        pj = all_pts[None, :, :]                                           # (1, M, 3)
        d_eucl = np.linalg.norm(pi - pj, axis=-1)                         # (M, M)
        d_z    = np.abs(pj[..., 2] - pi[..., 2]) * Config.WEIGHT_CLIMB
        yaw_diff = all_yaws[None, :] - all_yaws[:, None]
        d_yaw  = np.abs((yaw_diff + np.pi) % (2 * np.pi) - np.pi) * Config.WEIGHT_TURN
        dist_int = ((d_eucl + d_z + d_yaw) * 1000).astype(int)            # OR-Tools 需要整数
        np.fill_diagonal(dist_int, 0)

        # OR-Tools 求解器配置
        manager = pywrapcp.RoutingIndexManager(num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return int(dist_int[manager.IndexToNode(from_index),
                                manager.IndexToNode(to_index)])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
        solution = routing.SolveWithParameters(search_parameters)
        
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:   # 排除起飞点占位符（node=0）
                route.append(v_indices[node - 1])
            index = solution.Value(routing.NextVar(index))
        return route

    def plan(self, mesh, output_dir="."):
        subsets = self._cluster_tasks()
        all_uav_routes = []

        # 阶段 1：直线拓扑排序
        print("\n[阶段 1] 正在进行拓扑排序与直线可视化...")
        line_points = []
        for i, v_indices in enumerate(subsets):
            print(f" -> 规划 UAV {i+1} 的路径...")
            route_indices = self._solve_tsp(self.takeoff_points[i], v_indices)

            # 完整航点序列：起飞点 -> 视点 -> 回到起飞点
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
                # 每 0.05 m 一点，最少 50 点；向量化广播替代逐点 append
                n  = max(int(np.linalg.norm(p1 - p0) / 0.05), 50)
                ts = np.linspace(0.0, 1.0, n)[:, None]    # (n, 1)
                all_pts_list.append(p0 + ts * (p1 - p0))  # (n, 3)
                all_cols_list.append(np.tile(c, (n, 1)))   # (n, 3)

        if not all_pts_list:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts_list))
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols_list))
        o3d.io.write_point_cloud(filename, pcd)
        print(f" 已导出: {filename}")