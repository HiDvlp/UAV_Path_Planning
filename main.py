#!/usr/bin/env python3
"""
UAV Path Planning Pipeline — main.py

支持分阶段断点续跑：每个阶段完成后自动保存检查点，可从任意阶段恢复。

用法示例：
  python main.py                              # 完整运行（阶段 1→5）
  python main.py --from-stage 3              # 从阶段 3 恢复（需要阶段 2 检查点）
  python main.py --to-stage 3               # 运行阶段 1-3 后保存退出
  python main.py --from-stage 2 --to-stage 4 # 仅运行阶段 2-4
  python main.py --list-checkpoints          # 查看已有检查点状态
"""

import argparse
import os
import pickle
import time
import numpy as np
import open3d as o3d

from algorithms.config import Config
from algorithms.module_1_preprocessing import load_and_preprocess_mesh
from algorithms.module_2_viewpoint import ViewpointGenerator
from algorithms.module_3_set_cover import QualityAwareSetCover
from algorithms.module_4_path_planning import MultiUAVPlanner
from algorithms.module_5_trajectory_optimization import (
    CollisionChecker,
    VoxelAStarPlanner,
    build_smooth_path,
)

CHECKPOINT_DIR = "output/checkpoints"
VIZ_DIR        = "output/visualizations"
STL_INPUT      = "data/airplane_aligned.stl"
STL_PROCESSED  = "output/visualizations/airplane_preprocessed.stl"

STAGE_NAMES = {
    1: "网格预处理 & 特征点提取",
    2: "候选视点生成 & 覆盖质量评估",
    3: "质量感知集合覆盖优化",
    4: "多机任务分配 & TSP 拓扑排序",
    5: "A* 避障 & Catmull-Rom 几何路径平滑",
}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt_path(stage: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"stage_{stage}.pkl")


def save_checkpoint(stage: int, data: dict) -> None:
    """将阶段输出序列化为 pickle 检查点文件。"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = _ckpt_path(stage)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [Checkpoint] 阶段 {stage} 已保存 -> {path}  ({size_mb:.1f} MB)")


def load_checkpoint(stage: int) -> dict:
    """加载指定阶段的检查点，找不到时给出可操作的错误提示。"""
    path = _ckpt_path(stage)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[错误] 找不到阶段 {stage} 的检查点: {path}\n"
            f"  请先运行:  python main.py --to-stage {stage}\n"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [Checkpoint] 阶段 {stage} 已加载 <- {path}  ({size_mb:.1f} MB)")
    return data


def list_checkpoints() -> None:
    """打印所有阶段的检查点状态。"""
    print("\n── 检查点状态 ─────────────────────────────────────────────────")
    for s in range(1, 6):
        name = STAGE_NAMES[s]
        if s == 5:
            exists = os.path.exists(_ckpt_path(4))
            status = "✅ 阶段4路线已就绪（可运行阶段5）" if exists else "❌ 需先运行阶段4"
            print(f"  阶段 5 [{name}]: {status}")
            continue
        path = _ckpt_path(s)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            mtime   = time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(os.path.getmtime(path)))
            print(f"  ✅ 阶段 {s} [{name}]")
            print(f"      {path}  {size_mb:.1f} MB  ({mtime})")
        else:
            print(f"  ❌ 阶段 {s} [{name}]: 无检查点")
    print("────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 共享工具
# ─────────────────────────────────────────────────────────────────────────────

def _load_processed_mesh():
    """从预处理 STL 加载网格并重建射线场景（供阶段 2 / 4 / 5 使用）。"""
    if not os.path.exists(STL_PROCESSED):
        raise FileNotFoundError(
            f"\n[错误] 找不到预处理网格: {STL_PROCESSED}\n"
            "  请先运行:  python main.py --to-stage 1\n"
        )
    print(f"  [Main] 正在加载预处理网格: {STL_PROCESSED}")
    mesh = o3d.io.read_triangle_mesh(STL_PROCESSED)
    mesh.compute_vertex_normals()
    mesh_t    = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(mesh_t)
    return mesh, ray_scene


def _print_stage_header(stage: int) -> float:
    print(f"\n{'='*64}")
    print(f"  阶段 {stage} / 5 : {STAGE_NAMES[stage]}")
    print(f"{'='*64}")
    return time.time()


# ─────────────────────────────────────────────────────────────────────────────
# 各阶段执行函数
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_1() -> dict:
    """
    阶段 1：网格预处理与自适应特征点提取。
    输入 : airplane_aligned.stl
    输出 : pts (N,3), norms (N,3)
    副产品: airplane_preprocessed.stl
    """
    os.makedirs(VIZ_DIR, exist_ok=True)
    mesh, pts, norms, _ = load_and_preprocess_mesh(STL_INPUT, STL_PROCESSED)
    return {"pts": pts, "norms": norms}


def run_stage_2(pts: np.ndarray, norms: np.ndarray) -> dict:
    """
    阶段 2：候选视点生成 + 数字相机覆盖质量评估。
    输入 : pts, norms (来自阶段 1 检查点或内存)
    输出 : valid_viewpoints (K,3), coverage_dict {vp_id: {tri_id: score}}
    """
    mesh, ray_scene = _load_processed_mesh()
    vg = ViewpointGenerator(ray_scene, mesh, pts, norms)
    valid_viewpoints, coverage_dict = vg.generate_candidates()

    if len(valid_viewpoints) == 0:
        raise RuntimeError(
            "阶段 2 未能生成任何合法候选视点。\n"
            "请检查 Config.SAFE_RADIUS / Config.UNDERBELLY_SAFE_RADIUS 是否过大。"
        )
    return {"valid_viewpoints": valid_viewpoints, "coverage_dict": coverage_dict}


def run_stage_3(valid_viewpoints: np.ndarray, coverage_dict: dict) -> dict:
    """
    阶段 3：质量感知 Lazy Greedy 集合覆盖优化。
    输入 : valid_viewpoints, coverage_dict (来自阶段 2)
    输出 : final_waypoints (M,3)
    副产品: 3_final_waypoints.ply
    """
    solver           = QualityAwareSetCover(coverage_dict, quality_threshold_ratio=0.85)
    selected_indices = solver.optimize()

    if not selected_indices:
        raise RuntimeError(
            "阶段 3 未找到覆盖解。\n"
            "可尝试降低 quality_threshold_ratio（如 0.7）后重试。"
        )

    final_waypoints = valid_viewpoints[selected_indices]
    print(f"\n  集合覆盖完成：精选出 {len(final_waypoints)} 个优质航点")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_waypoints)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.65, 0.0], (len(final_waypoints), 1))
    )
    os.makedirs(VIZ_DIR, exist_ok=True)
    ply_out = os.path.join(VIZ_DIR, "3_final_waypoints.ply")
    o3d.io.write_point_cloud(ply_out, pcd)
    print(f"  已保存: {ply_out}")

    return {"final_waypoints": final_waypoints}


def run_stage_4(final_waypoints: np.ndarray) -> dict:
    """
    阶段 4：多机任务分配（KMeans）+ 4D-TSP 拓扑排序。
    输入 : final_waypoints (来自阶段 3)
    输出 : all_routes  list[ (route_pts(N+2,3), route_yaws(N,)) ]
    副产品: 4_topo_lines.ply
    """
    mesh, _ = _load_processed_mesh()
    planner  = MultiUAVPlanner(final_waypoints, Config.TAKEOFF_POINTS)
    all_routes = planner.plan(mesh, output_dir=VIZ_DIR)
    print(f"  [阶段 4 完成] {len(all_routes)} 架无人机的初始航线已生成。")
    return {"all_routes": all_routes}


def run_stage_5(all_routes: list) -> list:
    """
    阶段 5：A* 避障 + Catmull-Rom 平滑，输出各无人机几何路径。
    输入 : all_routes (来自阶段 4)
    输出 : list of (smooth_path(M,3), route_yaws(N,))
    """
    csv_dir = "output/trajectories"
    if os.path.isdir(csv_dir):
        removed = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        for f in removed:
            os.remove(os.path.join(csv_dir, f))
        if removed:
            print(f"  [清理] 已删除 {csv_dir}/ 下 {len(removed)} 个旧 CSV 文件。")

    mesh, _ = _load_processed_mesh()
    checker = CollisionChecker(mesh)
    astar   = VoxelAStarPlanner(checker, mesh)
    results = []
    for uav_id, (route_pts, route_yaws) in enumerate(all_routes, start=1):
        print(f"  [UAV {uav_id}] 正在规划平滑路径...")
        smooth = build_smooth_path(astar, route_pts)
        results.append((smooth, route_yaws))
        print(f"  [UAV {uav_id}] 路径点数: {len(smooth)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 主流程调度器
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(from_stage: int = 1, to_stage: int = 5) -> None:
    total_start = time.time()
    print(f"\n{'='*64}")
    print(f"  UAV 路径规划流水线  |  阶段 {from_stage} → {to_stage}")
    print(f"{'='*64}")

    # ── 加载前置检查点（若跳过了早期阶段）──────────────────────────────
    stage_data: dict = {}
    if from_stage > 1:
        print(f"\n  正在加载阶段 {from_stage - 1} 的检查点作为起始数据...")
        stage_data = load_checkpoint(from_stage - 1)

    # ── 阶段 1 ────────────────────────────────────────────────────────────
    if from_stage <= 1 <= to_stage:
        t0 = _print_stage_header(1)
        result = run_stage_1()
        stage_data.update(result)
        save_checkpoint(1, {"pts": stage_data["pts"], "norms": stage_data["norms"]})
        print(f"  [阶段 1 完成] 耗时 {time.time() - t0:.1f}s")

    # ── 阶段 2 ────────────────────────────────────────────────────────────
    if from_stage <= 2 <= to_stage:
        t0 = _print_stage_header(2)
        result = run_stage_2(stage_data["pts"], stage_data["norms"])
        stage_data.update(result)
        save_checkpoint(2, {
            "valid_viewpoints": stage_data["valid_viewpoints"],
            "coverage_dict":    stage_data["coverage_dict"],
        })
        print(f"  [阶段 2 完成] 耗时 {time.time() - t0:.1f}s")

    # ── 阶段 3 ────────────────────────────────────────────────────────────
    if from_stage <= 3 <= to_stage:
        t0 = _print_stage_header(3)
        result = run_stage_3(
            stage_data["valid_viewpoints"],
            stage_data["coverage_dict"],
        )
        stage_data.update(result)
        save_checkpoint(3, {"final_waypoints": stage_data["final_waypoints"]})
        print(f"  [阶段 3 完成] 耗时 {time.time() - t0:.1f}s")

    # ── 阶段 4 ────────────────────────────────────────────────────────────
    if from_stage <= 4 <= to_stage:
        t0 = _print_stage_header(4)
        result = run_stage_4(stage_data["final_waypoints"])
        stage_data.update(result)
        save_checkpoint(4, {"all_routes": stage_data["all_routes"]})
        print(f"  [阶段 4 完成] 耗时 {time.time() - t0:.1f}s")

    # ── 阶段 5 ────────────────────────────────────────────────────────────
    if from_stage <= 5 <= to_stage:
        t0 = _print_stage_header(5)
        stage_data["smooth_paths"] = run_stage_5(stage_data["all_routes"])
        print(f"  [阶段 5 完成] 耗时 {time.time() - t0:.1f}s")

    # ── 汇总 ──────────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*64}")
    print(f"  流水线完成！总耗时: {elapsed:.1f}s")
    if to_stage < 5:
        next_s = to_stage + 1
        print(f"  检查点已保存，继续运行请执行：")
        print(f"    python main.py --from-stage {next_s}")
    else:
        print(f"  各无人机平滑路径已生成（共 {Config.NUM_UAVS} 架），可在 stage_data 中访问。")
        print(f"  可视化:  output/visualizations/4_topo_lines.ply")
    print(f"{'='*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="UAV 路径规划流水线（支持断点续跑）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
各阶段说明：
  阶段 1  网格预处理 & 特征点提取       → airplane_preprocessed.stl
  阶段 2  候选视点生成 & 覆盖质量评估   → checkpoints/stage_2.pkl
  阶段 3  质量感知集合覆盖优化          → 3_final_waypoints.ply
  阶段 4  多机任务分配 & TSP 拓扑排序   → 4_topo_lines.ply
  阶段 5  避障轨迹优化 & CSV 导出       → trajectories/uav_N_trajectory.csv

示例：
  python main.py                            # 完整运行（阶段 1-5）
  python main.py --to-stage 2               # 运行阶段 1-2 后保存退出
  python main.py --from-stage 3             # 从阶段 3 恢复（需要阶段 2 检查点）
  python main.py --from-stage 2 --to-stage 4   # 仅运行阶段 2-4
  python main.py --list-checkpoints         # 查看检查点状态
        """,
    )
    parser.add_argument(
        "--from-stage", type=int, default=1, metavar="N",
        help="从第 N 阶段开始（需要第 N-1 阶段的检查点，默认=1）",
    )
    parser.add_argument(
        "--to-stage", type=int, default=5, metavar="N",
        help="运行到第 N 阶段后停止并保存（默认=5）",
    )
    parser.add_argument(
        "--list-checkpoints", action="store_true",
        help="列出已有检查点的状态后退出",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_checkpoints:
        list_checkpoints()
    else:
        from_s = args.from_stage
        to_s   = args.to_stage
        if not (1 <= from_s <= to_s <= 5):
            print(f"[错误] 需满足 1 <= --from-stage ({from_s}) <= --to-stage ({to_s}) <= 5")
            raise SystemExit(1)
        run_pipeline(from_stage=from_s, to_stage=to_s)
