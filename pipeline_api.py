"""
pipeline_api.py — UAV 路径规划流水线 统一 API

在 Jupyter Notebook 中执行 `from pipeline_api import *` 后即可使用全部函数。

阶段1 说明：从 .pcd 点云出发，经 SHS-Net 有方向法向估计 + Screened Poisson 重建
网格，再基于 PCA 曲率做自适应降采样，产出 airplane_reconstructed.ply 供下游使用。

函数一览：
  ── 运行控制 ──────────────────────────────────────────
  run("1-5")              完整运行或指定阶段范围
  stage1~5(force=False)   单独运行某阶段

  ── 状态与结果查看 ────────────────────────────────────
  status()                查看所有检查点和输出文件状态
  summary()               查看各阶段关键结果数字

  ── 参数调整 ──────────────────────────────────────────
  show_config()           显示全部配置参数
  set_config(KEY=val)     运行时修改配置（立即生效）
  reset_config()          恢复 config.py 中的默认值

  ── 检查点管理 ────────────────────────────────────────
  reset(from_stage=1)     清除指定阶段起的检查点
  save_snapshot(name)     将当前检查点保存为命名快照
  load_snapshot(name)     从命名快照恢复检查点
  list_snapshots()        列出所有已保存快照
"""

import os
import sys
import pickle
import time
import shutil

import numpy as np
import open3d as o3d

# 确保工作目录为脚本所在的项目根目录（import 时自动执行）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from algorithms.config import Config

# 启动时自动加载场景配置（若 scenes/default_scene.json 存在）
Config.load_scene()

from algorithms.module_1_preprocessing import load_and_preprocess_pcd
from algorithms.module_2_viewpoint import ViewpointGenerator
from algorithms.module_3_set_cover import QualityAwareSetCover
from algorithms.module_4_path_planning import MultiUAVPlanner
from algorithms.module_5_trajectory_optimization import (
    CollisionChecker,
    VoxelAStarPlanner,
    build_smooth_path,
)

# ── 路径常量 ──────────────────────────────────────────────────────────────────
_CKPT_DIR     = "output/checkpoints"
_SNAP_DIR     = "output/snapshots"
_VIZ_DIR      = "output/visualizations"
_PCD_INPUT    = "data/airplane_aligned.pcd"
_PLY_PROC     = "output/visualizations/airplane_reconstructed.ply"
_STAGE_NAMES = {
    1: "网格预处理 & 特征点提取",
    2: "候选视点生成 & 覆盖质量评估",
    3: "质量感知集合覆盖优化",
    4: "多机任务分配 & TSP 拓扑排序",
    5: "A* 避障 & Catmull-Rom 几何路径平滑",
}

# 运行时内存缓存（同一 session 内避免重复从磁盘加载大文件）
_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# 内部工具函数（下划线前缀，不对外暴露）
# ══════════════════════════════════════════════════════════════════════════════

def _ckpt_path(stage: int) -> str:
    return os.path.join(_CKPT_DIR, f"stage_{stage}.pkl")


def _exists(stage: int) -> bool:
    return os.path.exists(_ckpt_path(stage))


def _save(stage: int, data: dict) -> None:
    os.makedirs(_CKPT_DIR, exist_ok=True)
    with open(_ckpt_path(stage), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(_ckpt_path(stage)) / 1e6
    _cache[stage] = data
    print(f"  [✓] 检查点已保存: {_ckpt_path(stage)}  ({mb:.1f} MB)")


def _load(stage: int) -> dict:
    path = _ckpt_path(stage)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"检查点不存在: {path}\n请先运行: stage{stage}()"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    mb = os.path.getsize(path) / 1e6
    _cache[stage] = data
    print(f"  [✓] 检查点已加载: {path}  ({mb:.1f} MB)")
    return data


def _get(stage: int) -> dict:
    """从内存缓存或磁盘检查点获取指定阶段数据。"""
    if stage in _cache:
        return _cache[stage]
    if _exists(stage):
        return _load(stage)
    raise RuntimeError(
        f"阶段{stage}数据不可用，请先运行: stage{stage}()"
    )


def _load_mesh():
    """从重建 PLY 加载 Open3D 网格和射线场景。"""
    if not os.path.exists(_PLY_PROC):
        raise FileNotFoundError(
            f"找不到重建网格 {_PLY_PROC}，请先运行 stage1()"
        )
    mesh = o3d.io.read_triangle_mesh(_PLY_PROC)
    mesh.compute_vertex_normals()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return mesh, scene


def _parse_stages(stages) -> tuple[int, int]:
    """将 '1-5'、'3'、3 等格式解析为 (start, end) 元组。"""
    s = str(stages).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    n = int(s)
    return n, n


def _sec(title: str) -> None:
    print(f"\n{'─'*54}\n  {title}\n{'─'*54}")


# ══════════════════════════════════════════════════════════════════════════════
# 各阶段执行函数
# ══════════════════════════════════════════════════════════════════════════════

def stage1(force: bool = False,
           poisson_depth: int = 10,
           shs_device: str = "auto",
           shs_ckpt: str = None) -> dict:
    """阶段1：PCD 点云预处理 — SHS-Net 法向估计 + Poisson 重建 + 自适应特征点提取。

    Args:
        force:         True 则忽略已有检查点，强制重新计算。
        poisson_depth: Screened Poisson 重建的八叉树深度（默认 10）。
        shs_device:    SHS-Net 推理设备，"auto" 自动选择 GPU/CPU（默认 "auto"）。
        shs_ckpt:      SHS-Net 预训练权重路径，None 则使用模块默认路径。
    Returns:
        {"pts": (N,3), "norms": (N,3)}
    """
    _sec(f"阶段1 / 5 : {_STAGE_NAMES[1]}")
    if not force and 1 in _cache:
        print("  [缓存命中] 直接使用内存数据。")
        return _cache[1]
    if not force and _exists(1):
        return _load(1)
    t0 = time.time()
    os.makedirs(_VIZ_DIR, exist_ok=True)
    kwargs = dict(poisson_depth=poisson_depth, shs_device=shs_device)
    if shs_ckpt is not None:
        kwargs["shs_ckpt"] = shs_ckpt
    _, pts, norms, _ = load_and_preprocess_pcd(_PCD_INPUT, _PLY_PROC, **kwargs)
    _save(1, {"pts": pts, "norms": norms})
    print(f"  耗时: {time.time()-t0:.1f}s")
    return _cache[1]


def stage2(force: bool = False) -> dict:
    """阶段2：候选视点生成 & 数字相机覆盖质量评估。

    Args:
        force: True 则强制重新计算。
    Returns:
        {"valid_viewpoints": (K,3), "coverage_dict": {vp_id: {tri_id: score}}}
    """
    _sec(f"阶段2 / 5 : {_STAGE_NAMES[2]}")
    if not force and 2 in _cache:
        print("  [缓存命中] 直接使用内存数据。")
        return _cache[2]
    if not force and _exists(2):
        return _load(2)
    d1 = _get(1)
    mesh, ray_scene = _load_mesh()
    t0 = time.time()
    vg = ViewpointGenerator(ray_scene, mesh, d1["pts"], d1["norms"])
    vps, cov = vg.generate_candidates()
    if len(vps) == 0:
        raise RuntimeError("未生成任何候选视点，请检查 Config.SAFE_RADIUS。")
    _save(2, {"valid_viewpoints": vps, "coverage_dict": cov})
    print(f"  候选视点: {len(vps)}  耗时: {time.time()-t0:.1f}s")
    return _cache[2]


def stage3(force: bool = False, quality_threshold: float = 0.85) -> dict:
    """阶段3：质量感知 Lazy Greedy 集合覆盖优化。

    Args:
        force:             True 则强制重新计算。
        quality_threshold: 覆盖达标比例 0~1，越大要求越严、航点越多（默认 0.85）。
    Returns:
        {"final_waypoints": (M,3)}
    """
    _sec(f"阶段3 / 5 : {_STAGE_NAMES[3]}")
    if not force and 3 in _cache:
        print("  [缓存命中] 直接使用内存数据。")
        return _cache[3]
    if not force and _exists(3):
        return _load(3)
    d2     = _get(2)
    t0     = time.time()
    solver = QualityAwareSetCover(d2["coverage_dict"],
                                  quality_threshold_ratio=quality_threshold)
    sel    = solver.optimize()
    if not sel:
        raise RuntimeError("未找到覆盖解，请降低 quality_threshold。")
    fwp = d2["valid_viewpoints"][sel]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fwp)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.65, 0.0], (len(fwp), 1))
    )
    os.makedirs(_VIZ_DIR, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(_VIZ_DIR, "3_final_waypoints.ply"), pcd)
    _save(3, {"final_waypoints": fwp})
    print(f"  精选航点: {len(fwp)}  耗时: {time.time()-t0:.1f}s")
    return _cache[3]


def stage4(force: bool = False) -> dict:
    """阶段4：KMeans 多机任务分配 + 4D-TSP 拓扑排序。

    Args:
        force: True 则强制重新计算。
    Returns:
        {"all_routes": [(route_pts(N+2,3), route_yaws(N,)), ...]}
    """
    _sec(f"阶段4 / 5 : {_STAGE_NAMES[4]}")
    if not force and 4 in _cache:
        print("  [缓存命中] 直接使用内存数据。")
        return _cache[4]
    if not force and _exists(4):
        return _load(4)
    d3   = _get(3)
    mesh, _ = _load_mesh()
    t0   = time.time()
    planner = MultiUAVPlanner(d3["final_waypoints"], Config.TAKEOFF_POINTS)
    routes  = planner.plan(mesh, output_dir=_VIZ_DIR)
    _save(4, {"all_routes": routes})
    print(f"  {len(routes)} 架无人机路线已生成  耗时: {time.time()-t0:.1f}s")
    return _cache[4]


def stage5(force: bool = False) -> list:
    """阶段5：A* 避障 + Catmull-Rom 平滑，输出各无人机几何路径。

    Args:
        force: 传入此参数不影响运行（阶段5无检查点，每次均重新生成）。
    Returns:
        list of (smooth_path(M,3), route_yaws(N,)) — 每架无人机一个元组
    """
    _sec(f"阶段5 / 5 : {_STAGE_NAMES[5]}")
    csv_dir = "output/trajectories"
    if os.path.isdir(csv_dir):
        removed = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        for f in removed:
            os.remove(os.path.join(csv_dir, f))
        if removed:
            print(f"  [清理] 已删除 {csv_dir}/ 下 {len(removed)} 个旧 CSV 文件。")
    d4      = _get(4)
    mesh, _ = _load_mesh()
    t0      = time.time()
    checker = CollisionChecker(mesh)
    astar   = VoxelAStarPlanner(checker, mesh)
    results = []
    for uid, (rpts, ryaws) in enumerate(d4["all_routes"], start=1):
        print(f"\n  [UAV {uid}] 正在规划平滑路径...")
        smooth = build_smooth_path(astar, rpts)
        results.append((smooth, ryaws))
        print(f"  [UAV {uid}] 完成，路径点数: {len(smooth)}")
    print(f"\n  耗时: {time.time()-t0:.1f}s")

    # 导出可视化 PLY
    _TRAJ_COLORS = [
        [0.0, 1.0, 1.0],
        [1.0, 0.2, 0.8],
        [1.0, 0.9, 0.0],
        [1.0, 0.5, 0.0],
    ]
    all_pts, all_cols = [], []
    for uid, (smooth, _) in enumerate(results):
        c = np.array(_TRAJ_COLORS[uid % len(_TRAJ_COLORS)])
        all_pts.append(smooth)
        all_cols.append(np.tile(c, (len(smooth), 1)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    ply_out = os.path.join(_VIZ_DIR, "5_final_trajectories.ply")
    o3d.io.write_point_cloud(ply_out, pcd)
    print(f"  💾 已导出: {ply_out}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 组合运行
# ══════════════════════════════════════════════════════════════════════════════

def run(stages="1-5", force: bool = False) -> None:
    """运行指定范围的流水线阶段。

    Args:
        stages: 阶段范围，支持 '1-5'、'3-5'、'2'、3 等写法。
        force:  True 则强制重新计算，忽略已有检查点。

    Examples:
        run()                # 完整运行阶段 1-5
        run("3-5")           # 从阶段3开始运行
        run(3)               # 只运行阶段3
        run("2", force=True) # 强制重跑阶段2
    """
    a, b = _parse_stages(stages)
    _fns = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5}
    t0   = time.time()
    print(f"\n{'='*54}")
    print(f"  UAV 路径规划流水线  |  阶段 {a} → {b}")
    print(f"{'='*54}")
    for s in range(a, b + 1):
        _fns[s](force=force)
    print(f"\n{'='*54}")
    print(f"  ✅ 完成！总耗时: {time.time()-t0:.1f}s")
    print(f"{'='*54}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 状态与结果查看
# ══════════════════════════════════════════════════════════════════════════════

def status() -> None:
    """显示所有检查点、输出文件、内存缓存的当前状态。"""
    print(f"\n{'─'*60}")
    print("  检查点 & 输出状态")
    print(f"{'─'*60}")
    for s in range(1, 6):
        path  = _ckpt_path(s)
        cache = "  [已缓存]" if s in _cache else ""
        if s == 5:
            exists = os.path.exists(_ckpt_path(4))
            tag    = "✅ 阶段4路线已就绪（可运行阶段5）" if exists else "⬜  需先运行阶段4"
            print(f"  {tag}  阶段5  {_STAGE_NAMES[5]}")
            continue
        if os.path.exists(path):
            mtime = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path)))
            mb    = os.path.getsize(path) / 1e6
            print(f"  ✅ 阶段{s}  {_STAGE_NAMES[s]}")
            print(f"      {path}  {mb:.1f} MB  {mtime}{cache}")
        else:
            print(f"  ⬜ 阶段{s}  {_STAGE_NAMES[s]}  ── 未生成")
    if os.path.isdir(_VIZ_DIR):
        plys = sorted(f for f in os.listdir(_VIZ_DIR) if f.endswith(".ply"))
        if plys:
            print(f"\n  PLY 文件 ({_VIZ_DIR}/): {', '.join(plys)}")
    print(f"{'─'*60}\n")


def summary() -> None:
    """显示各阶段的关键结果数字（自动从缓存或检查点读取）。"""
    print(f"\n{'─'*52}\n  流水线结果摘要\n{'─'*52}")
    for s in range(1, 6):
        if s == 5:
            exists = _exists(4)
            print(f"  阶段5  {'阶段4路线已就绪，可执行 stage5()' if exists else '需先运行阶段4'}")
            continue
        if not (_exists(s) or s in _cache):
            print(f"  阶段{s}: 尚未运行")
            continue
        d = _get(s)
        if s == 1:
            pts = d["pts"]
            print(f"  阶段1  特征点: {len(pts):,}  "
                  f"Z=[{pts[:,2].min():.1f}, {pts[:,2].max():.1f}] m")
        elif s == 2:
            vps, cov = d["valid_viewpoints"], d["coverage_dict"]
            all_t = set()
            for v in cov.values(): all_t.update(v.keys())
            print(f"  阶段2  候选视点: {len(vps):,}  覆盖面片: {len(all_t):,}")
        elif s == 3:
            fwp = d["final_waypoints"]
            d2  = _cache.get(2) or (_load(2) if _exists(2) else None)
            ratio = (f"  (压缩至 {len(fwp)/len(d2['valid_viewpoints'])*100:.1f}%)"
                     if d2 else "")
            print(f"  阶段3  精选航点: {len(fwp):,}{ratio}")
        elif s == 4:
            for i, (rpts, _) in enumerate(d["all_routes"]):
                n    = len(rpts) - 2
                dist = sum(np.linalg.norm(rpts[j+1]-rpts[j])
                           for j in range(len(rpts)-1))
                print(f"  阶段4  UAV {i+1}: {n} 个视点  路径 {dist:.1f} m")
    print(f"{'─'*52}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 参数配置
# ══════════════════════════════════════════════════════════════════════════════

def show_config() -> None:
    """分类显示 config.py 中全部配置参数的当前值。"""
    params = {k: v for k, v in vars(Config).items()
              if not k.startswith("_") and not callable(v)}
    sections = [
        ("成像参数（影响阶段2）",
         ["CAMERA_DISTANCE", "UNDERBELLY_CAMERA_DIST", "FOV_DEG",
          "ASPECT_RATIO", "MAX_INCIDENCE_ANGLE",
          "PROBE_D_OFFSETS", "PROBE_THETAS"]),
        ("安全参数（阶段2/5）",
         ["SAFE_RADIUS", "UNDERBELLY_SAFE_RADIUS",
          "MIN_SAFE_Z", "FLIGHT_SAFE_RADIUS"]),
        ("飞行速度（阶段5）",
         ["CRUISE_SPEED", "CREEP_SPEED", "TRANSIT_SPEED"]),
        ("多机参数（阶段4）",
         ["NUM_UAVS", "TAKEOFF_POINTS", "WEIGHT_CLIMB", "WEIGHT_TURN",
          "UAV_BATTERY_CAPACITY", "ENERGY_PER_METER_FLIGHT",
          "ENERGY_PER_SECOND_HOVER", "USE_BSAE_ALLOCATION"]),
        ("A* 参数（阶段5）",
         ["VOXEL_SIZE"]),
        ("轨迹参数（阶段5）",
         ["HOVER_TIME", "CSV_FREQ"]),
        ("地理参考",
         ["ANCHOR_LAT", "ANCHOR_LON", "ANCHOR_ALT"]),
    ]
    print()
    for sec, keys in sections:
        print(f"  ── {sec} {'─'*(38-len(sec))}")
        for k in keys:
            if k in params:
                print(f"    {k:<30} = {params[k]}")
    print()


def set_config(**kwargs) -> None:
    """运行时修改配置参数，立即生效。

    修改影响阶段2的参数后，请执行 reset(from_stage=2) 再重跑。
    修改仅影响阶段5的参数后，直接重跑 stage5() 即可。

    Examples:
        set_config(FOV_DEG=80, SAFE_RADIUS=2.5)
        set_config(HOVER_TIME=8.0, TRANSIT_SPEED=1.5)
        set_config(NUM_UAVS=2, TAKEOFF_POINTS=[[20,0,0.5],[-20,0,0.5]])
    """
    changed = []
    for k, v in kwargs.items():
        if not hasattr(Config, k):
            print(f"  ⚠️  Config 中不存在参数 '{k}'，已跳过")
            continue
        old = getattr(Config, k)
        setattr(Config, k, v)
        changed.append((k, old, v))
    if changed:
        print("  参数已修改:")
        for k, old, new in changed:
            print(f"    {k:<30} {str(old):<20} →  {new}")


def reset_config() -> None:
    """重新加载 config.py，将全部参数恢复为文件中的默认值。"""
    import importlib
    import algorithms.config as _cfg
    importlib.reload(_cfg)
    # 让 Config 指向重新加载后的类
    from algorithms.config import Config as _NewConfig
    for k, v in vars(_NewConfig).items():
        if not k.startswith("_") and not callable(v):
            setattr(Config, k, v)
    print("  所有参数已恢复为 config.py 默认值。")


# ══════════════════════════════════════════════════════════════════════════════
# 检查点管理
# ══════════════════════════════════════════════════════════════════════════════

def reset(from_stage: int = 1) -> None:
    """清除指定阶段及后续阶段的检查点和内存缓存。

    Args:
        from_stage: 从哪个阶段开始清除（默认 1，即清除全部）。

    Examples:
        reset()      # 清除全部检查点
        reset(3)     # 仅清除阶段3、4（阶段1、2保留）
    """
    cleared = []
    for s in range(from_stage, 6):
        path = _ckpt_path(s)
        if os.path.exists(path):
            os.remove(path)
            cleared.append(s)
        _cache.pop(s, None)
    msg = f"已清除阶段 {cleared} 的检查点。" if cleared else "无需清除（对应阶段无检查点）。"
    print(f"  {msg}")


def save_snapshot(name: str) -> None:
    """将当前全部检查点打包为命名快照，便于对比不同参数组合的实验结果。

    Example:
        save_snapshot("fov90_radius3")
    """
    if not os.path.isdir(_CKPT_DIR):
        print("  无可保存的检查点。")
        return
    dst = os.path.join(_SNAP_DIR, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(_CKPT_DIR, dst)
    pkls = [f for f in os.listdir(dst) if f.endswith(".pkl")]
    print(f"  快照已保存: {dst}  ({len(pkls)} 个检查点)")


def load_snapshot(name: str) -> None:
    """从命名快照恢复检查点，并清除内存缓存（强制从磁盘重新读取）。

    Example:
        load_snapshot("fov90_radius3")
    """
    src = os.path.join(_SNAP_DIR, name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"快照不存在: {src}")
    if os.path.isdir(_CKPT_DIR):
        shutil.rmtree(_CKPT_DIR)
    shutil.copytree(src, _CKPT_DIR)
    _cache.clear()
    print(f"  快照已恢复: {name}")


def list_snapshots() -> None:
    """列出所有已保存的快照及其创建时间。"""
    if not os.path.isdir(_SNAP_DIR) or not os.listdir(_SNAP_DIR):
        print("  暂无快照。使用 save_snapshot(name) 创建。")
        return
    print(f"\n  {'─'*46}\n  已保存快照\n  {'─'*46}")
    for name in sorted(os.listdir(_SNAP_DIR)):
        path = os.path.join(_SNAP_DIR, name)
        if not os.path.isdir(path):
            continue
        mtime = time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(os.path.getmtime(path)))
        pkls  = [f for f in os.listdir(path) if f.endswith(".pkl")]
        print(f"  {name:<30} {mtime}  ({len(pkls)} 个检查点)")
    print()


# ── 加载完成提示 ──────────────────────────────────────────────────────────────
print("✅ pipeline_api 加载完成。")
print("   常用指令: run() | status() | summary() | show_config() | help(<函数名>)")
print("   换场景请编辑 scenes/default_scene.json，然后调用 Config.load_scene()")
