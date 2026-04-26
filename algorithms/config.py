# config.py
class Config:
    # ==========================================
    # 📷 1. 传感器成像参数 (Sensor / FOV Constraints)
    # ==========================================
    CAMERA_DISTANCE = 5.0           # 常规机背/侧面成像距离
    UNDERBELLY_CAMERA_DIST = 3.0    # 🎯 机腹/翼下仰视专属成像距离
    FOV_DEG = 90.0             # 水平视场角
    ASPECT_RATIO = 1.5           # 相机画幅长宽比 (大疆P1/M3T等常用3:2)
    MAX_INCIDENCE_ANGLE = 45.0   # 最大允许拍摄入射角 (防止擦边反光/严重畸变)
    PROBE_D_OFFSETS = [0.0, 0.5, 1.0, 1.5, 2.0]  # 距离试探步长
    PROBE_THETAS = [0.0, 15.0, 30.0, 45.0]        # 角度试探步长
    DIST_SCORE_SIGMA2 = 8.0   # 距离质量分高斯衰减的方差参数 σ²；调大→对距离容忍度更高

    # ==========================================
    # 🛡️ 2. 飞行安全与避障参数 (Kinematic / Safety Constraints)
    # ==========================================
    SAFE_RADIUS = 3.0               # Module 2 视点生成用：常规区域安全半径
    UNDERBELLY_SAFE_RADIUS = 1.0    # Module 2 视点生成用：机腹/翼下专属避障半径
    MIN_SAFE_Z = 0.5                # 全局最低飞行高度（距地面，单位：米）

    # 🆕 Module 5 专用：全程飞行轨迹安全距离
    # 区别于 SAFE_RADIUS（视点生成时用于寻找合法视点位置），
    # FLIGHT_SAFE_RADIUS 用于轨迹段的碰撞检测，控制转场路径与蒙皮的最小间距。
    # 建议值：略小于 SAFE_RADIUS，以保留足够的可飞行通道。
    FLIGHT_SAFE_RADIUS = 2.0        # 轨迹全程与蒙皮的最小安全距离（米）

    # 🎯 飞行速度控制 (m/s)
    CRUISE_SPEED = 1.0              # 正常巡航速度（Module 4 使用）
    CREEP_SPEED = 0.5               # 机腹防撞蠕行速度（Module 4 使用）
    TRANSIT_SPEED = 2.0             # 🆕 Module 5 转场飞行速度（m/s）

    # ==========================================
    # ⚙️ 3. DJI M350 RTK 双云台物理限位参数
    # ==========================================
    BOTTOM_CAM_PITCH_MIN = -120.0
    BOTTOM_CAM_PITCH_MAX = 30.0
    TOP_CAM_PITCH_MIN = -30.0
    TOP_CAM_PITCH_MAX = 120.0

    # ==========================================
    # 🚁 4. Module 4 协同与路径规划参数
    # ==========================================
    NUM_UAVS = 4
    TAKEOFF_POINTS = [
        [ 20.0,  20.0, 0.5],
        [ 20.0, -20.0, 0.5],
        [-20.0,  20.0, 0.5],
        [-20.0, -20.0, 0.5]
    ]
    WEIGHT_CLIMB = 3.0   # TSP 弧权重：相邻视点间高度变化代价（对称，上升/下降均计入）
    WEIGHT_TURN = 2.0

    # ── MUCS-BSAE 能耗与分配参数 ─────────────────────────────────
    UAV_BATTERY_CAPACITY    = 1500.0   # 单机电池容量（归一化能量单位）
    ENERGY_PER_METER_FLIGHT =    1.0   # SEL：每米飞行耗能
    ENERGY_PER_SECOND_HOVER =    0.5   # STEL：每秒悬停耗能
    USE_BSAE_ALLOCATION     = True     # True=MUCS-BSAE；False=原KMeans+TSP

    # ==========================================
    # 🗺️ 5. Module 5 体素化 A* 规划参数
    # ==========================================
    # 体素分辨率：越小路径越精细，但构建时间和内存随立方增长。
    # 典型飞机包围盒约 40×20×10m，0.5m 分辨率产生 ~64k 个体素，构建约 2~5s。
    # 若飞机体型较大或内存受限，可适当调大到 0.8 或 1.0。
    VOXEL_SIZE = 0.5                # A* 体素分辨率（米）

    # 高度代价权重（对应论文公式 5：J_altitude = h1*ȳ + h2*Var(y)）
    # 作用范围：A* 转场路径规划，引导路径优先走低空、高度变化平稳。
    # h1 单位：等效米/米高度（步代价附加量 = h1 × 邻格绝对高度）
    # h2 单位：等效米/米高度变化（步代价附加量 = h2 × 本步高度变化量）
    # 两者默认值远小于体素步长(0.5m)，对距离优化仅施加温和偏好，可按需调大。
    WEIGHT_ALT_MEAN = 0.01          # h1：A* 偏好低空路径的强度
    WEIGHT_ALT_VAR  = 0.20          # h2：A* 偏好平稳高度的强度

    # ==========================================
    # 📸 6. Module 5 视点悬停与 CSV 导出参数
    # ==========================================
    HOVER_TIME = 5.0                # 视点处悬停时长（秒）——用于拍照/数据采集
    CSV_FREQ = 10.0                 # CSV setpoint 输出频率（Hz）
                                    # PX4 offboard 模式建议 ≥ 2Hz，10Hz 足够平滑

    # ==========================================
    # 🌍 7. 地理与仿真原点配置 (Global Reference)
    # ==========================================
    ANCHOR_LAT = 47.397742
    ANCHOR_LON = 8.545594
    ANCHOR_ALT = 488.0

    # ==========================================
    # 🗂️ 8. 场景配置文件路径
    # ==========================================
    # 换场景时不必改源码：修改 scenes/default_scene.json 后重新 import 即可。
    # 若指定路径的文件不存在，保持上方默认值不变。
    SCENE_CONFIG_PATH = "scenes/default_scene.json"

    @classmethod
    def load_scene(cls, path: str = None) -> None:
        """从 JSON 文件加载场景参数（起飞点、无人机数量、地理锚点等）。

        Examples:
            Config.load_scene()                          # 读取默认场景文件
            Config.load_scene("scenes/hangar_north.json")  # 读取指定场景
        """
        import os, json
        target = path if path is not None else cls.SCENE_CONFIG_PATH
        if not os.path.exists(target):
            return
        with open(target, encoding="utf-8") as f:
            data = json.load(f)
        changed = []
        for k, v in data.items():
            if k.startswith("_"):
                continue
            if hasattr(cls, k):
                old = getattr(cls, k)
                setattr(cls, k, v)
                changed.append((k, old, v))
        if changed:
            print(f"[Config] 已从 {target} 加载场景参数:")
            for k, old, new in changed:
                print(f"  {k:<28} {str(old):<22} →  {new}")