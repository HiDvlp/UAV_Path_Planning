# UAV Path Planning — 运行指南

多无人机自主巡检路径规划系统，面向大型结构体（飞机蒙皮）的全覆盖视觉检测任务。

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境安装](#2-环境安装)
3. [快速开始](#3-快速开始)
4. [断点续跑](#4-断点续跑调试工作流)
5. [参数调整](#5-参数调整)
6. [输出文件说明](#6-输出文件说明)
7. [常见问题](#7-常见问题)

---

## 1. 项目结构

```
UAV_Path_Planning/
├── airplane_aligned.stl          # 输入：对齐好的飞机 STL 网格（必须存在）
│
├── main.py                       # 主入口：流水线调度 + 断点续跑
├── config.py                     # 全局参数配置（调参唯一入口）
│
├── module_1_preprocessing.py     # 阶段1：网格清理 + PCA 自适应特征采样
├── module_2_viewpoint.py         # 阶段2：候选视点生成 + 数字相机覆盖评估
├── module_3_set_cover.py         # 阶段3：质量感知 Lazy Greedy 集合覆盖
├── module_4_path_planning.py     # 阶段4：KMeans 任务分配 + 4D-TSP 排序
├── module_5_trajectory_optimization.py  # 阶段5：体素A* + Catmull-Rom + CSV
│
├── checkpoints/                  # 自动生成：各阶段中间检查点
│   ├── stage_1.pkl
│   ├── stage_2.pkl
│   ├── stage_3.pkl
│   └── stage_4.pkl
│
├── trajectories/                 # 自动生成：PX4 飞控指令
│   ├── uav_1_trajectory.csv
│   ├── uav_2_trajectory.csv
│   ├── uav_3_trajectory.csv
│   └── uav_4_trajectory.csv
│
├── airplane_preprocessed.stl     # 自动生成（阶段1）
├── 3_final_waypoints.ply         # 自动生成（阶段3）
├── 4_topo_lines.ply              # 自动生成（阶段4）
└── 5_final_trajectories.ply      # 自动生成（阶段5）
```

---

## 2. 环境安装

**Python 版本要求：** 3.8 ~ 3.11

```bash
pip install open3d numpy scipy scikit-learn ortools
```

| 库 | 用途 | 最低版本 |
|---|---|---|
| `open3d` | 网格处理、射线投射、点云导出 | 0.16 |
| `numpy` | 向量化数值计算 | 1.21 |
| `scipy` | KDTree 批量近邻查询（模块1） | 1.7 |
| `scikit-learn` | KMeans 任务聚类（模块4） | 1.0 |
| `ortools` | Google OR-Tools TSP 求解器（模块4） | 9.4 |

---

## 3. 快速开始

### 前提

将目标 STL 文件放置于项目根目录，命名为 `airplane_aligned.stl`。

### 完整运行（一键执行全部 5 个阶段）

```bash
cd UAV_Path_Planning
python main.py
```

运行完成后终端输出示例：

```
================================================================
  UAV 路径规划流水线  |  阶段 1 → 5
================================================================

================================================================
  阶段 1 / 5 : 网格预处理 & 特征点提取
================================================================
[Module 1] 启动预处理与特征提取 (airplane_aligned.stl)...
 -> 正在清理底层网格并提取顶点法向...
 -> 正在执行向量化 PCA 曲率计算与自适应采样...
    KDTree 批量近邻查询完成，耗时 12.3s
[Module 1] 预处理完成！特征点数: 12847  耗时: 45.2s
  [Checkpoint] 阶段 1 已保存 -> checkpoints/stage_1.pkl  (1.2 MB)
  [阶段 1 完成] 耗时 46.1s
...
================================================================
  流水线完成！总耗时: 831.4s
  轨迹 CSV: trajectories/uav_{1-4}_trajectory.csv
  可视化:  5_final_trajectories.ply
================================================================
```

### 查看结果

推荐使用 **CloudCompare**（免费）打开 PLY 文件：

```
File → Open → 选择 5_final_trajectories.ply
```

---

## 4. 断点续跑（调试工作流）

每个阶段完成后自动保存检查点，支持从任意阶段恢复，无需重跑上游耗时步骤。

### 命令格式

```bash
# 完整运行
python main.py

# 运行到第 N 阶段后停止
python main.py --to-stage N

# 从第 N 阶段开始（需要第 N-1 阶段的检查点）
python main.py --from-stage N

# 只运行阶段 M 到阶段 N
python main.py --from-stage M --to-stage N

# 查看已有检查点状态
python main.py --list-checkpoints
```

### 典型场景示例

**场景 A：首次运行，阶段2执行到一半中断**

```bash
# 重新从头运行（阶段1已有检查点则跳过）
python main.py --from-stage 1
```

**场景 B：调整覆盖质量阈值，重跑阶段3及之后**

```bash
# 1. 修改 module_3_set_cover.py 中的 quality_threshold_ratio
# 2. 从阶段3恢复（无需重跑耗时的阶段1、2）
python main.py --from-stage 3
```

**场景 C：调整飞行安全距离，只重跑轨迹生成**

```bash
# 1. 修改 config.py 中的 FLIGHT_SAFE_RADIUS
# 2. 只重新生成轨迹 CSV
python main.py --from-stage 5
```

**场景 D：调整无人机数量，重新分配任务**

```bash
# 1. 修改 config.py 中的 NUM_UAVS 和 TAKEOFF_POINTS
# 2. 从阶段4重跑
python main.py --from-stage 4
```

### 各阶段耗时参考

| 阶段 | 内容 | 典型耗时 | 检查点大小 |
|---|---|---|---|
| 1 | 网格清理 + PCA 特征采样 | 1 ~ 3 min | ~1 MB |
| 2 | 视点生成 + 相机仿真覆盖评估 | 5 ~ 20 min | ~50 MB |
| 3 | Lazy Greedy 集合覆盖优化 | 10 ~ 60 s | ~1 MB |
| 4 | KMeans + OR-Tools TSP | 30 ~ 120 s | ~1 MB |
| 5 | 体素A* + Catmull-Rom + CSV | 5 ~ 30 min | — |

> 阶段2是全流程最耗时的步骤（视点数量越多耗时越长），建议完成后务必保存检查点再调试下游模块。

---

## 5. 参数调整

所有参数集中在 `config.py`，按模块分区标注。

### 成像参数（影响阶段2）

```python
CAMERA_DISTANCE      = 5.0   # 常规拍摄距离（米）
UNDERBELLY_CAMERA_DIST = 3.0 # 机腹专属拍摄距离（米）
FOV_DEG              = 90.0  # 水平视场角（度）
MAX_INCIDENCE_ANGLE  = 45.0  # 最大允许拍摄入射角（度）
```

> 增大 `FOV_DEG` → 每个视点覆盖面积更大 → 最终航点更少  
> 减小 `CAMERA_DISTANCE` → 拍摄分辨率更高 → 视点密度更大

### 安全参数（影响阶段2、5）

```python
SAFE_RADIUS          = 3.0   # 视点生成安全半径（米）
FLIGHT_SAFE_RADIUS   = 2.0   # 轨迹全程安全距离（米）
MIN_SAFE_Z           = 0.5   # 最低飞行高度（米）
```

> `FLIGHT_SAFE_RADIUS` 越大，A* 可用通道越窄，规划失败概率越高  
> 若 A* 频繁退化为直线，可适当减小此值

### 多机参数（影响阶段4）

```python
NUM_UAVS             = 4
TAKEOFF_POINTS       = [
    [ 20.0,  20.0, 0.5],
    [ 20.0, -20.0, 0.5],
    [-20.0,  20.0, 0.5],
    [-20.0, -20.0, 0.5],
]
WEIGHT_CLIMB         = 3.0   # TSP 爬升代价权重
WEIGHT_TURN          = 2.0   # TSP 偏航代价权重
```

### 轨迹参数（影响阶段5）

```python
HOVER_TIME           = 5.0   # 视点处悬停时长（秒）
TRANSIT_SPEED        = 2.0   # 转场飞行速度（m/s）
CSV_FREQ             = 10.0  # setpoint 输出频率（Hz）
VOXEL_SIZE           = 0.5   # A* 体素分辨率（米）
```

> `VOXEL_SIZE` 越小，路径越精细，但体素图构建时间随立方增长  
> `HOVER_TIME` 控制每个视点的拍照时间，需与相机曝光时间匹配

---

## 6. 输出文件说明

### PLY 点云文件（可视化）

| 文件 | 内容 | 打开方式 |
|---|---|---|
| `3_final_waypoints.ply` | 最终拍照航点（橙色） | CloudCompare / MeshLab |
| `4_topo_lines.ply` | 4架无人机的拓扑路径（各色） | CloudCompare / MeshLab |
| `5_final_trajectories.ply` | 最终平滑轨迹（白色=视点，彩色=转场） | CloudCompare / MeshLab |

### CSV 轨迹文件（飞控指令）

每架无人机独立一个 CSV 文件，格式如下：

```
timestamp_s, x, y, z, yaw_rad, type
0.0000, 20.0000, 20.0000, 0.5000, 0.0000, TAKEOFF_START
0.1000, 20.0000, 20.0000, 0.5000, 0.0000, TAKEOFF
...
12.3000, 45.3210, 32.1245, 15.5000, 1.5708, WAYPOINT_START
...
17.3000, 45.3210, 32.1245, 15.5000, 1.5708, WAYPOINT_END
17.4000, 44.9870, 31.8800, 15.6100, 1.5708, TRANSIT
...
```

**type 字段说明：**

| type | 含义 |
|---|---|
| `TAKEOFF_START / END` | 起飞点悬停（起始/结束帧） |
| `WAYPOINT_START` | 视点悬停起始帧，此行 (x,y,z,yaw) 为精确拍照位置 |
| `WAYPOINT` | 视点悬停中间帧 |
| `WAYPOINT_END` | 视点悬停结束帧 |
| `TRANSIT` | 视点间转场飞行帧 |
| `LAND_START / END` | 返回起飞点落地等待帧 |

> PX4 Offboard 模式下以 10 Hz 读取此 CSV 下发 setpoint 即可直接使用。  
> `WAYPOINT_START` 行是触发相机快门的信号帧。

---

## 7. 常见问题

**Q：阶段2报错"未能生成任何合法候选视点"**

调整 `config.py` 中的安全半径：
```python
SAFE_RADIUS = 2.0           # 从 3.0 适当缩小
UNDERBELLY_SAFE_RADIUS = 0.8
```

---

**Q：阶段5 A\* 频繁打印"退化为直线"警告**

A\* 找不到满足安全距离的绕行路径时会退化。可尝试：
```python
FLIGHT_SAFE_RADIUS = 1.5    # 从 2.0 适当缩小
VOXEL_SIZE = 0.3            # 从 0.5 缩小提高搜索精度（耗时增加）
```

---

**Q：想从 1 架无人机开始测试**

```python
# config.py
NUM_UAVS = 1
TAKEOFF_POINTS = [[20.0, 20.0, 0.5]]
```

然后从阶段4重跑：
```bash
python main.py --from-stage 4
```

---

**Q：阶段1 PCA 计算太慢**

减少初始采样点数（在 `module_1_preprocessing.py` 第 33 行）：
```python
base_pcd = mesh.sample_points_uniformly(number_of_points=200_000)  # 原为 500_000
```

或缩小近邻数 K（同文件第 37 行）：
```python
K = 15   # 原为 30
```

---

**Q：如何验证轨迹没有碰撞**

打开 `5_final_trajectories.ply` 与 `airplane_preprocessed.stl`，在 CloudCompare 中叠加显示，检查彩色轨迹是否与灰色网格有交叉。

---

**Q：`checkpoints/` 目录占用空间过大**

阶段2的检查点（`stage_2.pkl`）包含完整覆盖质量矩阵，通常最大（50~200 MB）。确认阶段2结果满意后，可以只保留 `stage_4.pkl` 删除其余：

```bash
rm checkpoints/stage_1.pkl checkpoints/stage_2.pkl checkpoints/stage_3.pkl
```
