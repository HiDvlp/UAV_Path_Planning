# 多无人机覆盖巡检路径规划 — 算法详解

## 目录

1. [问题定义与总体框架](#1-问题定义与总体框架)
2. [阶段 1：网格预处理与特征点提取](#2-阶段-1网格预处理与特征点提取)
3. [阶段 2：候选视点生成与覆盖质量评估](#3-阶段-2候选视点生成与覆盖质量评估)
4. [阶段 3：质量感知集合覆盖优化](#4-阶段-3质量感知集合覆盖优化)
5. [阶段 4：多机任务分配与 TSP 拓扑排序](#5-阶段-4多机任务分配与-tsp-拓扑排序)
6. [阶段 5：无碰撞轨迹优化与 CSV 导出](#6-阶段-5无碰撞轨迹优化与-csv-导出)
7. [全局数据流与模块接口](#7-全局数据流与模块接口)
8. [参数敏感性速查](#8-参数敏感性速查)

---

## 1. 问题定义与总体框架

### 1.1 任务描述

**输入**：一架飞机的三维点云文件（`.pcd`），以及若干台无人机的起飞位置。

**输出**：每架无人机的飞行轨迹 CSV 文件，可直接导入 PX4 飞控以 Offboard 模式执行。

**约束条件**：

| 类别 | 具体要求 |
|------|---------|
| 覆盖完整性 | 飞机蒙皮所有可见区域均需被至少一个视点以足够质量拍摄 |
| 拍摄质量 | 入射角 ≤ 45°，拍摄距离接近 5m |
| 飞行安全 | 全程与蒙皮距离 ≥ 2m，飞行高度 ≥ 0.5m |
| 任务均衡 | 多机负载均衡，每机电池不超限 |
| 轨迹平滑 | 路径 $C^1$ 连续，速度方向无突变 |

### 1.2 总体流水线

```
原始点云 (.pcd)
    │
    ▼  阶段 1：网格预处理与特征点提取
    │  SHS-Net 有向法向估计 → Poisson 表面重建 → PCA 曲率自适应降采样
    │  输出：三角网格 + 特征点集合 {(p_i, n_i)}
    │
    ▼  阶段 2：候选视点生成与覆盖质量评估
    │  极坐标试探生成合法视点 → 虚拟相机射线阵列评估覆盖质量
    │  输出：视点集合 + 覆盖质量字典 {vp_id: {tri_id: Q_score}}
    │
    ▼  阶段 3：质量感知集合覆盖优化
    │  Lazy Greedy 次模优化，选出最小覆盖视点子集
    │  输出：精选航点集合（~85% 压缩率）
    │
    ▼  阶段 4：多机任务分配与 TSP 拓扑排序
    │  全局 TSP 排序 → BSAE 二分搜索负载均衡 → 接力点局部优化
    │  输出：每架无人机的有序航点路线
    │
    ▼  阶段 5：无碰撞轨迹优化与 CSV 导出
       体素 A* 避障 → Catmull-Rom 样条平滑 → 时间戳 setpoint 序列导出
       输出：uav_N_trajectory.csv（PX4 Offboard 格式）
```

---

## 2. 阶段 1：网格预处理与特征点提取

**对应文件**：`algorithms/module_1_preprocessing.py`

### 2.1 背景与动机

原始激光扫描点云有三个根本性缺陷，必须在进入下游算法前消除：

1. **无法向方向**：PCA 法向估计只给出法向轴，无法区分"朝外"和"朝内"（符号歧义）。后续视点生成需要知道相机应从哪侧拍摄。

2. **无拓扑结构**：点云没有面元、没有连接关系，无法做射线碰撞检测（判断视线是否被遮挡、路径是否穿越蒙皮）。

3. **点密度均匀**：平坦区域与复杂区域点密度相同。平坦区域不需要密集视点，均匀密度会导致下游视点数量爆炸。

**解决方案**：SHS-Net 估计有向法向 → Screened Poisson 重建三角网格 → PCA 曲率自适应降采样特征点。

---

### 2.2 步骤 1：读取点云

从 `.pcd` 文件读取原始 $(N, 3)$ 坐标数组，$N \approx 17.8$ 万点。

---

### 2.3 步骤 2：SHS-Net 有向法向估计

#### 2.3.1 问题与方法选择

传统 PCA 法向：对每个点取 $K$ 近邻，协方差矩阵最小特征值对应的特征向量即法向。缺陷：正负方向不确定，需要额外的符号传播（在复杂几何体上容易传播错误）。

SHS-Net（Shape-oriented Normal Estimation）通过深度学习同时利用局部 Patch 和全局 Shape 信息，直接输出有方向的单位法向，无需符号传播。

#### 2.3.2 SHS-Net 推理流程

对点云中每个查询点 $q_i$，推理过程分为五步：

**① 局部 Patch 构建**

用 KDTree 查找 $K=700$ 个近邻，设近邻坐标为 $\{p_k\}$：
- 中心化：$\tilde{p}_k = p_k - q_i$
- 归一化：$\hat{p}_k = \tilde{p}_k / d_{\max}$，其中 $d_{\max} = \max_k \|p_k - q_i\|$

归一化使 Patch 对绝对尺度不变，$K=700$ 足以捕获足够的局部几何细节。

**② PCA 旋转对齐（PCATrans）**

对归一化 Patch 做 SVD 分解：

$$[\tilde{P}]_{3 \times K} = U \Sigma V^T$$

其中 $U \in \mathbb{R}^{3 \times 3}$ 是局部主方向矩阵（即 PCA 坐标系基向量）。将 Patch 旋转到 PCA 坐标系：

$$\hat{P}_{\text{pca}} = U^T \cdot \hat{P}$$

**目的**：使网络输入对全局旋转具有不变性（旋转等变性），相同局部形状无论如何朝向，送入网络的 Patch 形状相同。

代码中通过批量 NumPy SVD（对 $(B, 3, K)$ 数组批量分解）消除了逐点 Python 循环：

```python
U, _, _ = np.linalg.svd(centered.transpose(0,2,1), full_matrices=False)  # U: (B,3,3)
rotated = np.einsum('bki,bij->bkj', centered, U)  # (B,K,3)
```

**③ 全局 Shape 采样**

从整个点云随机采样 $S=700$ 个点，做相同的中心化 + PCA 旋转处理，得到全局 Shape 特征。

**目的**：仅靠局部 Patch 无法判断法向朝向（局部对称），全局信息提供"哪侧是外部空间"的上下文。

**④ 网络前向推理**

```
Patch (B,K,3) + Shape (B,S,3) → Network → n_local (B,3)
```

网络输出的是 PCA 坐标系下的局部法向。

**⑤ PCA 逆变换回世界坐标系**

$$n_{\text{world}} = n_{\text{local}} \cdot U^T$$

（$U$ 是正交矩阵，逆 = 转置）

最后对所有法向做单位归一化：$\hat{n}_i = n_i / \|n_i\|$。

#### 2.3.3 批量推理优化

代码将 $N$ 个查询点按 `BATCH_SIZE=500`（CPU 建议用 64）分批处理，每批：
1. 从预计算的 KDTree 近邻索引中取出对应行（避免重复查询）
2. 批量 PCA 变换（向量化 SVD，无 Python 循环）
3. 向量化全局采样（用 `rng.integers` 有放回采样，与原版无放回等价）
4. 一次 GPU/CPU forward pass

内存控制：全量近邻索引 $(N, K)$ 约 1GB，超过阈值时分段查询。

---

### 2.4 步骤 3：Screened Poisson 表面重建

#### 2.4.1 算法原理

Screened Poisson 重建将带有方向法向的点云视为梯度场的样本，求解泊松方程：

$$\Delta \chi = \nabla \cdot \vec{V}$$

其中 $\chi$ 是待求的指示函数（内部 1，外部 0），$\vec{V}$ 是由点云法向构成的向量场。指示函数的等值面即为重建网格。

"Screened"是指在求解时加入点位置的约束项，防止网格偏离原始点云。

参数 `depth=10` 对应 $2^{10} = 1024$ 级八叉树分辨率，飞机尺度（~40m）下约 4cm 网格精度。

#### 2.4.2 后处理：去除虚假面片

Poisson 重建会在远离点云的包围盒边界处产生低密度虚假面片，需两步清理：

**密度过滤**：重建时同时输出每个顶点的密度估计值，删除密度最低 5% 分位数的顶点。

**包围盒裁剪**：额外安全网，删除超出点云包围盒 5%（对角线长度）的顶点：
```python
bbox_margin = bbox_diag * 0.05
pts_min = points.min(0) - bbox_margin
pts_max = points.max(0) + bbox_margin
```

最后执行标准网格清理：去除退化三角形、重复顶点、非流形边，重新计算顶点法向和面元法向。

实际输出：约 375k 顶点，746k 三角面元。

---

### 2.5 步骤 4：构建射线投射场景

将三角网格注册到 Open3D `RaycastingScene`：

```python
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
ray_scene = o3d.t.geometry.RaycastingScene()
ray_scene.add_triangles(mesh_t)
```

此场景在后续阶段 2、5 中共享使用，提供两类服务：
- `cast_rays()`：批量射线投射，返回命中的三角面元 ID 和距离
- `compute_distance()`：批量计算点到网格表面的最近距离

---

### 2.6 步骤 5：PCA 曲率计算

#### 2.6.1 曲率定义

对每个点取 $K=30$ 近邻，计算邻域协方差矩阵的特征值 $\lambda_1 \leq \lambda_2 \leq \lambda_3$：

$$\kappa_i = \frac{\lambda_1}{\lambda_1 + \lambda_2 + \lambda_3}$$

- $\kappa \approx 0$：平坦区域（$\lambda_1 \ll \lambda_2, \lambda_3$）
- $\kappa \approx 1/3$：各向同性区域（球面、角点）

#### 2.6.2 实现优化

**复用近邻索引**：SHS-Net 已建立 $(N, 700)$ 的近邻索引，曲率计算只需前 30 列，直接截取复用，避免重新建树：

```python
knn_idx = precomputed_knn_idx[:, :30]  # 截取，不重新查询
```

**向量化批量计算**（无 Python 逐点循环）：

```python
nbrs = points[knn_idx[start:end]]          # (B, K, 3)
ctrd = nbrs - nbrs.mean(axis=1, keepdims=True)  # (B, K, 3)
covs = np.einsum('nki,nkj->nij', ctrd, ctrd) / K  # (B, 3, 3)
eigs = np.linalg.eigvalsh(covs)             # (B, 3) 升序
curvatures = eigs[:,0] / eigs.sum(axis=1)  # (B,)
```

---

### 2.7 步骤 6：自适应降采样

**依据**：高曲率区域（边缘、棱角、翼梢）需要更密集的视点采样；平坦区域（机身蒙皮大面）可以稀疏采样。

**策略**：

| 区域 | 判据 | 保留比例 |
|------|------|---------|
| 高曲率 | $\kappa > 0.015$ | 50% |
| 低曲率 | $\kappa \leq 0.015$ | 10% |

随机采样在各区域内独立进行（固定随机种子保证可复现）。

实际效果：177,844 点 → 31,040 特征点，压缩约 83%，高曲率区域保留密度约 5× 低曲率区域。

---

## 3. 阶段 2：候选视点生成与覆盖质量评估

**对应文件**：`algorithms/module_2_viewpoint.py`

### 3.1 总体思路

阶段 2 分两个子阶段：

1. **视点生成**：对每个特征点，在其法向方向上找一个合法的拍摄位置（满足安全距离、高度限制、视线无遮挡）
2. **质量评估**：对找到的所有视点，模拟真实数字相机的视锥射线，计算每个视点能以什么质量覆盖哪些三角面元

### 3.2 预计算：相机物理参数与视锥射线阵列

**视锥射线阵列**（在构造函数中一次性预计算）：

模拟 `ray_res_x × ray_res_y = 200 × 133 = 26,600` 像素的数字相机，每个像素对应一条光线：

$$u \in \left[-\tan\frac{\theta_h}{2},\ \tan\frac{\theta_h}{2}\right], \quad v \in \left[-\tan\frac{\theta_v}{2},\ \tan\frac{\theta_v}{2}\right]$$

其中 $\theta_h = 90°$（水平视场角），$\theta_v = \theta_h / \text{aspect\_ratio}$（竖直视场角）。

归一化方向向量 $\vec{d}_{\text{cam}} = (u, v, 1) / \|(u,v,1)\|$，存储为 $(R, 3)$ 数组（$R=26600$），整个评估阶段共享。

**三角面元属性预计算**：

一次性计算网格所有三角面元的法向和面积：

$$S_j = \frac{1}{2} \|\vec{e}_1 \times \vec{e}_2\|$$

其中 $\vec{e}_1, \vec{e}_2$ 为面元两条边向量。

---

### 3.3 子阶段 1：极坐标交替试探视点生成

#### 3.3.1 算法框架

对 $N$ 个特征点，维护一个"已解决"掩码 `is_solved`。双层循环遍历距离偏移 $\Delta d$ 和偏转角 $\theta$，每轮对所有"未解决"的点同时尝试，找到合法视点即标记为已解决：

```
for Δd in [0.0, 0.5, 1.0, 1.5, 2.0]:      # 拉远距离
    for θ in [0°, 15°, 30°, 45°]:           # 偏转角度
        对所有 unsolved 点批量生成视点候选
        批量安全检验
        将通过检验的点标记为 solved
```

设计逻辑：先尝试正对法向方向（$\theta=0°$）的"标准位置"，若被障碍物遮挡则逐步拉远、倾斜，找到第一个可用位置即止，避免搜索过多候选。

#### 3.3.2 极坐标方向生成

对当前偏转角 $\theta$，在法向锥面上以 $\phi$ 均匀分布方位角（$\theta=0$ 时只有一个方向，其余时有 8 个）：

首先构建切平面正交基 $(\hat{T}_1, \hat{T}_2)$：

$$\hat{T}_1 = \frac{\hat{W} \times \hat{n}}{\|\hat{W} \times \hat{n}\|}, \quad \hat{T}_2 = \hat{n} \times \hat{T}_1$$

其中 $\hat{W} = (0,0,1)$（若法向与 $\hat{W}$ 几乎平行，则换用 $\hat{e}_1 = (1,0,0)$）。

方向向量：

$$\vec{d}(\theta, \phi) = \cos\theta \cdot \hat{n} + \sin\theta \cdot (\cos\phi \cdot \hat{T}_1 + \sin\phi \cdot \hat{T}_2)$$

视点候选坐标：

$$V = p + (R_{\text{base}} + \Delta d) \cdot \vec{d}(\theta, \phi)$$

其中 $R_{\text{base}}$ 根据法向是否朝下（机腹）区别对待：机腹视点使用更小的拍摄距离 $R=3\text{m}$ 和安全半径 $r=1\text{m}$。

代码全程向量化，所有未解决点在一次 NumPy 操作中同时生成候选并检验。

#### 3.3.3 三重安全检验

对每个候选视点 $V$（向量化批量执行）：

**① 高度约束**
$$V_z \geq \texttt{MIN\_SAFE\_Z} = 0.5 \text{ m}$$

**② 障碍物距离约束**

调用 `ray_scene.compute_distance(V)`，计算 $V$ 到蒙皮的最近距离，要求：
$$\text{dist}(V) \geq r_{\text{safe}}$$

**③ 视线无遮挡约束**

构造射线 $V \to p$，调用 `cast_rays()`，要求命中距离 $t_{\text{hit}} \geq \|V - p\| - 0.1$（0.1m 容差，防止浮点误差误判）。

三个条件全部满足，该视点才合法。对每个特征点，取通过检验的**第一个**方向对应的视点。

#### 3.3.4 视点池精简

收集所有合法视点，对视点池做 **体素降采样**（`voxel_size=0.5m`），去除过度密集的重叠视点（相邻特征点生成的视点可能几乎重合，保留一个即可）。

---

### 3.4 子阶段 2：虚拟相机射线阵列质量评估

#### 3.4.1 向量化相机坐标系构建

**批量 KDTree 查询**：一次调用 `tree.query(all_vps)` 获取所有视点的最近目标点，替代循环内的单点查询。

**向量化计算所有视点的相机坐标系**（$(K, 3)$ 级别操作）：

光轴方向（Z轴）：
$$\hat{Z}_i = \frac{p_{\text{nearest}}^{(i)} - V_i}{\|p_{\text{nearest}}^{(i)} - V_i\|}$$

横轴（X轴，保证 Roll = 0）：
$$X_{\text{raw}} = \begin{cases} \hat{W} \times \hat{Z} & |\hat{Z}_z| \leq 0.999 \\ \hat{e}_1 \times \hat{Z} & |\hat{Z}_z| > 0.999 \end{cases}$$
$$\hat{X} = X_{\text{raw}} / \|X_{\text{raw}}\|$$

纵轴：$\hat{Y} = \hat{Z} \times \hat{X}$

用 `np.where` 广播同时处理所有 $K$ 个视点，无 Python 循环。

#### 3.4.2 分批射线投射

每批 `CHUNK=64` 个视点：

**① 计算世界坐标系射线方向**

$$D_{\text{world}}^{(i,r)} = d_r^x \hat{X}_i + d_r^y \hat{Y}_i + d_r^z \hat{Z}_i$$

用 NumPy 广播一次计算整批：

```python
D_world = (dir_cam[None,:,0:1] * X_cams[:,None,:] +   # (C, R, 3)
           dir_cam[None,:,1:2] * Y_cams[:,None,:] +
           dir_cam[None,:,2:3] * Z_cams[:,None,:])
```

**② 构建射线张量并批量投射**

将 $C \times R$ 条射线拼成 $(C \cdot R, 6)$ 张量（前 3 列为起点，后 3 列为方向），一次 `cast_rays()` 调用返回所有命中结果。

批量大小 $C=64$ 时，单次射线张量约 40MB，在常见机器上不会造成内存压力。

**③ 逐视点统计覆盖质量**（在已获得批量结果后，逐视点整理）

对视点 $V_i$ 命中的所有射线：

- 过滤未命中射线（`hit_id == INVALID_ID`）
- 取每个唯一三角面元 $t_j$ 的**代表射线**（`np.unique` 取首次命中）
- 计算入射角余弦：$\cos\alpha_{ij} = -\vec{d}_r \cdot \hat{n}_j$（射线方向与面元法向的反向点积）
- 过滤入射角超限（$\cos\alpha < \cos 45°$）的面元

#### 3.4.3 覆盖质量分数

对视点 $V_i$ 覆盖的每个合格三角面元 $t_j$，计算质量分数：

$$Q(i,j) = \underbrace{S_j}_{\text{面积权重}} \cdot \underbrace{\cos\alpha_{ij}}_{\text{角度质量}} \cdot \underbrace{\exp\!\left(-\frac{(d_{ij} - D_{\text{opt}})^2}{\sigma^2}\right)}_{\text{距离质量}}$$

各项含义：
- $S_j$：三角面元真实物理面积（m²），面积越大的面元权重越高
- $\cos\alpha_{ij}$：入射角余弦，$\alpha=0°$ 时为 1（正对拍摄），$\alpha=45°$ 时为 $\frac{\sqrt{2}}{2}$
- 高斯距离衰减：$d_{ij}$ 越接近理想拍摄距离 $D_{\text{opt}}=5\text{m}$，得分越高；$\sigma^2 = 8.0$ 控制宽容度

输出：`coverage_dict = {vp_id: {tri_id: Q_score, ...}}`，仅存储非零分数的面元，稀疏存储。

#### 3.4.4 统计覆盖率

在评估完所有视点后，统计全局真实物理覆盖率：

$$\text{Coverage Rate} = \frac{\sum_{j \in \text{covered}} S_j}{\sum_j S_j} \times 100\%$$

"已覆盖"定义为被至少一个视点以有效质量拍摄的面元（即在 `coverage_dict` 中出现的面元）。

---

## 4. 阶段 3：质量感知集合覆盖优化

**对应文件**：`algorithms/module_3_set_cover.py`

### 4.1 问题建模

**输入**：`coverage_dict`，包含数千个候选视点和数十万个待覆盖三角面元。

**目标**：选出尽量少的视点子集 $S^* \subseteq V$，使每个三角面元 $t_j$ 的覆盖质量达标：

$$\sum_{i \in S^*} Q(i,j) \geq Q_{\text{thresh}}(j) = 0.85 \cdot \max_i Q(i,j)$$

直接意义：**不要求绝对完美拍摄，只要达到该面元理论最高拍摄质量的 85% 即可**。

这是一个 **NP-hard** 的加权集合覆盖问题，但其目标函数具有次模性，可用贪心近似求解。

### 4.2 次模函数与 Greedy 近似理论

**次模函数**：定义覆盖收益函数：

$$F(S) = \sum_j \min\!\left(\sum_{i \in S} Q(i,j),\ Q_{\text{thresh}}(j)\right)$$

这是一个**单调次模函数**（满足边际收益递减：$S \subseteq T \Rightarrow F(v|S) \geq F(v|T)$），其中 $F(v|S) = F(S \cup \{v\}) - F(S)$ 为边际收益。

**Greedy 近似保证**：对单调次模函数的最大化问题，Greedy 算法（每步选边际收益最大的元素）可达到 $(1 - 1/e) \approx 63.2\%$ 的最优比。

### 4.3 Lazy Greedy 算法

标准 Greedy 每轮需要重新计算所有 $|V|$ 个视点的边际收益，总时间复杂度 $O(|V|^2 \cdot |T|)$，对数千视点和数十万面元来说不可接受。

**Lazy Greedy 优化**：利用次模性的关键性质——**某视点的边际收益在当前轮只会比上一轮计算时更低（不会更高）**。

用优先队列（大根堆）存储"各视点上次计算的边际收益"，每次弹出堆顶时：
- 若其边际收益是**本轮刚算的**：直接录取（因为没有视点能比它更高）
- 若其边际收益是**过期的**：重新计算，放回堆中

```
初始化：
  for v in 所有视点：
    gain_v = Σ_j min(Q(v,j), thresh(j))   # 初始边际 = 全量收益
    push(-gain_v, iter=0, v) 到优先堆

while 堆非空：
    (-gain, last_iter, v) = 堆顶弹出
    
    if last_iter == current_iter:      ← 收益是最新的
        录取 v 到 selected
        current_scores += Q(v, ·)
        current_iter += 1
        continue
    
    # 收益过期，按最新战况重算
    deficit = thresh - current_scores  ← 各面元还差多少
    valid = deficit > 0                ← 只考虑还未达标的面元
    new_gain = Σ min(Q(v,j)[valid], deficit[valid])
    
    if new_gain > 0:
        push(-new_gain, iter=current_iter, v)
    # else: 该视点对所有面元均无增量贡献，丢弃
```

**关键性质**：由于次模性，过期的 `gain` 必然高于重算的 `new_gain`。因此堆顶弹出时，若收益是当前轮的，则它必然是所有视点中真实边际收益最大的，无需检查其他视点。

**向量化实现**：每次重算边际收益用 NumPy 向量操作：

```python
deficits = self.thresholds_arr[indices] - current_scores[indices]
valid = deficits > 0
new_gain = np.sum(np.minimum(scores[valid], deficits[valid]))
```

### 4.4 数据结构准备

在初始化时，将稀疏的 `coverage_dict` 转换为高速索引结构：

```python
# 每个视点的覆盖信息转为 (面元连续整数索引, 分数) 对
self.vp_data[vp] = (indices_array, scores_array)  # (ndarray, ndarray)
```

面元 ID 到连续整数索引的映射（`tri_to_idx`）使向量化索引成为可能。

### 4.5 实际效果

典型数据：7,237 个候选视点，463,262 个待覆盖三角面元 → 6,162 个精选航点，压缩至 85%，算法耗时约 4 秒。

---

## 5. 阶段 4：多机任务分配与 TSP 拓扑排序

**对应文件**：`algorithms/module_4_path_planning.py`

### 5.1 问题背景

$M \approx 6,162$ 个精选航点需要分配给 $K=4$ 架无人机，要求：
- 所有航点被覆盖（全覆盖约束）
- 各机电池不超限（能量约束）
- 各机总飞行时间尽量均衡（负载均衡）
- 每机内部航点访问顺序尽量短（局部 TSP 最优）

### 5.2 MUCS-BSAE 算法框架

**三阶段流程**：

```
全局 TSP 排序 → 二分搜索负载均衡 (BSAE) → 接力点局部优化 (IMUCS)
```

### 5.3 阶段 A：全局 TSP 排序

**动机**：直接对 6,162 个航点做 K-means 聚类再独立 TSP，各机路径之间可能"交错"——邻近航点被分到不同机器，任何一机都在全局范围内迂回。先做全局 TSP 建立一条"最短哈密顿路径"，再沿此路径切段分配，可在分配阶段天然保持局部聚集性。

**4D 代价矩阵**：

$$C(i,j) = d_{\text{Eucl}}(i,j) + w_{\text{climb}} \cdot |\Delta z_{ij}| + w_{\text{turn}} \cdot |\Delta\psi_{ij}|$$

- $d_{\text{Eucl}}$：欧几里得距离（主项）
- $w_{\text{climb}} \cdot |\Delta z|$：高度变化惩罚（$w=3.0$），减少频繁升降的能耗
- $w_{\text{turn}} \cdot |\Delta\psi|$：偏航角变化惩罚（$w=2.0$），减少大角度转弯

偏航角按当前实现取视点指向坐标原点的方向。

**OR-Tools TSP 求解器**：

```python
search_params.first_solution_strategy = PATH_CHEAPEST_ARC
search_params.time_limit.seconds = 120  # 全局 TSP 通常需要 80~200s
```

`PATH_CHEAPEST_ARC`：从起点出发，每步选代价最小的未访问节点连接，快速构造初始解后局部搜索改进。

### 5.4 阶段 B：二分搜索负载均衡（BSAE）

#### 5.4.1 核心思想

给定一个"时间预算" $T$，贪心地为每架无人机从全局有序序列中分配视点——当前无人机"吃"到将超出时间预算或电量预算时停止，下一架从断点继续。

找到**最小可行时间预算 $T^*$** 使得 $K$ 架无人机恰好能覆盖全部 $M$ 个视点，即为最优负载均衡。

#### 5.4.2 贪心分配（`_bsae_segment`）

从 `global_order[start_wp_idx:]` 开始，为第 $k$ 架无人机贪心分配：

对序列中的第 $i$ 个视点 $w_i$，计算加入该视点后的**前瞻预算**：

$$t_{\text{elapsed}} + \underbrace{\frac{\|w_i - \text{pos}\|}{v}}_{\text{飞到此点}} + \underbrace{t_{\text{hover}}}_{\text{悬停拍照}} + \underbrace{\frac{\|w_i - \text{base}\|}{v}}_{\text{返回基地}} \leq T$$

$$E_{\text{accum}} + (d_{\text{flight}} + d_{\text{return}}) \cdot E_{\text{meter}} + t_{\text{hover}} \cdot E_{\text{hover}} \leq B_{\text{battery}}$$

若任一约束即将违反，停止分配，将 `(assigned_wps, next_start_idx)` 返回。

**关键设计**：前瞻检查包含"返回基地"的代价，确保无人机在任何时刻都有足够能量返回，不会困在外面。

#### 5.4.3 二分搜索最小可行 $T$

```
T_max = 单机飞完全程所需时间
T_lo, T_hi = 0, T_max

while T_lo < T_hi - 1:
    T_mid = (T_lo + T_hi) // 2
    ok, segs = try_budget(T_mid)     # 用 T_mid 尝试分配，检查是否覆盖全部 M 个视点
    if ok: T_hi, best_segs = T_mid, segs
    else:  T_lo = T_mid

return best_segs  # 使用最小可行 T_hi 对应的分配结果
```

二分搜索在整数秒上进行（$T_{\max}$ 通常在数百到数千秒之间），循环次数约 $\log_2 T_{\max} \approx 10\text{~}12$ 次。

#### 5.4.4 回退策略

若电池约束过紧（即 $T_{\max}$ 也无法完成分配），则退化为简单均分：第 $k$ 架无人机分配 `global_order[k*M//K : (k+1)*M//K]`。

### 5.5 阶段 C：接力点局部优化（IMUCS）

**问题**：BSAE 分配的子任务段顺序继承自全局 TSP，但全局 TSP 的起点是所有机器的"虚拟中心"，某架无人机分到的段起点可能离该机真实基地很远，造成无效空驶。

**解决方案**：

1. 在子任务段中，找**距本机起飞点最近的视点**作为段首（"接力点"）
2. 以接力点为新起点，对剩余视点重新求解局部 TSP

```python
relay_idx = argmin(dist(seg_pts, uav_takeoff))   # 找最近视点
relay_vp  = segment[relay_idx]
remaining = segment \ {relay_vp}
ordered   = solve_tsp(relay_vp, remaining)        # 以接力点为起点的 TSP
return [relay_vp] + ordered
```

这确保每架无人机从最近的视点出发，极大减少无效飞行距离。

### 5.6 路线组装与可视化

每架无人机的最终路线：

$$\text{route\_pts} = [\underbrace{P_{\text{takeoff}}}_{\text{起飞点}}, w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(N_k)}, \underbrace{P_{\text{takeoff}}}_{\text{返回}}]$$

可视化：在每段路径上以 0.05m 间距均匀插值，生成高密度彩色点云导出为 `4_topo_lines.ply`。

---

## 6. 阶段 5：无碰撞轨迹优化与 CSV 导出

**对应文件**：`algorithms/module_5_trajectory_optimization.py`

### 6.1 层次架构

```
CollisionChecker          ← 距离场，判定点/线段安全性
      ↓
VoxelAStarPlanner         ← 体素 A*，生成无碰撞折线路径
      ↓
_smooth_catmull_rom()     ← Catmull-Rom 样条，平滑折线
      ↓
UAVTrajectoryPlanner      ← 添加时间戳，生成 setpoint 序列
      ↓
export_trajectory_csv()   ← 写入 PX4 兼容 CSV
```

### 6.2 碰撞检测器（CollisionChecker）

基于 Open3D `RaycastingScene.compute_distance()`，批量计算任意点集到网格蒙皮的最近欧氏距离。

**单点安全判定**：
$$\text{is\_safe}(p) = [\text{dist}(p) \geq r_{\text{flight}}]$$

其中 $r_{\text{flight}} = \texttt{FLIGHT\_SAFE\_RADIUS} = 2\text{m}$。

**线段安全判定**：在线段 $p_1 \to p_2$ 上均匀采样 $n=12$ 个点，全部满足安全距离则线段安全：

$$\text{is\_safe\_segment}(p_1, p_2) = \bigwedge_{k=0}^{n-1} \text{is\_safe}\!\left(p_1 + \frac{k}{n-1}(p_2-p_1)\right)$$

注意：`FLIGHT_SAFE_RADIUS=2m` 专用于轨迹安全检测，与阶段 2 的视点生成安全半径 `SAFE_RADIUS=3m` 是两个独立概念（视点生成要求更大间距，飞行转场可以靠近些）。

### 6.3 体素 A\* 规划器（VoxelAStarPlanner）

#### 6.3.1 体素安全图构建

在飞机包围盒（含 $2r_{\text{flight}}$ 安全余量）内建立均匀体素网格：

- 分辨率 `VOXEL_SIZE = 0.5m`
- 实际尺寸约 $92 \times 99 \times 42 \approx 38$ 万体素
- 安全体素比例约 75%（25% 被蒙皮及其安全缓冲区占据）

批量查询所有体素中心的距离，标记安全（`True`）/不安全（`False`）：

```python
safe_flat = dists >= checker.safe_radius
self.safe_grid = safe_flat.reshape(grid_shape)  # (nx, ny, nz) bool 数组
```

额外约束：高度 $z < \texttt{MIN\_SAFE\_Z}$ 的体素强制标记为不安全。

#### 6.3.2 坐标转换

世界坐标 ↔ 体素索引：

$$\text{idx} = \text{clip}\!\left(\text{round}\!\left(\frac{p - \text{origin}}{v_s}\right), 0, \text{shape}-1\right)$$

$$p = \text{origin} + \text{idx} \cdot v_s$$

#### 6.3.3 A\* 核心算法

**启发函数（可接纳性）**：

使用欧几里得距离，始终 $\leq$ 真实路径代价（可接纳，保证找到最优路径）：

$$h(a, b) = \sqrt{(\Delta x)^2 + (\Delta y)^2 + (\Delta z)^2} \cdot v_s$$

注意：若用曼哈顿距离 $(|\Delta x| + |\Delta y| + |\Delta z|) \cdot v_s$，在 26 连通格网中**高估**斜向移动代价（如 $3 > \sqrt{3}$ for $(1,1,1)$ 方向），导致不可接纳，可能跳过最优路径。

**步代价（含高度偏好）**：

$$c(n_i \to n_j) = \underbrace{\|n_i - n_j\|_2 \cdot v_s}_{\text{欧氏距离}} + \underbrace{h_1 \cdot z_j}_{\text{低空偏好}} + \underbrace{h_2 \cdot |\Delta z_{ij}| \cdot v_s}_{\text{平稳偏好}}$$

参数 $h_1 = 0.01$，$h_2 = 0.20$（远小于体素步长，不影响避障优先级，仅施加温和高度偏好）。

**搜索流程**：

```
open_heap = [(0, 0, s_idx)]   # (f_score, 计数器, 体素索引)
g_score   = {s_idx: 0.0}
came_from = {}
closed    = set()

while open_heap:
    _, _, current = heappop(open_heap)
    if current in closed: continue
    closed.add(current)
    if current == g_idx: found!
    
    for neighbor in 26-neighbors(current):
        if not safe_grid[neighbor]: skip
        tent_g = g_score[current] + step_cost(current, neighbor)
        if tent_g < g_score.get(neighbor, ∞):
            g_score[neighbor] = tent_g
            came_from[neighbor] = current
            f = tent_g + h(neighbor, g_idx)
            heappush(open_heap, (f, counter++, neighbor))
```

**快速路优化**：在启动 A\* 前，先用 `is_safe_segment(start, goal, n=20)` 检查是否可直飞。直飞成功则直接返回，跳过 A\* 搜索（实际运行中约 60~70% 的段可直飞）。

**退化策略**：若 A\* 搜索失败（安全通道极窄或参数设置过严），退化为直线并打印警告，不中断流程。

#### 6.3.4 贪心可视性剪枝

A\* 回溯路径为一系列体素中心，存在大量"共线"或"可视"的中间节点。贪心剪枝从前向后跳跃：

```
pruned = [path[0]]
i = 0
while i < len(path)-1:
    j = len(path)-1
    while j > i+1:
        if is_safe_segment(path[i], path[j]):
            break        # 找到最远可直飞节点
        j -= 1
    pruned.append(path[j])
    i = j
```

通过逐段测试安全性（均匀采样 15 点），跳过所有中间节点，显著减少路径点数量。

### 6.4 Catmull-Rom 样条平滑

A\* 输出为折线路径，转折处存在速度方向突变（对飞控不友好）。Catmull-Rom 样条保证相邻段在连接点处**切线方向连续**（$C^1$）。

对路径点序列 $P = [p_0, p_1, \ldots, p_n]$，在 $p_i \to p_{i+1}$ 段生成插值点：

$$P(t) = \frac{1}{2}\begin{bmatrix}1 & t & t^2 & t^3\end{bmatrix} \begin{bmatrix}0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1\end{bmatrix} \begin{bmatrix}p_{i-1} \\ p_i \\ p_{i+1} \\ p_{i+2}\end{bmatrix}$$

参数 $t \in [0,1)$，每段生成 20 个插值点（不含终点），最后追加精确终点。

**边界处理**：首尾各添加反向幻影控制点（$p_{-1} = p_0 + (p_0 - p_1)$），保证端点切线方向正确（速度方向与离开/到达方向一致）。

### 6.5 轨迹时序生成（UAVTrajectoryPlanner）

将几何路径转换为带时间戳的飞控指令序列，采用**"停走停"模式**：

**视点悬停段**（`HOVER_TIME=5s`）：

在视点处以 `CSV_FREQ=10Hz` 输出静止 setpoint：
- 首帧：`WAYPOINT_START`（触发相机快门的信号帧）
- 中间帧：`WAYPOINT`
- 末帧：`WAYPOINT_END`

**转场飞行段**：

将平滑路径按 `TRANSIT_SPEED=2m/s` 做**弧长重采样**（等弧长间距），输出 `TRANSIT` 类型 setpoint。

向量化弧长参数化插值：

```python
# 计算累积弧长
cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
# 等间距弧长采样点
s_vals = np.linspace(0, total_len, n_steps)
# 向量化插值（searchsorted 定位所在段，线性插值）
idxs  = np.searchsorted(cumlen, s_vals) - 1
t_loc = (s_vals - cumlen[idxs]) / seg_lengths[idxs]
pts   = path[idxs] + t_loc[:,None] * diffs[idxs]
```

**偏航角策略**：转场段保持出发视点的 yaw 值不变（无 yaw 旋转要求），仅在视点处精确对齐到拍摄方向。

### 6.6 CSV 格式与 PX4 兼容性

输出格式（每架无人机独立一个文件）：

```
timestamp_s, x, y, z, yaw_rad, type
0.0000, 20.0000, 20.0000, 0.5000, 0.000000, TAKEOFF_START
0.1000, 20.0000, 20.0000, 0.5000, 0.000000, TAKEOFF
...
12.3000, 45.3210, 32.1245, 15.5000, 1.570796, WAYPOINT_START
...
17.3000, 45.3210, 32.1245, 15.5000, 1.570796, WAYPOINT_END
17.4000, 44.9870, 31.8800, 15.6100, 1.570796, TRANSIT
...
```

**type 字段含义**：

| type | 含义 |
|------|------|
| `TAKEOFF_START / TAKEOFF / TAKEOFF_END` | 起飞点稳定悬停（1s） |
| `WAYPOINT_START` | 视点悬停起始帧，`(x,y,z,yaw)` 为精确拍摄位置，触发快门 |
| `WAYPOINT` | 视点悬停中间帧 |
| `WAYPOINT_END` | 视点悬停结束帧 |
| `TRANSIT` | 视点间转场飞行帧 |
| `LAND_START / LAND / LAND_END` | 返回起飞点落地等待（2s） |

PX4 Offboard 模式建议 setpoint 频率 $\geq 2$Hz，10Hz 输出留有充足余量。

---

## 7. 全局数据流与模块接口

```
data/airplane_aligned.pcd
         │ 读取 (N,3) 点云
         ▼
┌─────────────────────────────────┐
│  Module 1: 预处理               │
│  输入: pcd 文件路径              │
│  输出: mesh (TriangleMesh)       │
│        pts   (M,3) float32      │
│        norms (M,3) float32      │
│        ray_scene (RaycastingScene)│
└──────────┬──────────────────────┘
           │ (pts, norms) → stage_1.pkl
           ▼
┌─────────────────────────────────┐
│  Module 2: 视点生成              │
│  输入: ray_scene, mesh, pts, norms│
│  输出: valid_viewpoints (K,3)   │
│        coverage_dict            │
│        {vp_id: {tri_id: Q}}     │
└──────────┬──────────────────────┘
           │ → stage_2.pkl (~50MB)
           ▼
┌─────────────────────────────────┐
│  Module 3: 集合覆盖              │
│  输入: coverage_dict            │
│        valid_viewpoints         │
│  输出: final_waypoints (N,3)    │
└──────────┬──────────────────────┘
           │ → stage_3.pkl (~1MB)
           ▼
┌─────────────────────────────────┐
│  Module 4: 任务分配              │
│  输入: final_waypoints (N,3)    │
│        takeoff_points (K,3)     │
│  输出: all_routes               │
│   [ (route_pts(Ni+2,3),         │
│      route_yaws(Ni,)) × K ]     │
└──────────┬──────────────────────┘
           │ → stage_4.pkl (~1MB)
           ▼
┌─────────────────────────────────┐
│  Module 5: 轨迹优化              │
│  输入: all_routes, mesh         │
│  输出: smooth_paths (几何)      │
│        uav_N_trajectory.csv × K │
└─────────────────────────────────┘
           │
           ▼
output/visualizations/*.ply   ← 可视化
output/trajectories/*.csv     ← PX4 飞控指令
```

---

## 8. 参数敏感性速查

### 成像参数（影响阶段 2）

| 参数 | 默认值 | 增大效果 | 减小效果 |
|------|--------|---------|---------|
| `CAMERA_DISTANCE` | 5.0 m | 覆盖面积↑，分辨率↓ | 分辨率↑，视点密度↑ |
| `FOV_DEG` | 90° | 单视点覆盖↑，视点数↓ | 单视点覆盖↓，视点数↑ |
| `MAX_INCIDENCE_ANGLE` | 45° | 斜面也可覆盖，质量↓ | 仅正面拍摄，覆盖率↓ |
| `DIST_SCORE_SIGMA2` | 8.0 | 对距离偏差容忍↑ | 对拍摄距离要求更严 |

### 安全参数（影响阶段 2、5）

| 参数 | 默认值 | 增大效果 | 减小效果 |
|------|--------|---------|---------|
| `SAFE_RADIUS` | 3.0 m | 视点生成成功率↓ | 视点可更靠近蒙皮 |
| `FLIGHT_SAFE_RADIUS` | 2.0 m | A\* 可用通道↓，失败率↑ | 路径可更贴近蒙皮 |
| `MIN_SAFE_Z` | 0.5 m | 低空飞行被禁止 | 可在地面附近飞行 |

### 多机参数（影响阶段 4）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_UAVS` | 4 | 无人机数量，修改后需同步更新 `TAKEOFF_POINTS` |
| `TAKEOFF_POINTS` | 见 `scenes/default_scene.json` | 各机起飞点坐标，决定 BSAE 分配结果 |
| `WEIGHT_CLIMB` | 3.0 | TSP 爬升代价权重，↑ → 路径高度更平稳 |
| `WEIGHT_TURN` | 2.0 | TSP 偏航代价权重，↑ → 减少大角度转弯 |

### 轨迹参数（影响阶段 5）

| 参数 | 默认值 | 增大效果 | 减小效果 |
|------|--------|---------|---------|
| `VOXEL_SIZE` | 0.5 m | 构建速度↑，路径精度↓ | 路径精度↑，构建时间↑³ |
| `FLIGHT_SAFE_RADIUS` | 2.0 m | 见安全参数 | 见安全参数 |
| `HOVER_TIME` | 5.0 s | 拍照时间更充裕 | 总任务时间↓ |
| `TRANSIT_SPEED` | 2.0 m/s | 任务时间↓，动态响应要求↑ | 飞行更平缓 |
| `CSV_FREQ` | 10.0 Hz | setpoint 更密集，平滑↑ | — |

### 阶段间调参指引

```
修改成像/安全参数 → reset(2)，重跑 stage2→5
修改覆盖阈值     → stage3(quality_threshold=x)，重跑 stage4→5
修改多机参数     → reset(4)，重跑 stage4→5
修改轨迹参数     → 直接重跑 stage5()
```
