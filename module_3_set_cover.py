# module_3_set_cover.py
import time
import heapq
import numpy as np

class QualityAwareSetCover:
    def __init__(self, coverage_dict, quality_threshold_ratio=0.85):
        """
        :param coverage_dict: {视点ID: {三角面元ID: 质量分Q, ...}}
        :param quality_threshold_ratio: 达标阈值比例 (如 0.85 表示需达到该面元理论最高分的 85%)
        """
        self.coverage_dict = coverage_dict
        self.ratio = quality_threshold_ratio
        
        self.all_vps = list(coverage_dict.keys())
        
        print("\n[Module 3] 正在解析质量矩阵与计算物理达标阈值...")
        self.max_q_dict = {}
        for vp, tri_scores in self.coverage_dict.items():
            for tri_id, q_score in tri_scores.items():
                if tri_id not in self.max_q_dict or q_score > self.max_q_dict[tri_id]:
                    self.max_q_dict[tri_id] = q_score
                    
        self.all_targets_list = list(self.max_q_dict.keys())
        self.num_targets = len(self.all_targets_list)
        self.num_vps = len(self.all_vps)
        
        # 建立面元 ID 到 连续整数索引的映射，这是极速 Numpy 计算的关键
        self.tri_to_idx = {tri_id: idx for idx, tri_id in enumerate(self.all_targets_list)}
        
        # 将 thresholds 转为高速 Numpy 数组
        self.thresholds_arr = np.zeros(self.num_targets)
        for tri_id, max_q in self.max_q_dict.items():
            self.thresholds_arr[self.tri_to_idx[tri_id]] = max_q * self.ratio
            
        # 将原始字典转化为 Numpy 高速索引与分值数组
        self.vp_data = {}
        for vp in self.all_vps:
            tri_ids = list(self.coverage_dict[vp].keys())
            indices = np.array([self.tri_to_idx[t] for t in tri_ids], dtype=np.int32)
            scores = np.array(list(self.coverage_dict[vp].values()), dtype=np.float32)
            self.vp_data[vp] = (indices, scores)
            
        print(f" -> 矩阵解析完成！待优选候选视点: {self.num_vps} 个 | 待覆盖三角面元: {self.num_targets} 个")

    def optimize(self):
        print("\n" + "="*60)
        print(f" 🗜️ 启动模块 3: 极速张量化 Lazy Greedy 次模边缘贪心优化")
        print("="*60)
        start_t = time.time()
        
        current_scores = np.zeros(self.num_targets, dtype=np.float32)
        selected_vps = []
        
        # 构建优先队列 (Min-Heap, 我们存入负数以模拟 Max-Heap)
        # 数据结构: (负边际收益, 上次更新的迭代轮次, 视点ID)
        pq = []
        print("  -> 正在初始化张量优先堆...")
        for vp in self.all_vps:
            indices, scores = self.vp_data[vp]
            gain = np.sum(np.minimum(scores, self.thresholds_arr[indices]))
            if gain > 1e-6:
                heapq.heappush(pq, (-gain, 0, vp))
                
        iteration = 0
        while pq:
            neg_gain, last_update_iter, vp = heapq.heappop(pq)
            
            # 堆顶有效性校验：如果成绩是最新的，直接录取
            if last_update_iter == iteration:
                if -neg_gain < 1e-6:
                    break # 所有面元都已达标
                
                selected_vps.append(vp)
                indices, scores = self.vp_data[vp]
                current_scores[indices] += scores
                iteration += 1
                continue
                
            # 成绩过期，重新按最新战况结算边际收益
            indices, scores = self.vp_data[vp]
            deficits = self.thresholds_arr[indices] - current_scores[indices]
            
            valid_mask = deficits > 0
            if not np.any(valid_mask):
                continue
                
            new_gain = np.sum(np.minimum(scores[valid_mask], deficits[valid_mask]))
            
            if new_gain > 1e-6:
                heapq.heappush(pq, (-new_gain, iteration, vp))

        print("\n" + "="*60)
        print(f" 🏁 质量感知航点精简完成 | 最终胜出航点数: {len(selected_vps)} 个")
        print(f" 💡 压缩率: 选出了全集 {len(selected_vps)/self.num_vps*100:.2f}% 的精英视点")
        print(f" ⏱️ 核心算法耗时: {time.time() - start_t:.2f} 秒")
        print("="*60)
        
        return selected_vps