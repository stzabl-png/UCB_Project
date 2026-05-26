#!/usr/bin/env python3
"""
model/grasp_scorer.py — Diffusion 抓取评分系统

根据仿真场景布局和成功抓取统计分布，对多个候选抓取 pose 评分排序。

评分维度（总分 100）：
  ① 方向可行性 (40%): +Y/-Z 优先，-Y/+Z 直接淘汰
  ② 中轴距离   (25%): 指尖中点到物体中轴越近越好
  ③ 接触高度   (20%): 物体中上部偏好
  ④ 宽度合理性 (15%): 2~8cm 范围

用法：
    from model.grasp_scorer import GraspScorer
    scorer = GraspScorer()
    best = scorer.select_best(candidates, pc)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────────
# 仿真场景常量（来自 sim/run_grasp_sim.py）
# ─────────────────────────────────────────────────────────────
ROBOT_POSITION = np.array([0.2, -0.05, 0.80])
OBJECT_POSITION = np.array([0.0, 0.55, 0.80])
TABLE_TOP_Z = 0.80

# Franka 夹爪参数
MIN_WIDTH = 0.010   # 1cm
MAX_WIDTH = 0.085   # 8.5cm
TCP_OFFSET = 0.105  # m


@dataclass
class ScoredGrasp:
    """评分后的抓取候选"""
    index: int                      # 原始候选索引
    total_score: float              # 总分 0~100
    direction_score: float          # 方向得分
    centroid_score: float           # 中轴距离得分
    height_score: float             # 高度得分
    width_score: float              # 宽度得分
    approach: np.ndarray            # 接近方向 (3,)
    finger_mid: np.ndarray          # 指尖中点 (3,)
    width: float                    # 夹爪宽度 (m)
    L: Optional[np.ndarray] = None  # 左指尖 (3,)
    R: Optional[np.ndarray] = None  # 右指尖 (3,)
    rejected: bool = False          # 是否被硬过滤淘汰
    reject_reason: str = ""


class GraspScorer:
    """
    对 Diffusion 预测的多个候选抓取进行评分排序。

    所有输入坐标在物体坐标系（OBJ local frame）。
    假设物体竖直放置，物体坐标系 Z ≈ 世界 Z。
    """

    def __init__(self,
                 w_direction: float = 40.0,
                 w_centroid: float = 25.0,
                 w_height: float = 20.0,
                 w_width: float = 15.0,
                 centroid_sigma: float = 0.01,    # 1cm
                 height_best_range: tuple = (0.0, 0.08),  # 0~8cm
                 ):
        self.w_direction = w_direction
        self.w_centroid = w_centroid
        self.w_height = w_height
        self.w_width = w_width
        self.centroid_sigma = centroid_sigma
        self.height_best = height_best_range

    # ── 方向评分 (40%) ─────────────────────────────────────

    def _score_direction(self, approach: np.ndarray) -> float:
        """
        approach: (3,) 归一化接近方向（物体坐标系）

        +Y (从前方): 最佳
        -Z (从上方): 次佳
        ±X (侧面):  可行
        -Y (从后方): 禁止
        +Z (从下方): 禁止
        """
        ay, az = approach[1], approach[2]

        # 硬过滤
        if az > 0.5:     # 从下方（+Z）
            return -1.0  # 标记淘汰
        if ay < -0.5:    # 从后方（-Y）
            return -1.0  # 标记淘汰

        # 软评分：偏好 +Y 和 -Z
        s_y = max(ay, 0.0)              # +Y 分量越大越好
        s_z = max(-az, 0.0)             # -Z 分量越大越好（上方接近）
        score = s_y * 0.6 + s_z * 0.4   # 加权

        return min(score, 1.0)

    # ── 中轴距离评分 (25%) ─────────────────────────────────

    def _score_centroid(self, finger_mid: np.ndarray,
                        centroid: np.ndarray) -> float:
        """指尖中点到物体中轴（Z轴）的水平距离越近越好"""
        rel = finger_mid - centroid
        d_xy = np.sqrt(rel[0]**2 + rel[1]**2)
        return float(np.exp(-d_xy / self.centroid_sigma))

    # ── 高度评分 (20%) ─────────────────────────────────────

    def _score_height(self, finger_mid: np.ndarray,
                      centroid: np.ndarray,
                      pc: np.ndarray) -> float:
        """物体中上部偏好，基于物体高度归一化"""
        z_rel = finger_mid[2] - centroid[2]
        z_range = pc[:, 2].max() - pc[:, 2].min()
        if z_range < 1e-6:
            return 0.5

        # 归一化到 [0, 1]，0=底部，1=顶部
        z_norm = (finger_mid[2] - pc[:, 2].min()) / z_range

        # 中上部 (30%~90%) 最佳
        if 0.3 <= z_norm <= 0.9:
            return 1.0
        elif 0.1 <= z_norm < 0.3:
            return 0.7
        elif 0.9 < z_norm <= 1.0:
            return 0.6
        else:
            return 0.3

    # ── 宽度评分 (15%) ─────────────────────────────────────

    def _score_width(self, width: float) -> float:
        """夹爪宽度合理性"""
        if width < MIN_WIDTH or width > MAX_WIDTH:
            return 0.0
        if 0.02 <= width <= 0.08:
            return 1.0
        return 0.5

    # ── 综合评分 ───────────────────────────────────────────

    def score_one(self, approach: np.ndarray,
                  finger_mid: np.ndarray,
                  width: float,
                  pc: np.ndarray,
                  L: np.ndarray = None,
                  R: np.ndarray = None,
                  index: int = 0) -> ScoredGrasp:
        """对单个候选评分"""
        centroid = pc.mean(0)

        s_dir = self._score_direction(approach)
        if s_dir < 0:
            return ScoredGrasp(
                index=index, total_score=0.0,
                direction_score=0, centroid_score=0,
                height_score=0, width_score=0,
                approach=approach, finger_mid=finger_mid,
                width=width, L=L, R=R,
                rejected=True,
                reject_reason='+Z approach' if approach[2] > 0.5 else '-Y approach'
            )

        s_ctr = self._score_centroid(finger_mid, centroid)
        s_hgt = self._score_height(finger_mid, centroid, pc)
        s_wid = self._score_width(width)

        total = (s_dir * self.w_direction +
                 s_ctr * self.w_centroid +
                 s_hgt * self.w_height +
                 s_wid * self.w_width)

        return ScoredGrasp(
            index=index, total_score=total,
            direction_score=s_dir * self.w_direction,
            centroid_score=s_ctr * self.w_centroid,
            height_score=s_hgt * self.w_height,
            width_score=s_wid * self.w_width,
            approach=approach, finger_mid=finger_mid,
            width=width, L=L, R=R,
        )

    # ── 批量评分 + 排序 ───────────────────────────────────

    def rank(self, candidates: list, pc: np.ndarray) -> List[ScoredGrasp]:
        """
        对多个候选评分并排序。

        candidates: list of dict, 每个包含:
            - 'approach':   (3,) 接近方向
            - 'finger_mid': (3,) 指尖中点
            - 'width':      float 夹爪宽度 (m)
            - 'L':          (3,) 可选，左指尖
            - 'R':          (3,) 可选，右指尖
        pc: (N, 3) 物体点云

        Returns: 按分数降序排列的 ScoredGrasp 列表
        """
        scored = []
        for i, c in enumerate(candidates):
            sg = self.score_one(
                approach=c['approach'],
                finger_mid=c['finger_mid'],
                width=c.get('width', 0.05),
                pc=pc,
                L=c.get('L'), R=c.get('R'),
                index=i,
            )
            scored.append(sg)

        # 按分数降序，rejected 的排最后
        scored.sort(key=lambda s: (-int(not s.rejected), -s.total_score))
        return scored

    def select_best(self, candidates: list, pc: np.ndarray) -> Optional[ScoredGrasp]:
        """返回最佳候选，如果全部被淘汰则返回 None"""
        ranked = self.rank(candidates, pc)
        if not ranked or ranked[0].rejected:
            return None
        return ranked[0]

    # ── 打印报告 ──────────────────────────────────────────

    @staticmethod
    def print_report(ranked: List[ScoredGrasp], top_n: int = 10):
        """打印评分报告"""
        print(f'\n{"Rank":<5} {"Score":<8} {"Dir":<7} {"Ctr":<7} '
              f'{"Hgt":<7} {"Wid":<7} {"Width":<8} {"Status":<10}')
        print('-' * 65)
        for i, sg in enumerate(ranked[:top_n]):
            status = f'❌ {sg.reject_reason}' if sg.rejected else '✅'
            print(f'{i+1:<5} {sg.total_score:<8.1f} '
                  f'{sg.direction_score:<7.1f} {sg.centroid_score:<7.1f} '
                  f'{sg.height_score:<7.1f} {sg.width_score:<7.1f} '
                  f'{sg.width*100:<8.2f} {status}')

        n_ok = sum(1 for s in ranked if not s.rejected)
        n_rej = sum(1 for s in ranked if s.rejected)
        print(f'\n  有效: {n_ok}  淘汰: {n_rej}  '
              f'最高分: {ranked[0].total_score:.1f}' if ranked else '')


# ─────────────────────────────────────────────────────────────
# 独立测试
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 模拟测试
    np.random.seed(42)
    pc = np.random.randn(4096, 3) * 0.03
    pc[:, 2] += 0.04   # 物体中心偏上

    candidates = [
        # 好的：从前方接近，中间高度
        {'approach': np.array([0, 0.9, -0.4]),
         'finger_mid': np.array([0.001, -0.002, 0.03]),
         'width': 0.05},
        # 好的：从上方接近
        {'approach': np.array([0.1, 0.2, -0.97]),
         'finger_mid': np.array([0.0, 0.0, 0.05]),
         'width': 0.04},
        # 差的：从下方
        {'approach': np.array([0, 0, 1.0]),
         'finger_mid': np.array([0.0, 0.0, 0.01]),
         'width': 0.05},
        # 差的：太偏
        {'approach': np.array([0, 0.8, -0.3]),
         'finger_mid': np.array([0.05, 0.04, 0.03]),
         'width': 0.05},
    ]

    scorer = GraspScorer()
    ranked = scorer.rank(candidates, pc)
    GraspScorer.print_report(ranked)

    best = scorer.select_best(candidates, pc)
    if best:
        print(f'\n  最佳候选: #{best.index}  分数={best.total_score:.1f}')
