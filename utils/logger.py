# utils/logger.py
import os
import csv
import numpy as np


class PerformanceEvaluator:
    """
    외란 복원 및 호버링 제어 정량 평가지표 계산기
    """
    def __init__(self, target_pos=np.array([0.0, 0.0, 1.0]), tolerance=0.05, dt=0.02):
        self.target_pos = target_pos
        self.tolerance = tolerance  # 정착 판정 허용 오차 (5cm)
        self.dt = dt

    def compute_rmse(self, positions):
        """3차원 위치 추적 RMSE (m)"""
        pos_errors = positions - self.target_pos
        return float(np.sqrt(np.mean(np.sum(pos_errors**2, axis=1))))

    def compute_max_displacement(self, positions, times, gust_start_time=3.0):
        """돌풍 인가 이후 최대 밀림 거리 (m)"""
        pos_errors = positions - self.target_pos
        gust_idx = np.where(times >= gust_start_time)[0]
        if len(gust_idx) > 0:
            return float(np.max(np.linalg.norm(pos_errors[gust_idx], axis=1)))
        return float(np.max(np.linalg.norm(pos_errors, axis=1)))

    def compute_settling_time(self, positions, times, gust_end_time=3.8, window_steps=25):
        """돌풍 종료 후 오차 범위 내로 복귀하여 유지되는 정착 시간 (s)"""
        pos_errors = positions - self.target_pos
        post_idx = np.where(times >= gust_end_time)[0]
        if len(post_idx) == 0:
            return None

        start_i = post_idx[0]
        within_tol = np.linalg.norm(pos_errors[start_i:], axis=1) <= self.tolerance
        for i in range(len(within_tol)):
            if np.all(within_tol[i:i + window_steps]):
                return float(times[start_i + i] - gust_end_time)
        return None

    def compute_jitter(self, actions):
        """제어 입력 차분 제곱평균(Jittering)"""
        if len(actions) <= 1:
            return 0.0
        act_diff = np.diff(actions, axis=0)
        return float(np.mean(np.sum(act_diff**2, axis=1)))


def export_metrics_to_csv(filepath, rows, headers):
    """실험 통계 테이블을 CSV로 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[CSV 저장 완료] {filepath}")