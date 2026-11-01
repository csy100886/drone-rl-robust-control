# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper


def find_model_path():
    """저장된 최적 모델 또는 최종 모델 경로 탐색"""
    best_path = "./models/best_model/best_model.zip"
    final_path = "./models/ppo_quadrotor_final.zip"
    
    if os.path.exists(best_path):
        print(f"[Info] 최고 성능 모델을 로드합니다: {best_path}")
        return best_path
    elif os.path.exists(final_path):
        print(f"[Info] 최종 학습 모델을 로드합니다: {final_path}")
        return final_path
    else:
        # checkpoints 내 최신 파일 탐색
        ckpt_dir = "./models/checkpoints"
        if os.path.exists(ckpt_dir):
            ckpts = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getctime)
                print(f"[Info] 체크포인트 모델을 로드합니다: {latest_ckpt}")
                return latest_ckpt
        raise FileNotFoundError("학습된 가중치 파일(.zip)을 찾을 수 없습니다. 학습을 먼저 진행하세요.")


def run_evaluation(model_path, render=False):
    # 환경 초기화
    raw_env = QuadrotorEnv()
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    
    # 모델 로드 (결정론적 추론 deterministic=True)
    model = PPO.load(model_path)
    
    obs, _ = env.reset(seed=100)
    
    # 데이터 로깅용 리스트
    times = []
    positions = []
    velocities = []
    euler_angles = []  # Roll, Pitch, Yaw (deg)
    actions = []
    winds = []
    
    step = 0
    done = False
    
    print("=" * 60)
    print("평가 비행 시뮬레이션 시작 (돌풍 인가: t = 3.0s ~ 3.8s)")
    print("=" * 60)

    while step < config.MAX_STEPS and not done:
        curr_time = step * config.DT
        
        # PPO 모델 행동 추론
        action, _ = model.predict(obs, deterministic=True)
        
        # 기체 내부 상태 기록 (raw_env 접근)
        pos = raw_env.state["pos"].copy()
        vel = raw_env.state["vel"].copy()
        rot_mat = raw_env.state["rot"].copy()
        rpy = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
        
        # 외란 기록
        wind = raw_env.disturbance.step_turbulence() + raw_env.disturbance.get_step_gust(curr_time)
        
        # 기록
        times.append(curr_time)
        positions.append(pos)
        velocities.append(vel)
        euler_angles.append(rpy)
        actions.append(action.copy())
        winds.append(wind)
        
        # 환경 스텝 진행
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        done = terminated or truncated

    times = np.array(times)
    positions = np.array(positions)
    euler_angles = np.array(euler_angles)
    actions = np.array(actions)
    winds = np.array(winds)

    # -------------------------------------------------------------
    # 수치적 평가 지표(Metric) 계산
    # -------------------------------------------------------------
    target = np.array([0.0, 0.0, 1.0])
    pos_errors = positions - target
    rmse_pos = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
    
    # 돌풍 구간(3.0s 이후) 최대 밀림 거리
    gust_mask = times >= 3.0
    if np.any(gust_mask):
        max_displacement = np.max(np.linalg.norm(pos_errors[gust_mask], axis=1))
    else:
        max_displacement = np.max(np.linalg.norm(pos_errors, axis=1))

    # 허용 오차(5cm = 0.05m) 복구 시간(Settling Time) 계산
    settling_time = None
    gust_end_idx = np.where(times >= 3.8)[0]
    if len(gust_end_idx) > 0:
        post_gust_indices = gust_end_idx[0]
        within_tolerance = np.linalg.norm(pos_errors[post_gust_indices:], axis=1) <= 0.05
        # 연속으로 오차 범위 내에 머무른 시점 탐색
        for i in range(len(within_tolerance)):
            if np.all(within_tolerance[i:i+25]):  # 0.5초(25스텝) 이상 안정 유지
                settling_time = times[post_gust_indices + i] - 3.8
                break

    # 제어 평활도 (Jitter metric)
    act_diff = np.diff(actions, axis=0)
    jitter = np.mean(np.sum(act_diff**2, axis=1))

    print("\n[정량적 성능 지표 (Quantitative Results)]")
    print("-" * 50)
    print(f"1. 위치 추적 RMSE         : {rmse_pos:.4f} m")
    print(f"2. 돌풍 후 최대 밀림 거리  : {max_displacement:.4f} m")
    print(f"3. 허용 오차(5cm) 정착 시간: {settling_time:.2f} s" if settling_time else "3. 정착 시간: 측정 구간 내 미수렴")
    print(f"4. 제어 입력 평활도(Jitter): {jitter:.5f}")
    print("-" * 50)

    # -------------------------------------------------------------
    # 시각화 플롯 생성 및 저장
    # -------------------------------------------------------------
    os.makedirs("./results", exist_ok=True)
    plot_results(times, positions, euler_angles, actions, winds, target)


def plot_results(times, positions, euler_angles, actions, winds, target):
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D 궤적 플롯
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='PPO Trajectory', color='royalblue', lw=2)
    ax1.scatter([target[0]], [target[1]], [target[2]], color='crimson', s=60, marker='*', label='Target (0, 0, 1)')
    ax1.scatter([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], color='green', s=30, label='Start')
    ax1.set_title("3D Flight Trajectory under Gust Disturbance", fontsize=11, fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.legend(loc='best')
    ax1.grid(True)

    # 2. 시간에 따른 축별 위치 오차 (X, Y, Z)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, positions[:, 0], label='X', color='darkorange')
    ax2.plot(times, positions[:, 1], label='Y', color='forestgreen')
    ax2.plot(times, positions[:, 2], label='Z', color='royalblue')
    ax2.axhline(target[2], color='red', linestyle='--', alpha=0.6, label='Target Altitude')
    ax2.axvspan(3.0, 3.8, color='gray', alpha=0.25, label='Step Gust (3.0s-3.8s)')
    ax2.set_title("Position Response vs Time", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.legend(loc='best')
    ax2.grid(True)

    # 3. 시간에 따른 자세각 (Roll, Pitch, Yaw)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, euler_angles[:, 0], label='Roll ($\phi$)', color='crimson')
    ax3.plot(times, euler_angles[:, 1], label='Pitch ($\\theta$)', color='darkcyan')
    ax3.plot(times, euler_angles[:, 2], label='Yaw ($\psi$)', color='purple', alpha=0.7)
    ax3.axvspan(3.0, 3.8, color='gray', alpha=0.25)
    ax3.set_title("Attitude Angles (RPY) Recovery", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (deg)")
    ax3.legend(loc='best')
    ax3.grid(True)

    # 4. 모터별 정규화 제어 명령 [-1, 1]
    ax4 = fig.add_subplot(2, 2, 4)
    for m in range(4):
        ax4.plot(times, actions[:, m], label=f'Motor {m+1}', alpha=0.8)
    ax4.axvspan(3.0, 3.8, color='gray', alpha=0.25)
    ax4.set_title("Normalized Motor Control Inputs", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Action [-1, 1]")
    ax4.legend(loc='best', ncol=2)
    ax4.grid(True)

    plt.tight_layout()
    save_path = "./results/evaluation_plots.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] 평가 그래프가 저장되었습니다: {save_path}")
    plt.show()


if __name__ == "__main__":
    model_file = find_model_path()
    run_evaluation(model_file)