# benchmark_wind_sensitivity.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper


# -------------------------------------------------------------
# 테스트 시나리오 설정
# -------------------------------------------------------------
WIND_SPEEDS = [2.0, 5.0, 8.0]      # 단계별 계단형 돌풍 세기 (m/s)
NUM_EPISODES = 25                  # 풍속당 통계 검증 반복 횟수
GUST_START = 3.0
GUST_DURATION = 0.8


def get_model_paths():
    vanilla_path = "./models/best_model/best_model.zip"
    if not os.path.exists(vanilla_path):
        vanilla_path = "./models/ppo_quadrotor_final.zip"

    dr_path = "./models/dr_best_model/best_model.zip"
    if not os.path.exists(dr_path):
        dr_path = "./models/dr_model/ppo_dr_final.zip"

    if not os.path.exists(vanilla_path):
        raise FileNotFoundError(f"Vanilla PPO 모델을 찾을 수 없습니다: {vanilla_path}")
    if not os.path.exists(dr_path):
        raise FileNotFoundError(f"DR-PPO 모델을 찾을 수 없습니다: {dr_path}")

    return vanilla_path, dr_path


def evaluate_at_wind_speed(model, wind_speed, num_episodes=25):
    """지정된 돌풍 속도에서 모델의 통계적 성능 측정"""
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)

    # 전역 돌풍 벡터를 현재 풍속(X축 방향)으로 설정
    config.STEP_GUST_MAGNITUDE = np.array([wind_speed, 0.0, 0.0], dtype=np.float32)

    rmses = []
    max_disps = []
    survivals = []
    sample_trajectory = None

    target = np.array([0.0, 0.0, 1.0])

    for ep in range(num_episodes):
        seed = 2000 + ep
        obs, _ = env.reset(seed=seed)

        ep_positions = []
        done = False
        step = 0

        while step < config.MAX_STEPS and not done:
            curr_time = step * config.DT
            action, _ = model.predict(obs, deterministic=True)

            ep_positions.append(raw_env.state["pos"].copy())
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated

        ep_positions = np.array(ep_positions)
        survived = (step >= config.MAX_STEPS)
        survivals.append(survived)

        # 위치 오차(RMSE) 계산
        pos_err = ep_positions - target
        rmse = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rmses.append(rmse)

        # 돌풍 인가 구간(t >= 3.0s) 최대 밀림 거리 계산
        times = np.arange(len(ep_positions)) * config.DT
        gust_idx = np.where(times >= GUST_START)[0]
        if len(gust_idx) > 0:
            max_disp = np.max(np.linalg.norm(pos_err[gust_idx], axis=1))
        else:
            max_disp = np.max(np.linalg.norm(pos_err, axis=1))
        max_disps.append(max_disp)

        if ep == 0:
            sample_trajectory = {
                "times": times,
                "positions": ep_positions,
                "survived": survived
            }

    return {
        "wind_speed": wind_speed,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "max_disp_mean": np.mean(max_disps),
        "max_disp_std": np.std(max_disps),
        "survival_rate": np.mean(survivals) * 100.0,
        "sample": sample_trajectory
    }


def plot_sensitivity_results(vanilla_stats, dr_stats, high_wind_speed=8.0):
    speeds = [s["wind_speed"] for s in vanilla_stats]
    
    v_rmse = [s["rmse_mean"] for s in vanilla_stats]
    v_rmse_std = [s["rmse_std"] for s in vanilla_stats]
    d_rmse = [s["rmse_mean"] for s in dr_stats]
    d_rmse_std = [s["rmse_std"] for s in dr_stats]

    v_disp = [s["max_disp_mean"] for s in vanilla_stats]
    v_disp_std = [s["max_disp_std"] for s in vanilla_stats]
    d_disp = [s["max_disp_mean"] for s in dr_stats]
    d_disp_std = [s["max_disp_std"] for s in dr_stats]

    v_surv = [s["survival_rate"] for s in vanilla_stats]
    d_surv = [s["survival_rate"] for s in dr_stats]

    fig = plt.figure(figsize=(15, 10))

    # 1. 풍속별 위치 RMSE 추이 (오차 막대 포함)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(speeds, v_rmse, yerr=v_rmse_std, fmt='-o', color='crimson', label='Vanilla PPO', capsize=5, lw=2)
    ax1.errorbar(speeds, d_rmse, yerr=d_rmse_std, fmt='-s', color='royalblue', label='DR-PPO (Ours)', capsize=5, lw=2)
    ax1.set_title("Position Tracking Error (RMSE) vs Gust Speed", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Gust Wind Speed (m/s)")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_xticks(speeds)
    ax1.grid(True)
    ax1.legend(loc="upper left")

    # 2. 풍속별 최대 밀림 거리 추이
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(speeds, v_disp, yerr=v_disp_std, fmt='-o', color='crimson', label='Vanilla PPO', capsize=5, lw=2)
    ax2.errorbar(speeds, d_disp, yerr=d_disp_std, fmt='-s', color='royalblue', label='DR-PPO (Ours)', capsize=5, lw=2)
    ax2.set_title("Max Displacement vs Gust Speed", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Gust Wind Speed (m/s)")
    ax2.set_ylabel("Max Displacement (m)")
    ax2.set_xticks(speeds)
    ax2.grid(True)
    ax2.legend(loc="upper left")

    # 3. 풍속별 비행 생존율 (막대 그래프)
    ax3 = fig.add_subplot(2, 2, 3)
    bar_width = 0.35
    x_indices = np.arange(len(speeds))
    ax3.bar(x_indices - bar_width/2, v_surv, width=bar_width, color='crimson', alpha=0.8, label='Vanilla PPO')
    ax3.bar(x_indices + bar_width/2, d_surv, width=bar_width, color='royalblue', alpha=0.8, label='DR-PPO (Ours)')
    ax3.set_title("Flight Survival Rate vs Gust Speed", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Gust Wind Speed (m/s)")
    ax3.set_ylabel("Survival Rate (%)")
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f"{s} m/s" for s in speeds])
    ax3.set_ylim([0, 110])
    ax3.grid(True, axis='y')
    ax3.legend(loc="lower left")

    # 4. 극한 풍속(8 m/s)에서의 X축 밀림 및 복원 응답 시계열 비교
    ax4 = fig.add_subplot(2, 2, 4)
    v_high = [s for s in vanilla_stats if s["wind_speed"] == high_wind_speed][0]["sample"]
    d_high = [s for s in dr_stats if s["wind_speed"] == high_wind_speed][0]["sample"]

    ax4.plot(v_high["times"], v_high["positions"][:, 0], label=f'Vanilla X (Surv: {v_high["survived"]})', color='crimson', linestyle='--', lw=1.8)
    ax4.plot(d_high["times"], d_high["positions"][:, 0], label=f'DR-PPO X (Surv: {d_high["survived"]})', color='royalblue', lw=2.0)
    ax4.axhline(0.0, color='black', linestyle=':')
    ax4.axvspan(GUST_START, GUST_START + GUST_DURATION, color='gray', alpha=0.25, label=f'Step Gust {high_wind_speed} m/s')
    ax4.set_title(f"Dynamic Response at Strong Gust ({high_wind_speed} m/s)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("X Position (m)")
    ax4.grid(True)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/wind_sensitivity_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] 풍속 감도 분석 플롯이 저장되었습니다: {save_path}")
    plt.show()


def main():
    vanilla_path, dr_path = get_model_paths()
    print("=" * 70)
    print("풍속 감도 벤치마크 평가 시작: [2.0, 5.0, 8.0] m/s")
    print(f"풍속당 평가 횟수: {NUM_EPISODES}회")
    print("=" * 70)

    vanilla_model = PPO.load(vanilla_path)
    dr_model = PPO.load(dr_path)

    vanilla_stats = []
    dr_stats = []

    for w in WIND_SPEEDS:
        print(f"\n>> 풍속 {w:.1f} m/s 테스트 수행 중...")
        v_res = evaluate_at_wind_speed(vanilla_model, wind_speed=w, num_episodes=NUM_EPISODES)
        d_res = evaluate_at_wind_speed(dr_model, wind_speed=w, num_episodes=NUM_EPISODES)
        vanilla_stats.append(v_res)
        dr_stats.append(d_res)

    # -------------------------------------------------------------
    # 콘솔 요약 테이블 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'풍속':<8} | {'모델':<16} | {'위치 RMSE (m)':<20} | {'최대 밀림 (m)':<18} | {'생존율 (%)':<10}")
    print("-" * 78)
    for v, d in zip(vanilla_stats, dr_stats):
        w_str = f"{v['wind_speed']} m/s"
        print(f"{w_str:<8} | {'Vanilla PPO':<16} | {v['rmse_mean']:.4f} ± {v['rmse_std']:.4f}     | {v['max_disp_mean']:.4f} ± {v['max_disp_std']:.4f}  | {v['survival_rate']:>6.1f} %")
        print(f"{'':<8} | {'DR-PPO (Ours)':<16} | {d['rmse_mean']:.4f} ± {d['rmse_std']:.4f}     | {d['max_disp_mean']:.4f} ± {d['max_disp_std']:.4f}  | {d['survival_rate']:>6.1f} %")
        print("-" * 78)
    print("=" * 78)

    plot_sensitivity_results(vanilla_stats, dr_stats, high_wind_speed=8.0)


if __name__ == "__main__":
    main()