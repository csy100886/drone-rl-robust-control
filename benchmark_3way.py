# benchmark_3way.py
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper
from baselines.pid_controller import CascadedPIDController


WIND_SPEEDS = [2.0, 5.0, 8.0]      # 단계별 풍속 외란
NUM_EPISODES = 20                  # 풍속당 반복 평가 횟수
GUST_START = 3.0
GUST_DURATION = 0.8


def get_rl_models():
    v_path = "./models/best_model/best_model.zip"
    if not os.path.exists(v_path):
        v_path = "./models/ppo_quadrotor_final.zip"

    dr_path = "./models/dr_best_model/best_model.zip"
    if not os.path.exists(dr_path):
        dr_path = "./models/dr_model/ppo_dr_final.zip"

    if not os.path.exists(v_path) or not os.path.exists(dr_path):
        raise FileNotFoundError("학습된 PPO 가중치(.zip)가 존재하지 않습니다. train.py 및 train_dr.py를 먼저 실행하세요.")

    return PPO.load(v_path), PPO.load(dr_path)


def evaluate_controller(controller_type, model_obj, wind_speed, num_episodes=20):
    """
    controller_type: 'pid', 'vanilla', 'dr'
    """
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    config.STEP_GUST_MAGNITUDE = np.array([wind_speed, 0.0, 0.0], dtype=np.float32)

    pid = CascadedPIDController(dt=config.DT) if controller_type == 'pid' else None

    rmses = []
    max_disps = []
    survivals = []
    sample_trajectory = None

    target = np.array([0.0, 0.0, 1.0])

    for ep in range(num_episodes):
        seed = 3000 + ep
        obs, _ = env.reset(seed=seed)
        if pid:
            pid.reset()

        ep_positions = []
        done = False
        step = 0

        while step < config.MAX_STEPS and not done:
            if controller_type == 'pid':
                action = pid.compute_action(raw_env.state, target_pos=target)
            else:
                action, _ = model_obj.predict(obs, deterministic=True)

            ep_positions.append(raw_env.state["pos"].copy())
            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated

        ep_positions = np.array(ep_positions)
        survived = (step >= config.MAX_STEPS)
        survivals.append(survived)

        pos_err = ep_positions - target
        rmse = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rmses.append(rmse)

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
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "max_disp_mean": np.mean(max_disps),
        "max_disp_std": np.std(max_disps),
        "survival_rate": np.mean(survivals) * 100.0,
        "sample": sample_trajectory
    }


def plot_3way_results(all_results):
    speeds = WIND_SPEEDS
    x = np.arange(len(speeds))
    width = 0.25

    fig = plt.figure(figsize=(15, 10))

    # 1. 위치 RMSE 비교 플롯
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(speeds, [all_results['pid'][w]['rmse_mean'] for w in speeds], 'o--', color='forestgreen', label='Cascaded PID', lw=2)
    ax1.plot(speeds, [all_results['vanilla'][w]['rmse_mean'] for w in speeds], 's--', color='crimson', label='Vanilla PPO', lw=2)
    ax1.plot(speeds, [all_results['dr'][w]['rmse_mean'] for w in speeds], '^-', color='royalblue', label='DR-PPO (Ours)', lw=2.2)
    ax1.set_title("Position Tracking Error (RMSE)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Gust Wind Speed (m/s)")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_xticks(speeds)
    ax1.grid(True)
    ax1.legend()

    # 2. 돌풍 후 최대 밀림 거리 비교 플롯
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(speeds, [all_results['pid'][w]['max_disp_mean'] for w in speeds], 'o--', color='forestgreen', label='Cascaded PID', lw=2)
    ax2.plot(speeds, [all_results['vanilla'][w]['max_disp_mean'] for w in speeds], 's--', color='crimson', label='Vanilla PPO', lw=2)
    ax2.plot(speeds, [all_results['dr'][w]['max_disp_mean'] for w in speeds], '^-', color='royalblue', label='DR-PPO (Ours)', lw=2.2)
    ax2.set_title("Max Displacement under Wind Gust", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Gust Wind Speed (m/s)")
    ax2.set_ylabel("Max Displacement (m)")
    ax2.set_xticks(speeds)
    ax2.grid(True)
    ax2.legend()

    # 3. 비행 생존율 (막대 그래프)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(x - width, [all_results['pid'][w]['survival_rate'] for w in speeds], width, label='Cascaded PID', color='forestgreen', alpha=0.85)
    ax3.bar(x, [all_results['vanilla'][w]['survival_rate'] for w in speeds], width, label='Vanilla PPO', color='crimson', alpha=0.85)
    ax3.bar(x + width, [all_results['dr'][w]['survival_rate'] for w in speeds], width, label='DR-PPO (Ours)', color='royalblue', alpha=0.85)
    ax3.set_title("Flight Survival Rate vs Wind Speed", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Gust Wind Speed (m/s)")
    ax3.set_ylabel("Survival Rate (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{w} m/s" for w in speeds])
    ax3.set_ylim([0, 115])
    ax3.grid(True, axis='y')
    ax3.legend()

    # 4. 강풍(8 m/s) 타격 시 X축 실시간 복원 궤적 비교
    ax4 = fig.add_subplot(2, 2, 4)
    pid_sample = all_results['pid'][8.0]['sample']
    van_sample = all_results['vanilla'][8.0]['sample']
    dr_sample = all_results['dr'][8.0]['sample']

    ax4.plot(pid_sample["times"], pid_sample["positions"][:, 0], label=f"PID (Surv: {pid_sample['survived']})", color='forestgreen', linestyle=':', lw=1.8)
    ax4.plot(van_sample["times"], van_sample["positions"][:, 0], label=f"Vanilla PPO (Surv: {van_sample['survived']})", color='crimson', linestyle='--', lw=1.8)
    ax4.plot(dr_sample["times"], dr_sample["positions"][:, 0], label=f"DR-PPO (Surv: {dr_sample['survived']})", color='royalblue', lw=2.2)
    ax4.axhline(0.0, color='black', linestyle='-')
    ax4.axvspan(GUST_START, GUST_START + GUST_DURATION, color='gray', alpha=0.25, label='Gust 8.0 m/s')
    ax4.set_title("Transient Response under 8.0 m/s Gust", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("X Position (m)")
    ax4.grid(True)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/benchmark_3way_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] 3자 벤치마크 플롯 저장 완료: {save_path}")
    plt.show()


def main():
    vanilla_model, dr_model = get_rl_models()
    models = {
        'pid': None,
        'vanilla': vanilla_model,
        'dr': dr_model
    }

    all_results = {'pid': {}, 'vanilla': {}, 'dr': {}}

    print("=" * 80)
    print("3자 제어기 성능 벤치마크 시작 (Cascaded PID vs Vanilla PPO vs DR-PPO)")
    print("=" * 80)

    for w in WIND_SPEEDS:
        print(f"\n>> 외란 풍속 {w:.1f} m/s 벤치마크 진행 중...")
        for c_type in ['pid', 'vanilla', 'dr']:
            res = evaluate_controller(c_type, models[c_type], wind_speed=w, num_episodes=NUM_EPISODES)
            all_results[c_type][w] = res

    # ---------------------------------------------------------
    # 터미널 정량 비교 요약 테이블 출력
    # ---------------------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'풍속':<8} | {'제어기':<15} | {'위치 RMSE (m)':<20} | {'최대 밀림 (m)':<18} | {'생존율 (%)':<10}")
    print("-" * 84)
    for w in WIND_SPEEDS:
        p_res = all_results['pid'][w]
        v_res = all_results['vanilla'][w]
        d_res = all_results['dr'][w]
        w_str = f"{w} m/s"

        print(f"{w_str:<8} | {'Cascaded PID':<15} | {p_res['rmse_mean']:.4f} ± {p_res['rmse_std']:.4f}     | {p_res['max_disp_mean']:.4f} ± {p_res['max_disp_std']:.4f}  | {p_res['survival_rate']:>6.1f} %")
        print(f"{'':<8} | {'Vanilla PPO':<15} | {v_res['rmse_mean']:.4f} ± {v_res['rmse_std']:.4f}     | {v_res['max_disp_mean']:.4f} ± {v_res['max_disp_std']:.4f}  | {v_res['survival_rate']:>6.1f} %")
        print(f"{'':<8} | {'DR-PPO (Ours)':<15} | {d_res['rmse_mean']:.4f} ± {d_res['rmse_std']:.4f}     | {d_res['max_disp_mean']:.4f} ± {d_res['max_disp_std']:.4f}  | {d_res['survival_rate']:>6.1f} %")
        print("-" * 84)
    print("=" * 84)

    plot_3way_results(all_results)


if __name__ == "__main__":
    main()