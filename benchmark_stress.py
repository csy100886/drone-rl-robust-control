# benchmark_stress.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper


# -------------------------------------------------------------
# 극한 조건 파라미터 정의 (Stress Test Conditions)
# -------------------------------------------------------------
STRESS_MASS_FACTOR = 1.30        # 기준 질량 대비 +30% 증가
STRESS_MOTOR_TAU = 0.050         # 모터 응답 지연 50ms (기준 20ms의 2.5배)
STRESS_INERTIA_FACTOR = 1.30     # 관성 모멘트 +30% 증가
NUM_EVAL_EPISODES = 30           # 통계 검증용 반복 에피소드 수


def get_model_paths():
    vanilla_path = "./models/best_model/best_model.zip"
    if not os.path.exists(vanilla_path):
        vanilla_path = "./models/ppo_quadrotor_final.zip"

    dr_path = "./models/dr_best_model/best_model.zip"
    if not os.path.exists(dr_path):
        dr_path = "./models/dr_model/ppo_dr_final.zip"

    if not os.path.exists(vanilla_path):
        raise FileNotFoundError(f"Vanilla PPO 모델 가중치를 찾을 수 없습니다: {vanilla_path}")
    if not os.path.exists(dr_path):
        raise FileNotFoundError(f"DR-PPO 모델 가중치를 찾을 수 없습니다: {dr_path}")

    return vanilla_path, dr_path


def apply_stress_conditions(raw_env):
    """에피소드 리셋 직후 극한 물리 환경 파라미터 강제 주입"""
    raw_env.mass = config.MASS * STRESS_MASS_FACTOR
    raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32) * STRESS_INERTIA_FACTOR
    raw_env.motor_tau = STRESS_MOTOR_TAU
    raw_env.drag_coeff = 0.15


def evaluate_policy_under_stress(model, num_episodes=30):
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)

    rmses = []
    max_displacements = []
    jitters = []
    survival_flags = []
    
    sample_trajectory = None  # 시각화용 0번 시드 궤적

    target = np.array([0.0, 0.0, 1.0])

    for ep in range(num_episodes):
        seed = 1000 + ep
        obs, _ = env.reset(seed=seed)
        apply_stress_conditions(raw_env)  # 극한 조건 설정

        ep_positions = []
        ep_actions = []
        done = False
        step = 0

        while step < config.MAX_STEPS and not done:
            curr_time = step * config.DT
            action, _ = model.predict(obs, deterministic=True)

            ep_positions.append(raw_env.state["pos"].copy())
            ep_actions.append(action.copy())

            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated

        ep_positions = np.array(ep_positions)
        ep_actions = np.array(ep_actions)

        # 생존 여부: 10초(500스텝) 완주 판정
        survived = (step >= config.MAX_STEPS)
        survival_flags.append(survived)

        # 오차 메트릭
        pos_err = ep_positions - target
        rmse = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rmses.append(rmse)

        # 돌풍 인가 구간(t >= 3.0s) 최대 밀림 거리
        times = np.arange(len(ep_positions)) * config.DT
        gust_idx = np.where(times >= 3.0)[0]
        if len(gust_idx) > 0:
            max_disp = np.max(np.linalg.norm(pos_err[gust_idx], axis=1))
        else:
            max_disp = np.max(np.linalg.norm(pos_err, axis=1))
        max_displacements.append(max_disp)

        # 제어 명령 진동(Jitter)
        if len(ep_actions) > 1:
            act_diff = np.diff(ep_actions, axis=0)
            jitter = np.mean(np.sum(act_diff**2, axis=1))
        else:
            jitter = 0.0
        jitters.append(jitter)

        # 첫 번째 에피소드는 그래프 비교용 데이터로 보관
        if ep == 0:
            sample_trajectory = {
                "times": times,
                "positions": ep_positions,
                "actions": ep_actions,
                "survived": survived
            }

    metrics = {
        "survival_rate": np.mean(survival_flags) * 100.0,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "max_disp_mean": np.mean(max_displacements),
        "max_disp_std": np.std(max_displacements),
        "jitter_mean": np.mean(jitters),
        "jitter_std": np.std(jitters),
        "sample": sample_trajectory
    }
    return metrics


def plot_comparison(vanilla_data, dr_data):
    v_sample = vanilla_data["sample"]
    d_sample = dr_data["sample"]
    target_z = 1.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # 1. 고도(Z축) 응답 비교
    axes[0].plot(v_sample["times"], v_sample["positions"][:, 2], label="Vanilla PPO (Trained w/o DR)", color="crimson", lw=1.8, linestyle="--")
    axes[0].plot(d_sample["times"], d_sample["positions"][:, 2], label="DR-PPO (Trained w/ DR)", color="royalblue", lw=2.0)
    axes[0].axhline(target_z, color="black", linestyle=":", label="Target Altitude (1.0m)")
    axes[0].axvspan(3.0, 3.8, color="gray", alpha=0.25, label="Step Gust (4.0 m/s)")
    axes[0].set_ylabel("Altitude Z (m)", fontsize=11)
    axes[0].set_title(f"Stress Benchmark: Mass +30% ({config.MASS*STRESS_MASS_FACTOR*1000:.1f}g), Motor Lag 50ms", fontsize=12, fontweight="bold")
    axes[0].grid(True)
    axes[0].legend(loc="lower right")

    # 2. X축 밀림 및 복귀 응답 비교
    axes[1].plot(v_sample["times"], v_sample["positions"][:, 0], label="Vanilla PPO X", color="crimson", lw=1.8, linestyle="--")
    axes[1].plot(d_sample["times"], d_sample["positions"][:, 0], label="DR-PPO X", color="royalblue", lw=2.0)
    axes[1].axhline(0.0, color="black", linestyle=":")
    axes[1].axvspan(3.0, 3.8, color="gray", alpha=0.25)
    axes[1].set_ylabel("Position X (m)", fontsize=11)
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    # 3. 1번 모터 제어 입력 변화 비교 (Jittering 확인)
    axes[2].plot(v_sample["times"], v_sample["actions"][:, 0], label="Vanilla Motor 1 Cmd", color="crimson", lw=1.2, alpha=0.7)
    axes[2].plot(d_sample["times"], d_sample["actions"][:, 0], label="DR-PPO Motor 1 Cmd", color="royalblue", lw=1.2, alpha=0.85)
    axes[2].axvspan(3.0, 3.8, color="gray", alpha=0.25)
    axes[2].set_ylabel("Motor 1 Action [-1, 1]", fontsize=11)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].grid(True)
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    save_path = "./results/stress_benchmark_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] 벤치마크 비교 그래프가 저장되었습니다: {save_path}")
    plt.show()


def main():
    vanilla_path, dr_path = get_model_paths()
    print("=" * 70)
    print("극한 스트레스 환경 벤치마크 평가 시작")
    print(f"- 기체 질량: {config.MASS*1000:.1f}g -> {config.MASS*STRESS_MASS_FACTOR*1000:.1f}g (+30%)")
    print(f"- 모터 시상수: {config.MOTOR_TAU*1000:.1f}ms -> {STRESS_MOTOR_TAU*1000:.1f}ms (2.5배 지연)")
    print(f"- 평가 에피소드 수: {NUM_EVAL_EPISODES}회")
    print("=" * 70)

    print("\n[1/2] Vanilla PPO 모델 평가 진행 중...")
    vanilla_model = PPO.load(vanilla_path)
    vanilla_results = evaluate_policy_under_stress(vanilla_model, num_episodes=NUM_EVAL_EPISODES)

    print("[2/2] DR-PPO 모델 평가 진행 중...")
    dr_model = PPO.load(dr_path)
    dr_results = evaluate_policy_under_stress(dr_model, num_episodes=NUM_EVAL_EPISODES)

    # -------------------------------------------------------------
    # 작성용 결과 요약 테이블 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'평가 지표 (Stress Test)':<28} | {'Vanilla PPO':<18} | {'DR-PPO (제안 기법)':<18}")
    print("-" * 70)
    print(f"{'비행 생존율 (Survival Rate)':<28} | {vanilla_results['survival_rate']:>6.1f} %          | {dr_results['survival_rate']:>6.1f} %")
    print(f"{'위치 오차 (RMSE)':<28} | {vanilla_results['rmse_mean']:.4f} ± {vanilla_results['rmse_std']:.4f} m | {dr_results['rmse_mean']:.4f} ± {dr_results['rmse_std']:.4f} m")
    print(f"{'돌풍 최대 밀림 (Max Disp)':<28} | {vanilla_results['max_disp_mean']:.4f} ± {vanilla_results['max_disp_std']:.4f} m | {dr_results['max_disp_mean']:.4f} ± {dr_results['max_disp_std']:.4f} m")
    print(f"{'제어 입력 진동 (Jitter)':<28} | {vanilla_results['jitter_mean']:.5f} ± {vanilla_results['jitter_std']:.5f} | {dr_results['jitter_mean']:.5f} ± {dr_results['jitter_std']:.5f}")
    print("=" * 70)

    os.makedirs("./results", exist_ok=True)
    plot_comparison(vanilla_results, dr_results)


if __name__ == "__main__":
    main()