# ablation_history.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper


K_CANDIDATES = [0, 2, 4]
EVAL_EPISODES = 30
GUST_SPEED = 6.0                   # 외란 세기 (6.0 m/s)
GUST_START = 3.0
GUST_DURATION = 0.8
MODELS_DIR = "./models/ablation"
RESULTS_DIR = "./results/ablation"


# -------------------------------------------------------------
# 1. K별 학습 환경 생성 및 학습 루프
# -------------------------------------------------------------
def make_env_for_k(k_len: int, rank: int, enable_dr: bool = True, seed: int = 100):
    def _init():
        raw_env = QuadrotorEnv(enable_dr=enable_dr)
        env = HistoryBufferWrapper(raw_env, history_length=k_len)
        log_dir = f"./logs/ablation_k{k_len}_env_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ablation_model(k_len: int, timesteps: int = 400_000):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"ppo_k{k_len}.zip")

    if os.path.exists(model_save_path):
        print(f"[Info] K={k_len} 모델이 이미 존재합니다: {model_save_path}")
        return

    print("=" * 60)
    print(f">> [학습 시작] 관찰 이력 크기 K={k_len} (관찰 차원: {18 * (k_len + 1)})")
    print("=" * 60)

    num_envs = 4
    train_env = DummyVecEnv([make_env_for_k(k_len, i, enable_dr=True) for i in range(num_envs)])

    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="auto"
    )

    model.learn(total_timesteps=timesteps)
    model.save(model_save_path)
    print(f"[학습 완료] K={k_len} 가중치 저장 완료: {model_save_path}")


# -------------------------------------------------------------
# 2. 통계적 성능 평가
# -------------------------------------------------------------
def evaluate_k_model(k_len: int, num_episodes: int = EVAL_EPISODES):
    model_path = os.path.join(MODELS_DIR, f"ppo_k{k_len}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    model = PPO.load(model_path)
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=k_len)
    config.STEP_GUST_MAGNITUDE = np.array([GUST_SPEED, 0.0, 0.0], dtype=np.float32)

    target = np.array([0.0, 0.0, 1.0])
    rmses = []
    max_disps = []
    settling_times = []
    jitters = []
    sample_trajectory = None

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        ep_positions = []
        ep_actions = []
        step = 0
        done = False

        while step < config.MAX_STEPS and not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_positions.append(raw_env.state["pos"].copy())
            ep_actions.append(action.copy())

            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated

        ep_positions = np.array(ep_positions)
        ep_actions = np.array(ep_actions)
        times = np.arange(len(ep_positions)) * config.DT

        # 1. 위치 RMSE
        pos_err = ep_positions - target
        rmse = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rmses.append(rmse)

        # 2. 돌풍 후 최대 밀림 거리
        gust_idx = np.where(times >= GUST_START)[0]
        if len(gust_idx) > 0:
            max_disp = np.max(np.linalg.norm(pos_err[gust_idx], axis=1))
        else:
            max_disp = np.max(np.linalg.norm(pos_err, axis=1))
        max_disps.append(max_disp)

        # 3. 정착 시간 (돌풍 종료 t=3.8s 이후 오차 5cm 이내 유지 시점)
        settling_time = 6.2  # 최대 시간(10초 - 3.8초) 기본값
        post_gust_idx = np.where(times >= (GUST_START + GUST_DURATION))[0]
        if len(post_gust_idx) > 0:
            start_i = post_gust_idx[0]
            within_tol = np.linalg.norm(pos_err[start_i:], axis=1) <= 0.05
            for i in range(len(within_tol)):
                if np.all(within_tol[i:i+25]):  # 0.5초(25스텝) 연속 유지
                    settling_time = times[start_i + i] - (GUST_START + GUST_DURATION)
                    break
        settling_times.append(settling_time)

        # 4. 제어 입력 진동 (Jitter)
        act_diff = np.diff(ep_actions, axis=0)
        jitter = np.mean(np.sum(act_diff**2, axis=1))
        jitters.append(jitter)

        if ep == 0:
            sample_trajectory = {
                "times": times,
                "positions": ep_positions,
                "actions": ep_actions
            }

    return {
        "k": k_len,
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "disp_mean": np.mean(max_disps),
        "disp_std": np.std(max_disps),
        "settling_mean": np.mean(settling_times),
        "settling_std": np.std(settling_times),
        "jitter_mean": np.mean(jitters),
        "jitter_std": np.std(jitters),
        "sample": sample_trajectory
    }


# -------------------------------------------------------------
# 3. 소거 실험 결과 플롯
# -------------------------------------------------------------
def plot_ablation_results(results_dict):
    fig = plt.figure(figsize=(15, 10))
    k_labels = [f"K={k} ({18*(k+1)}D)" for k in K_CANDIDATES]
    colors = ['gray', 'darkorange', 'royalblue']

    # 1. 6.0 m/s 돌풍 타격 시 X축 실시간 과도 응답 시계열
    ax1 = fig.add_subplot(2, 2, 1)
    for i, k in enumerate(K_CANDIDATES):
        s = results_dict[k]["sample"]
        ax1.plot(s["times"], s["positions"][:, 0], label=f"K={k}", color=colors[i], lw=2.0)
    ax1.axhline(0.0, color='black', linestyle=':')
    ax1.axvspan(GUST_START, GUST_START + GUST_DURATION, color='lightgray', alpha=0.5, label=f'Gust {GUST_SPEED} m/s')
    ax1.set_title("X-Position Response under 6.0 m/s Gust", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X Position (m)")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # 2. 위치 RMSE 비교 (막대 그래프)
    ax2 = fig.add_subplot(2, 2, 2)
    means = [results_dict[k]["rmse_mean"] for k in K_CANDIDATES]
    stds = [results_dict[k]["rmse_std"] for k in K_CANDIDATES]
    bars = ax2.bar(k_labels, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    ax2.set_title("Position Tracking Error (RMSE)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("RMSE (m)")
    ax2.grid(True, axis='y')
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.3f}m", ha='center', va='bottom', fontweight='bold')

    # 3. 최대 밀림 거리 비교
    ax3 = fig.add_subplot(2, 2, 3)
    disp_means = [results_dict[k]["disp_mean"] for k in K_CANDIDATES]
    disp_stds = [results_dict[k]["disp_std"] for k in K_CANDIDATES]
    bars3 = ax3.bar(k_labels, disp_means, yerr=disp_stds, capsize=6, color=colors, alpha=0.85)
    ax3.set_title("Max Displacement under Gust", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Displacement (m)")
    ax3.grid(True, axis='y')
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}m", ha='center', va='bottom', fontweight='bold')

    # 4. 허용 오차 복귀 정착 시간(Settling Time) 비교
    ax4 = fig.add_subplot(2, 2, 4)
    settling_means = [results_dict[k]["settling_mean"] for k in K_CANDIDATES]
    settling_stds = [results_dict[k]["settling_std"] for k in K_CANDIDATES]
    bars4 = ax4.bar(k_labels, settling_means, yerr=settling_stds, capsize=6, color=colors, alpha=0.85)
    ax4.set_title("Settling Time to ±5cm Window", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Settling Time (s)")
    ax4.grid(True, axis='y')
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f}s", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "ablation_history_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] 소거 실험 그래프가 저장되었습니다: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="History Buffer Ablation Study")
    parser.add_argument("--eval_only", action="store_true", help="학습을 생략하고 평가만 진행")
    parser.add_argument("--timesteps", type=int, default=350_000, help="모델당 학습 스텝 수")
    args = parser.parse_args()

    # 1. 모델 순차 학습 (없는 경우 자동 훈련)
    if not args.eval_only:
        for k in K_CANDIDATES:
            train_ablation_model(k, timesteps=args.timesteps)

    # 2. 성능 평가
    print("\n" + "=" * 70)
    print(f"소거 실험 평가 시작 (총 {len(K_CANDIDATES)}개 모델 x {EVAL_EPISODES}회 에피소드)")
    print(f"평가 외란 조건: {GUST_SPEED} m/s 계단형 돌풍")
    print("=" * 70)

    results = {}
    for k in K_CANDIDATES:
        print(f">> K={k} 모델 평가 진행 중...")
        results[k] = evaluate_k_model(k)

    # 3. 콘솔 표 출력
    print("\n" + "=" * 82)
    print(f"{'설정 (K)':<12} | {'관찰 차원':<10} | {'위치 RMSE (m)':<18} | {'최대 밀림 (m)':<16} | {'정착 시간 (s)':<14}")
    print("-" * 82)
    for k in K_CANDIDATES:
        r = results[k]
        dim_str = f"{18 * (k + 1)}D"
        print(f"K={k:<9} | {dim_str:<10} | {r['rmse_mean']:.4f} ± {r['rmse_std']:.4f}  | {r['disp_mean']:.4f} ± {r['disp_std']:.4f} | {r['settling_mean']:.2f} ± {r['settling_std']:.2f}")
    print("=" * 82)

    # 4. 시각화 생성
    plot_ablation_results(results)


if __name__ == "__main__":
    main()