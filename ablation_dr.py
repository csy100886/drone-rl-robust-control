# ablation_dr.py
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
from utils.domain_randomizer import DomainRandomizer


MODELS_DIR = "./models/ablation_dr"
RESULTS_DIR = "./results/ablation_dr"
EVAL_EPISODES = 25
GUST_SPEED = 6.0
GUST_START = 3.0
GUST_DURATION = 0.8


# -------------------------------------------------------------
# 1. 특정 파라미터만 선택적으로 무작위화하는 Randomizer
# -------------------------------------------------------------
class SelectiveDomainRandomizer(DomainRandomizer):
    def __init__(self, randomize_mass=True, randomize_tau=True):
        super().__init__()
        self.randomize_mass = randomize_mass
        self.randomize_tau = randomize_tau

    def sample_parameters(self):
        params = super().sample_parameters()
        if not self.randomize_mass:
            params["mass"] = config.MASS
            params["inertia"] = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32)
        if not self.randomize_tau:
            params["motor_tau"] = config.MOTOR_TAU
        return params


def make_env_selective(rand_mass: bool, rand_tau: bool, rank: int, seed: int = 500):
    def _init():
        enable_dr = rand_mass or rand_tau
        raw_env = QuadrotorEnv(enable_dr=enable_dr)
        if enable_dr:
            raw_env.randomizer = SelectiveDomainRandomizer(randomize_mass=rand_mass, randomize_tau=rand_tau)

        env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
        log_dir = f"./logs/ablation_dr_m{int(rand_mass)}_t{int(rand_tau)}_env_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------------------------------------
# 2. 모델 학습 루프
# -------------------------------------------------------------
CONFIGS = {
    "Full_DR": {"rand_mass": True, "rand_tau": True, "label": "Full DR (Mass + Tau)", "color": "royalblue"},
    "No_Mass_DR": {"rand_mass": False, "rand_tau": True, "label": "No Mass DR (Tau Only)", "color": "darkorange"},
    "No_Tau_DR": {"rand_mass": True, "rand_tau": False, "label": "No Tau DR (Mass Only)", "color": "purple"},
    "Vanilla": {"rand_mass": False, "rand_tau": False, "label": "Vanilla (No DR)", "color": "crimson"}
}


def train_dr_variant(variant_key: str, timesteps: int = 350_000):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"ppo_{variant_key.lower()}.zip")

    # 기존 학습된 최적 모델이 있을 경우 복사 재활용
    if variant_key == "Full_DR" and os.path.exists("./models/dr_best_model/best_model.zip"):
        print(f"[Info] 기존 Full DR 모델을 로드합니다.")
        return "./models/dr_best_model/best_model.zip"
    if variant_key == "Vanilla" and os.path.exists("./models/best_model/best_model.zip"):
        print(f"[Info] 기존 Vanilla 모델을 로드합니다.")
        return "./models/best_model/best_model.zip"

    if os.path.exists(model_save_path):
        print(f"[Info] {variant_key} 모델이 이미 존재합니다: {model_save_path}")
        return model_save_path

    cfg = CONFIGS[variant_key]
    print("=" * 65)
    print(f">> [학습 시작] {cfg['label']} (총 {timesteps:,} 스텝)")
    print("=" * 65)

    num_envs = 4
    train_env = DummyVecEnv([make_env_selective(cfg["rand_mass"], cfg["rand_tau"], i) for i in range(num_envs)])

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
    print(f"[저장 완료] {model_save_path}")
    return model_save_path


# -------------------------------------------------------------
# 3. 스트레스 평가 시나리오 정의 및 벤치마크
# -------------------------------------------------------------
SCENARIOS = {
    "Mass_Stress": {
        "title": "Mass Stress (+30%)",
        "mass_factor": 1.30,
        "motor_tau": config.MOTOR_TAU,
        "inertia_factor": 1.30
    },
    "Lag_Stress": {
        "title": "Actuator Lag Stress (50ms)",
        "mass_factor": 1.00,
        "motor_tau": 0.050,
        "inertia_factor": 1.00
    },
    "Combined_Stress": {
        "title": "Combined Stress (Mass +30% & 50ms Lag)",
        "mass_factor": 1.30,
        "motor_tau": 0.050,
        "inertia_factor": 1.30
    }
}


def evaluate_variant_under_scenario(model_path, scenario_cfg, num_episodes=25):
    model = PPO.load(model_path)
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    config.STEP_GUST_MAGNITUDE = np.array([GUST_SPEED, 0.0, 0.0], dtype=np.float32)

    target = np.array([0.0, 0.0, 1.0])
    rmses = []
    survivals = []
    sample_trajectory = None

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=6000 + ep)

        # 스트레스 파라미터 강제 주입
        raw_env.mass = config.MASS * scenario_cfg["mass_factor"]
        raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32) * scenario_cfg["inertia_factor"]
        raw_env.motor_tau = scenario_cfg["motor_tau"]

        ep_positions = []
        step = 0
        done = False

        while step < config.MAX_STEPS and not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_positions.append(raw_env.state["pos"].copy())

            obs, _, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated

        ep_positions = np.array(ep_positions)
        times = np.arange(len(ep_positions)) * config.DT
        survived = (step >= config.MAX_STEPS)
        survivals.append(survived)

        pos_err = ep_positions - target
        rmse = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
        rmses.append(rmse)

        if ep == 0:
            sample_trajectory = {
                "times": times,
                "positions": ep_positions,
                "survived": survived
            }

    return {
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses),
        "survival_rate": np.mean(survivals) * 100.0,
        "sample": sample_trajectory
    }


# -------------------------------------------------------------
# 4. 시각화 플롯 생성
# -------------------------------------------------------------
def plot_dr_ablation_results(results):
    fig = plt.figure(figsize=(16, 10))
    variant_keys = list(CONFIGS.keys())
    labels = [CONFIGS[k]["label"] for k in variant_keys]
    colors = [CONFIGS[k]["color"] for k in variant_keys]

    # Subplot 1: 질량 스트레스 환경(Mass +30%)에서의 위치 오차
    ax1 = fig.add_subplot(2, 2, 1)
    means_mass = [results[k]["Mass_Stress"]["rmse_mean"] for k in variant_keys]
    stds_mass = [results[k]["Mass_Stress"]["rmse_std"] for k in variant_keys]
    bars1 = ax1.bar(labels, means_mass, yerr=stds_mass, capsize=5, color=colors, alpha=0.85)
    ax1.set_title("Tracking Error under Mass Stress (+30%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("RMSE (m)")
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax1.grid(True, axis='y')
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.005, f"{b.get_height():.3f}m", ha='center', va='bottom', fontsize=9)

    # Subplot 2: 모터 지연 스트레스 환경(50ms Lag)에서의 위치 오차
    ax2 = fig.add_subplot(2, 2, 2)
    means_lag = [results[k]["Lag_Stress"]["rmse_mean"] for k in variant_keys]
    stds_lag = [results[k]["Lag_Stress"]["rmse_std"] for k in variant_keys]
    bars2 = ax2.bar(labels, means_lag, yerr=stds_lag, capsize=5, color=colors, alpha=0.85)
    ax2.set_title("Tracking Error under Motor Lag Stress (50ms)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("RMSE (m)")
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax2.grid(True, axis='y')
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2.0, b.get_height() + 0.005, f"{b.get_height():.3f}m", ha='center', va='bottom', fontsize=9)

    # Subplot 3: 복합 스트레스 환경(Mass +30% & 50ms Lag)에서의 비행 생존율
    ax3 = fig.add_subplot(2, 2, 3)
    surv_comb = [results[k]["Combined_Stress"]["survival_rate"] for k in variant_keys]
    bars3 = ax3.bar(labels, surv_comb, color=colors, alpha=0.85)
    ax3.set_title("Survival Rate under Combined Stress", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Survival Rate (%)")
    ax3.set_ylim([0, 115])
    ax3.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax3.grid(True, axis='y')
    for b in bars3:
        ax3.text(b.get_x() + b.get_width()/2.0, b.get_height() + 2.0, f"{b.get_height():.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Subplot 4: 복합 스트레스 시 고도(Z축) 과도 응답 시계열
    ax4 = fig.add_subplot(2, 2, 4)
    for k in variant_keys:
        sample = results[k]["Combined_Stress"]["sample"]
        ax4.plot(sample["times"], sample["positions"][:, 2], label=CONFIGS[k]["label"], color=CONFIGS[k]["color"], lw=1.8)
    ax4.axhline(1.0, color='black', linestyle=':', label='Target Altitude (1.0m)')
    ax4.axvspan(GUST_START, GUST_START + GUST_DURATION, color='gray', alpha=0.25, label=f'Gust {GUST_SPEED} m/s')
    ax4.set_title("Altitude (Z) Response under Combined Stress", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Altitude Z (m)")
    ax4.grid(True)
    ax4.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "ablation_dr_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[성공] DR 소거 실험 그래프 저장 완료: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Domain Randomization Component Ablation")
    parser.add_argument("--eval_only", action="store_true", help="학습 건너뛰고 기존 모델로 평가만 수행")
    parser.add_argument("--timesteps", type=int, default=350_000, help="모델당 훈련 스텝")
    args = parser.parse_args()

    model_paths = {}

    # 1. 4개 모델 순차 훈련/확보
    for k in CONFIGS.keys():
        if not args.eval_only:
            model_paths[k] = train_dr_variant(k, timesteps=args.timesteps)
        else:
            default_path = os.path.join(MODELS_DIR, f"ppo_{k.lower()}.zip")
            if k == "Full_DR" and os.path.exists("./models/dr_best_model/best_model.zip"):
                default_path = "./models/dr_best_model/best_model.zip"
            elif k == "Vanilla" and os.path.exists("./models/best_model/best_model.zip"):
                default_path = "./models/best_model/best_model.zip"
            model_paths[k] = default_path

    # 2. 스트레스 시나리오 평가 진행
    print("\n" + "=" * 70)
    print("도메인 무작위화 파라미터 소거 실험(Ablation on DR) 벤치마크")
    print("=" * 70)

    results = {k: {} for k in CONFIGS.keys()}

    for s_key, s_cfg in SCENARIOS.items():
        print(f"\n>> 시나리오: [{s_cfg['title']}] 평가 중...")
        for v_key in CONFIGS.keys():
            res = evaluate_variant_under_scenario(model_paths[v_key], s_cfg, num_episodes=EVAL_EPISODES)
            results[v_key][s_key] = res

    # 3. 콘솔 요약 테이블 출력
    print("\n" + "=" * 88)
    print(f"{'모델 구성':<22} | {'질량 스트레스 RMSE':<18} | {'지연 스트레스 RMSE':<18} | {'복합 스트레스 생존율':<18}")
    print("-" * 88)
    for k in CONFIGS.keys():
        m_rmse = f"{results[k]['Mass_Stress']['rmse_mean']:.4f} ± {results[k]['Mass_Stress']['rmse_std']:.4f}"
        l_rmse = f"{results[k]['Lag_Stress']['rmse_mean']:.4f} ± {results[k]['Lag_Stress']['rmse_std']:.4f}"
        c_surv = f"{results[k]['Combined_Stress']['survival_rate']:>5.1f} %"
        print(f"{CONFIGS[k]['label']:<22} | {m_rmse:<18} | {l_rmse:<18} | {c_surv:<18}")
    print("=" * 88)

    # 4. 결과 시각화
    plot_dr_ablation_results(results)


if __name__ == "__main__":
    main()