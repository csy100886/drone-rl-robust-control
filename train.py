# train.py
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper
from models.ppo_agent import create_ppo_agent
from utils.logger import PerformanceEvaluator, export_metrics_to_csv


def make_env(rank: int, seed: int = 42):
    def _init():
        raw_env = QuadrotorEnv(enable_dr=False)
        env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
        log_dir = f"./logs/env_{rank}"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate_and_log_summary(model, save_csv_path):
    """학습 직후 표준 환경에서 최종 성능 측정 및 CSV 요약 출력"""
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    evaluator = PerformanceEvaluator(dt=config.DT)

    rmses, max_disps, settling_times, jitters = [], [], [], []
    num_episodes = 10

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_positions, ep_actions = [], []
        done, step = False, 0

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

        rmses.append(evaluator.compute_rmse(ep_positions))
        max_disps.append(evaluator.compute_max_displacement(ep_positions, times))
        s_time = evaluator.compute_settling_time(ep_positions, times)
        settling_times.append(s_time if s_time is not None else 6.2)
        jitters.append(evaluator.compute_jitter(ep_actions))

    headers = ["Model", "RMSE_Mean(m)", "RMSE_Std(m)", "Max_Disp_Mean(m)", "Settling_Time_Mean(s)", "Jitter_Mean"]
    row = [
        "Vanilla_PPO",
        f"{np.mean(rmses):.4f}", f"{np.std(rmses):.4f}",
        f"{np.mean(max_disps):.4f}", f"{np.mean(settling_times):.2f}",
        f"{np.mean(jitters):.5f}"
    ]
    export_metrics_to_csv(save_csv_path, [row], headers)

    print("\n" + "=" * 65)
    print(f"[Vanilla PPO 학습 직후 성과 검증 (10 에피소드 평균)]")
    print(f"- 위치 추적 RMSE         : {np.mean(rmses):.4f} ± {np.std(rmses):.4f} m")
    print(f"- 돌풍 최대 밀림 거리    : {np.mean(max_disps):.4f} m")
    print(f"- 복귀 정착 시간(5cm 기준): {np.mean(settling_times):.2f} s")
    print(f"- 제어 평활도(Jitter)    : {np.mean(jitters):.5f}")
    print("=" * 65)


def main():
    models_dir = "./models"
    checkpoints_dir = "./models/checkpoints"
    best_model_dir = "./models/best_model"
    log_dir = "./logs/tensorboard"
    
    for path in [models_dir, checkpoints_dir, best_model_dir, log_dir]:
        os.makedirs(path, exist_ok=True)

    num_envs = 4
    train_env = DummyVecEnv([make_env(i) for i in range(num_envs)])
    eval_env = DummyVecEnv([make_env(rank=999, seed=123)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // num_envs, 1),
        save_path=checkpoints_dir,
        name_prefix="ppo_quadrotor"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(5000 // num_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # models.ppo_agent 공통 모듈 호출
    model = create_ppo_agent(
        env=train_env,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        device="auto"
    )
    model.verbose = 1

    total_timesteps = 500_000
    print("=" * 65)
    print(f"Vanilla PPO 학습 시작 (총 {total_timesteps:,} 스텝)")
    print(f"디바이스: {model.device} | 병렬 환경 수: {num_envs}")
    print("=" * 65)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False
    )

    # 모델 저장
    final_model_path = os.path.join(models_dir, "ppo_quadrotor_final.zip")
    model.save(final_model_path)
    print(f"\n[저장 완료] 최종 모델 가중치: {final_model_path}")

    # utils.logger를 통한 즉각 성능 검증 및 CSV 기록
    evaluate_and_log_summary(model, "./results/vanilla_training_summary.csv")


if __name__ == "__main__":
    main()