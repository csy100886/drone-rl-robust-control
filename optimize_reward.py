# optimize_reward.py
import os
import json
import argparse
import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
import optuna
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper
from models.ppo_agent import create_ppo_agent
from utils.logger import PerformanceEvaluator


RESULTS_DIR = "./results"
DB_PATH = "sqlite:///results/optuna_study.db"
PKL_PATH = os.path.join(RESULTS_DIR, "optuna_study.pkl")
JSON_PATH = os.path.join(RESULTS_DIR, "best_reward_weights.json")


# -------------------------------------------------------------
# 1. 보상 가중치를 동적으로 주입받는 환경 클래스
# -------------------------------------------------------------
class TunableRewardEnv(QuadrotorEnv):
    def __init__(self, reward_weights: dict, enable_dr: bool = True):
        super().__init__(enable_dr=enable_dr)
        self.weights = reward_weights

    def _compute_reward(self, action):
        pos_err = np.linalg.norm(self.target_pos - self.state["pos"])
        vel_norm = np.linalg.norm(self.state["vel"])
        ang_vel_norm = np.linalg.norm(self.state["ang_vel"])

        # 1. 자세 오차: Roll/Pitch 및 Yaw 독립 제어
        rpy = R.from_matrix(self.state["rot"]).as_euler('xyz')
        roll_pitch_err = np.sum(rpy[:2]**2)
        yaw_err = rpy[2]**2
        r_att = np.exp(-4.0 * roll_pitch_err) * (1.0 - self.weights["w_yaw"]) + \
                np.exp(-2.0 * yaw_err) * self.weights["w_yaw"]

        # 2. 위치 수렴 지수 보상
        r_pos = np.exp(-self.weights["pos_exp"] * pos_err**2)

        # 3. 속도/각속도 감쇠 (Damping)
        r_vel = np.exp(-0.5 * vel_norm**2)
        r_ang = np.exp(-0.5 * ang_vel_norm**2)

        # 4. 제어 평활화 (Jittering 억제)
        action_diff = np.sum((action - self.prev_action)**2)
        action_mag = np.sum(action**2)
        r_smooth = (self.weights["w_smooth"] * action_diff) + (self.weights["w_action_mag"] * action_mag)

        reward = (self.weights["w_pos"] * r_pos) + \
                 (self.weights["w_att"] * r_att) + \
                 (0.4 * r_vel) + (0.3 * r_ang) - r_smooth + 0.5
        return float(reward)


def make_tunable_env(weights, rank, enable_dr=True, seed=100):
    def _init():
        raw_env = TunableRewardEnv(reward_weights=weights, enable_dr=enable_dr)
        env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
        env.reset(seed=seed + rank)
        return env
    return _init


# -------------------------------------------------------------
# 2. Optuna Objective 함수 (단축 학습 및 목적 지표 평가)
# -------------------------------------------------------------
def objective(trial: optuna.Trial, train_timesteps: int = 100_000):
    # 탐색 공간(Search Space) 정의
    weights = {
        "w_pos": trial.suggest_float("w_pos", 1.2, 3.5, step=0.1),
        "pos_exp": trial.suggest_float("pos_exp", 3.0, 7.0, step=0.5),
        "w_att": trial.suggest_float("w_att", 0.8, 2.0, step=0.1),
        "w_yaw": trial.suggest_float("w_yaw", 0.15, 0.50, step=0.05),
        "w_smooth": trial.suggest_float("w_smooth", 0.3, 1.8, step=0.1),
        "w_action_mag": trial.suggest_float("w_action_mag", 0.02, 0.10, step=0.01)
    }

    # 4개 병렬 환경에서 지정된 스텝만큼 고속 학습
    num_envs = 4
    train_env = DummyVecEnv([make_tunable_env(weights, i, enable_dr=True, seed=trial.number * 10) for i in range(num_envs)])
    model = create_ppo_agent(env=train_env, device="auto")
    model.learn(total_timesteps=train_timesteps)

    # 평가 환경 (6.0 m/s 돌풍 인가 조건에서 10회 검증)
    eval_raw = TunableRewardEnv(reward_weights=weights, enable_dr=False)
    eval_env = HistoryBufferWrapper(eval_raw, history_length=config.HISTORY_LENGTH)
    config.STEP_GUST_MAGNITUDE = np.array([6.0, 0.0, 0.0], dtype=np.float32)
    evaluator = PerformanceEvaluator(dt=config.DT)

    rmses, jitters, survivals = [], [], []

    for ep in range(10):
        obs, _ = eval_env.reset(seed=9000 + ep)
        ep_pos, ep_act = [], []
        step, done = 0, False

        while step < config.MAX_STEPS and not done:
            act, _ = model.predict(obs, deterministic=True)
            ep_pos.append(eval_raw.state["pos"].copy())
            ep_act.append(act.copy())
            obs, _, term, trunc, _ = eval_env.step(act)
            step += 1
            done = term or trunc

        ep_pos = np.array(ep_pos)
        ep_act = np.array(ep_act)
        rmses.append(evaluator.compute_rmse(ep_pos))
        jitters.append(evaluator.compute_jitter(ep_act))
        survivals.append(step >= config.MAX_STEPS)

    mean_rmse = float(np.mean(rmses))
    mean_jitter = float(np.mean(jitters))
    survival_rate = float(np.mean(survivals) * 100.0)

    # 다목적 합성 비용 함수 (Cost): RMSE 최소화 + Jitter 최소화 + 추락 페널티
    cost = mean_rmse + (3.0 * mean_jitter) + (100.0 - survival_rate) * 0.05

    # plot_optuna_results.py에서 시각화할 속성 저장
    trial.set_user_attr("mean_rmse", mean_rmse)
    trial.set_user_attr("mean_jitter", mean_jitter)
    trial.set_user_attr("survival_rate", survival_rate)

    print(f">> [Trial {trial.number}] Cost: {cost:.4f} | RMSE: {mean_rmse:.4f}m | Jitter: {mean_jitter:.5f} | Surv: {survival_rate:.1f}%")
    return cost


# -------------------------------------------------------------
# 3. 메인 실행 및 최적 파라미터 저장
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna Reward Optimization")
    parser.add_argument("--n-trials", type=int, default=15, help="최적화 시도 횟수 (권장: 15~20회)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Trial당 학습 스텝 수")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 75)
    print("Optuna 기반 쿼드로터 보상 함수 자율 최적화 파이프라인 가동")
    print(f"총 Trial 수: {args.n_trials}회 | Trial당 학습 스텝: {args.timesteps:,} 스텝")
    print("최적화 목표: 위치 추적 오차(RMSE) 및 제어 진동(Jitter) 동시 최소화")
    print(f"스터디 DB   : {DB_PATH}")
    print("=" * 75)

    # SQLite DB 저장 설정 (plot_optuna_results.py와 완전 연동)
    study = optuna.create_study(
        study_name="quadrotor_reward_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=DB_PATH,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, train_timesteps=args.timesteps), n_trials=args.n_trials)

    # 1. Pickle 파일로 스터디 객체 백업 저장
    joblib.dump(study, PKL_PATH)
    print(f"\n[저장 완료] 스터디 객체 백업: {PKL_PATH}")

    # 2. 최적 결과 콘솔 출력
    print("\n" + "=" * 75)
    print("[최적화 완료! Best Trial 결과]")
    best = study.best_trial
    print(f"- Trial 번호     : #{best.number}")
    print(f"- 최소 종합 Cost : {best.value:.4f}")
    print(f"- 위치 RMSE      : {best.user_attrs['mean_rmse']:.4f} m")
    print(f"- 제어 Jitter    : {best.user_attrs['mean_jitter']:.5f}")
    print(f"- 비행 생존율    : {best.user_attrs['survival_rate']:.1f} %")
    print("\n[도출된 최적 보상 가중치]")
    for k, v in best.params.items():
        print(f"  '{k}': {v}")
    print("=" * 75)

    # 3. 최적 가중치 JSON 파일 저장
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=4)
    print(f"\n[저장 완료] 최적 가중치 JSON 파일: {JSON_PATH}")

    # 4. envs/quadrotor_env.py 적용 코드 스니펫 안내
    p = best.params
    print("\n--- [envs/quadrotor_env.py의 _compute_reward 적용 권장 코드] ---")
    print(f"""
    def _compute_reward(self, action):
        pos_err = np.linalg.norm(self.target_pos - self.state["pos"])
        vel_norm = np.linalg.norm(self.state["vel"])
        ang_vel_norm = np.linalg.norm(self.state["ang_vel"])

        rpy = R.from_matrix(self.state["rot"]).as_euler('xyz')
        roll_pitch_err = np.sum(rpy[:2]**2)
        yaw_err = rpy[2]**2
        r_att = np.exp(-4.0 * roll_pitch_err) * {1.0 - p['w_yaw']:.2f} + np.exp(-2.0 * yaw_err) * {p['w_yaw']:.2f}

        r_pos = np.exp(-{p['pos_exp']:.1f} * pos_err**2)
        r_vel = np.exp(-0.5 * vel_norm**2)
        r_ang = np.exp(-0.5 * ang_vel_norm**2)

        action_diff = np.sum((action - self.prev_action)**2)
        action_mag = np.sum(action**2)
        r_smooth = {p['w_smooth']:.2f} * action_diff + {p['w_action_mag']:.2f} * action_mag

        reward = ({p['w_pos']:.2f} * r_pos) + ({p['w_att']:.2f} * r_att) + (0.4 * r_vel) + (0.3 * r_ang) - r_smooth + 0.5
        return float(reward)
    """)


if __name__ == "__main__":
    main()