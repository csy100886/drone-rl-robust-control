# generate_latex_tables.py
import os
import argparse
import numpy as np
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper
from baselines.pid_controller import CascadedPIDController
from utils.logger import PerformanceEvaluator


OUTPUT_TEX_PATH = "./results/thesis_tables.tex"


# -------------------------------------------------------------
# 유틸리티: 모델 로드 헬퍼
# -------------------------------------------------------------
def load_model_safe(paths):
    for p in paths:
        if os.path.exists(p):
            return PPO.load(p), p
    return None, None


# -------------------------------------------------------------
# 1. 3자 제어기 (PID vs Vanilla vs DR-PPO) 풍속별 벤치마크
# -------------------------------------------------------------
def evaluate_3way_benchmark(episodes=15):
    print("\n[1/4] 3자 제어기 풍속별 성능 평가 수행 중...")
    v_model, _ = load_model_safe(["./models/best_model/best_model.zip", "./models/ppo_quadrotor_final.zip"])
    dr_model, _ = load_model_safe(["./models/dr_best_model/best_model.zip", "./models/dr_model/ppo_dr_final.zip"])
    
    if v_model is None or dr_model is None:
        print("[경고] Vanilla 또는 DR-PPO 가중치가 없어 3자 비교 평가를 건너뜁니다.")
        return None

    speeds = [2.0, 5.0, 8.0]
    models = {'PID': None, 'Vanilla PPO': v_model, 'DR-PPO (Ours)': dr_model}
    evaluator = PerformanceEvaluator(dt=config.DT)
    results = {m: {} for m in models}

    for w in speeds:
        config.STEP_GUST_MAGNITUDE = np.array([w, 0.0, 0.0], dtype=np.float32)
        for m_name, model in models.items():
            raw_env = QuadrotorEnv(enable_dr=False)
            env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
            pid = CascadedPIDController(dt=config.DT) if m_name == 'PID' else None

            rmses, max_disps, survivals, settlings = [], [], [], []

            for ep in range(episodes):
                obs, _ = env.reset(seed=5000 + ep)
                if pid:
                    pid.reset()
                ep_pos = []
                step, done = 0, False

                while step < config.MAX_STEPS and not done:
                    if m_name == 'PID':
                        act = pid.compute_action(raw_env.state)
                    else:
                        act, _ = model.predict(obs, deterministic=True)
                    ep_pos.append(raw_env.state["pos"].copy())
                    obs, _, term, trunc, _ = env.step(act)
                    step += 1
                    done = term or trunc

                ep_pos = np.array(ep_pos)
                times = np.arange(len(ep_pos)) * config.DT
                survived = (step >= config.MAX_STEPS)

                rmses.append(evaluator.compute_rmse(ep_pos))
                max_disps.append(evaluator.compute_max_displacement(ep_pos, times))
                survivals.append(survived)
                s_time = evaluator.compute_settling_time(ep_pos, times)
                settlings.append(s_time if s_time is not None else 6.2)

            results[m_name][w] = {
                "rmse": (np.mean(rmses), np.std(rmses)),
                "disp": (np.mean(max_disps), np.std(max_disps)),
                "surv": np.mean(survivals) * 100.0,
                "settling": (np.mean(settlings), np.std(settlings))
            }
    return results


def format_latex_table_3way(data):
    if not data:
        return "% [Table 1 생략: 모델 가중치 미존재]\n"

    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{정량적 비행 성능 벤치마크: 외란 풍속별 위치 추적 오차 및 안정성 비교}")
    tex.append(r"\label{tab:benchmark_3way}")
    tex.append(r"\begin{tabular}{llcccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{풍속 ($v_w$)} & \textbf{제어기 (Controller)} & \textbf{위치 RMSE (m)} $\downarrow$ & \textbf{최대 밀림 ($e_{\max}$) (m)} $\downarrow$ & \textbf{정착 시간 ($T_s$) (s)} $\downarrow$ & \textbf{생존율 (\%)} $\uparrow$ \\")
    tex.append(r"\midrule")

    speeds = [2.0, 5.0, 8.0]
    for w in speeds:
        tex.append(f"\\multirow{{3}}{{*}}{{{w:.1f} m/s}}")
        for m_name in ['PID', 'Vanilla PPO', 'DR-PPO (Ours)']:
            d = data[m_name][w]
            rmse_str = f"{d['rmse'][0]:.3f} \\pm {d['rmse'][1]:.3f}"
            disp_str = f"{d['disp'][0]:.3f} \\pm {d['disp'][1]:.3f}"
            settle_str = f"{d['settling'][0]:.2f} \\pm {d['settling'][1]:.2f}"
            surv_str = f"{d['surv']:.1f}\\%"

            if m_name == 'DR-PPO (Ours)':
                rmse_str = f"\\textbf{{{rmse_str}}}"
                disp_str = f"\\textbf{{{disp_str}}}"
                settle_str = f"\\textbf{{{settle_str}}}"
                surv_str = f"\\textbf{{{surv_str}}}"

            tex.append(f" & {m_name:<16} & {rmse_str} & {disp_str} & {settle_str} & {surv_str} \\\\")
        if w != speeds[-1]:
            tex.append(r"\cmidrule{1-6}")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")
    return "\n".join(tex)


# -------------------------------------------------------------
# 2. 극한 스트레스 환경 벤치마크 (Mass +30%, Delay 50ms)
# -------------------------------------------------------------
def evaluate_stress_benchmark(episodes=15):
    print("[2/4] 극한 스트레스 조건(Mass +30%, Lag 50ms) 평가 수행 중...")
    v_model, _ = load_model_safe(["./models/best_model/best_model.zip", "./models/ppo_quadrotor_final.zip"])
    dr_model, _ = load_model_safe(["./models/dr_best_model/best_model.zip", "./models/dr_model/ppo_dr_final.zip"])

    if v_model is None or dr_model is None:
        return None

    models = {'Vanilla PPO': v_model, 'DR-PPO (Ours)': dr_model}
    evaluator = PerformanceEvaluator(dt=config.DT)
    results = {}

    config.STEP_GUST_MAGNITUDE = np.array([5.0, 0.0, 0.0], dtype=np.float32)

    for m_name, model in models.items():
        raw_env = QuadrotorEnv(enable_dr=False)
        env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)

        # 극한 조건 강제 주입
        raw_env.mass = config.MASS * 1.30
        raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32) * 1.30
        raw_env.motor_tau = 0.050

        rmses, disps, jitters, survivals = [], [], [], []

        for ep in range(episodes):
            obs, _ = env.reset(seed=6000 + ep)
            ep_pos, ep_act = [], []
            step, done = 0, False

            while step < config.MAX_STEPS and not done:
                act, _ = model.predict(obs, deterministic=True)
                ep_pos.append(raw_env.state["pos"].copy())
                ep_act.append(act.copy())
                obs, _, term, trunc, _ = env.step(act)
                step += 1
                done = term or trunc

            ep_pos = np.array(ep_pos)
            ep_act = np.array(ep_act)
            times = np.arange(len(ep_pos)) * config.DT

            rmses.append(evaluator.compute_rmse(ep_pos))
            disps.append(evaluator.compute_max_displacement(ep_pos, times))
            jitters.append(evaluator.compute_jitter(ep_act))
            survivals.append(step >= config.MAX_STEPS)

        results[m_name] = {
            "rmse": (np.mean(rmses), np.std(rmses)),
            "disp": (np.mean(disps), np.std(disps)),
            "jitter": (np.mean(jitters), np.std(jitters)),
            "surv": np.mean(survivals) * 100.0
        }
    return results


def format_latex_table_stress(data):
    if not data:
        return "% [Table 2 생략: 모델 가중치 미존재]\n"

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{극한 파라미터 변동 환경(질량 $+30\%$, 모터 지연 $50\,\text{ms}$) 강인성 평가}")
    tex.append(r"\label{tab:stress_benchmark}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{제어기} & \textbf{위치 RMSE (m)} $\downarrow$ & \textbf{최대 밀림 (m)} $\downarrow$ & \textbf{제어 진동 (Jitter)} $\downarrow$ & \textbf{생존율 (\%)} $\uparrow$ \\")
    tex.append(r"\midrule")

    for m_name in ['Vanilla PPO', 'DR-PPO (Ours)']:
        d = data[m_name]
        rmse_str = f"{d['rmse'][0]:.3f} \\pm {d['rmse'][1]:.3f}"
        disp_str = f"{d['disp'][0]:.3f} \\pm {d['disp'][1]:.3f}"
        jit_str = f"{d['jitter'][0]:.5f}"
        surv_str = f"{d['surv']:.1f}\\%"

        if 'Ours' in m_name:
            rmse_str = f"\\textbf{{{rmse_str}}}"
            disp_str = f"\\textbf{{{disp_str}}}"
            jit_str = f"\\textbf{{{jit_str}}}"
            surv_str = f"\\textbf{{{surv_str}}}"

        tex.append(f"{m_name:<16} & {rmse_str} & {disp_str} & {jit_str} & {surv_str} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# -------------------------------------------------------------
# 3. 과거 관찰 이력 크기(K) 소거 실험
# -------------------------------------------------------------
def evaluate_history_ablation(episodes=15):
    print("[3/4] 과거 관찰 이력 길이(K=0, 2, 4) 소거 실험 평가 수행 중...")
    k_list = [0, 2, 4]
    evaluator = PerformanceEvaluator(dt=config.DT)
    results = {}
    config.STEP_GUST_MAGNITUDE = np.array([6.0, 0.0, 0.0], dtype=np.float32)

    for k in k_list:
        model_path = f"./models/ablation/ppo_k{k}.zip"
        if not os.path.exists(model_path):
            if k == 4:
                model_path = "./models/dr_best_model/best_model.zip"
            else:
                continue

        model = PPO.load(model_path)
        raw_env = QuadrotorEnv(enable_dr=False)
        env = HistoryBufferWrapper(raw_env, history_length=k)

        rmses, disps, settlings, jitters = [], [], [], []

        for ep in range(episodes):
            obs, _ = env.reset(seed=7000 + ep)
            ep_pos, ep_act = [], []
            step, done = 0, False

            while step < config.MAX_STEPS and not done:
                act, _ = model.predict(obs, deterministic=True)
                ep_pos.append(raw_env.state["pos"].copy())
                ep_act.append(act.copy())
                obs, _, term, trunc, _ = env.step(act)
                step += 1
                done = term or trunc

            ep_pos = np.array(ep_pos)
            ep_act = np.array(ep_act)
            times = np.arange(len(ep_pos)) * config.DT

            rmses.append(evaluator.compute_rmse(ep_pos))
            disps.append(evaluator.compute_max_displacement(ep_pos, times))
            s_time = evaluator.compute_settling_time(ep_pos, times)
            settlings.append(s_time if s_time is not None else 6.2)
            jitters.append(evaluator.compute_jitter(ep_act))

        results[k] = {
            "dim": 18 * (k + 1),
            "rmse": (np.mean(rmses), np.std(rmses)),
            "disp": (np.mean(disps), np.std(disps)),
            "settling": (np.mean(settlings), np.std(settlings)),
            "jitter": (np.mean(jitters), np.std(jitters))
        }
    return results


def format_latex_table_history(data):
    if not data:
        return "% [Table 3 생략: K별 소거 모델 가중치 미존재]\n"

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{과거 관찰 이력 크기($K$)에 따른 소거 실험 ($6.0\,\text{m/s}$ 돌풍 환경)}")
    tex.append(r"\label{tab:ablation_history}")
    tex.append(r"\begin{tabular}{cccccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{버퍼 길이 ($K$)} & \textbf{관찰 차원} & \textbf{위치 RMSE (m)} $\downarrow$ & \textbf{최대 밀림 (m)} $\downarrow$ & \textbf{정착 시간 ($T_s$) (s)} $\downarrow$ & \textbf{Jitter} $\downarrow$ \\")
    tex.append(r"\midrule")

    for k in sorted(data.keys()):
        d = data[k]
        k_str = f"$K={k}$" if k > 0 else "$K=0$ (Markovian)"
        rmse_str = f"{d['rmse'][0]:.3f} \\pm {d['rmse'][1]:.3f}"
        disp_str = f"{d['disp'][0]:.3f} \\pm {d['disp'][1]:.3f}"
        set_str = f"{d['settling'][0]:.2f} \\pm {d['settling'][1]:.2f}"
        jit_str = f"{d['jitter'][0]:.5f}"

        if k == 4:
            k_str = f"\\textbf{{{k_str} (Ours)}}"
            rmse_str = f"\\textbf{{{rmse_str}}}"
            disp_str = f"\\textbf{{{disp_str}}}"
            set_str = f"\\textbf{{{set_str}}}"
            jit_str = f"\\textbf{{{jit_str}}}"

        tex.append(f"{k_str:<20} & {d['dim']} & {rmse_str} & {disp_str} & {set_str} & {jit_str} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# -------------------------------------------------------------
# 4. 도메인 무작위화 구성 요소 소거 실험
# -------------------------------------------------------------
def evaluate_dr_ablation(episodes=15):
    print("[4/4] 도메인 무작위화 파라미터 소거 실험 평가 수행 중...")
    variants = {
        "Full DR (Mass + Tau)": "./models/dr_best_model/best_model.zip",
        "No Mass DR (Tau Only)": "./models/ablation_dr/ppo_no_mass_dr.zip",
        "No Tau DR (Mass Only)": "./models/ablation_dr/ppo_no_tau_dr.zip",
        "Vanilla (No DR)": "./models/best_model/best_model.zip"
    }

    evaluator = PerformanceEvaluator(dt=config.DT)
    results = {}

    for label, path in variants.items():
        if not os.path.exists(path):
            continue
        model = PPO.load(path)
        raw_env = QuadrotorEnv(enable_dr=False)
        env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)

        # 1. 질량 스트레스 (+30%)
        raw_env.mass = config.MASS * 1.30
        raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32) * 1.30
        raw_env.motor_tau = config.MOTOR_TAU

        m_rmses = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=8000 + ep)
            ep_pos, step, done = [], 0, False
            while step < config.MAX_STEPS and not done:
                act, _ = model.predict(obs, deterministic=True)
                ep_pos.append(raw_env.state["pos"].copy())
                obs, _, term, trunc, _ = env.step(act)
                step += 1
                done = term or trunc
            m_rmses.append(evaluator.compute_rmse(np.array(ep_pos)))

        # 2. 지연 스트레스 (50ms)
        raw_env.mass = config.MASS
        raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32)
        raw_env.motor_tau = 0.050

        l_rmses = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=8100 + ep)
            ep_pos, step, done = [], 0, False
            while step < config.MAX_STEPS and not done:
                act, _ = model.predict(obs, deterministic=True)
                ep_pos.append(raw_env.state["pos"].copy())
                obs, _, term, trunc, _ = env.step(act)
                step += 1
                done = term or trunc
            l_rmses.append(evaluator.compute_rmse(np.array(ep_pos)))

        # 3. 복합 스트레스 생존율
        raw_env.mass = config.MASS * 1.30
        raw_env.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32) * 1.30
        raw_env.motor_tau = 0.050

        survivals = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=8200 + ep)
            step, done = 0, False
            while step < config.MAX_STEPS and not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(act)
                step += 1
                done = term or trunc
            survivals.append(step >= config.MAX_STEPS)

        results[label] = {
            "mass_rmse": (np.mean(m_rmses), np.std(m_rmses)),
            "lag_rmse": (np.mean(l_rmses), np.std(l_rmses)),
            "comb_surv": np.mean(survivals) * 100.0
        }
    return results


def format_latex_table_dr(data):
    if not data:
        return "% [Table 4 생략: DR 소거 모델 가중치 미존재]\n"

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{도메인 무작위화 구성 파라미터 소거 실험}")
    tex.append(r"\label{tab:ablation_dr}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{학습 구성 (Ablation Variant)} & \textbf{질량 스트레스 RMSE (m)} $\downarrow$ & \textbf{지연 스트레스 RMSE (m)} $\downarrow$ & \textbf{복합 스트레스 생존율 (\%)} $\uparrow$ \\")
    tex.append(r"\midrule")

    for label in ["Full DR (Mass + Tau)", "No Mass DR (Tau Only)", "No Tau DR (Mass Only)", "Vanilla (No DR)"]:
        if label not in data:
            continue
        d = data[label]
        m_str = f"{d['mass_rmse'][0]:.3f} \\pm {d['mass_rmse'][1]:.3f}"
        l_str = f"{d['lag_rmse'][0]:.3f} \\pm {d['lag_rmse'][1]:.3f}"
        s_str = f"{d['comb_surv']:.1f}\\%"

        if "Full DR" in label:
            label = f"\\textbf{{{label} (Ours)}}"
            m_str = f"\\textbf{{{m_str}}}"
            l_str = f"\\textbf{{{l_str}}}"
            s_str = f"\\textbf{{{s_str}}}"

        tex.append(f"{label:<28} & {m_str} & {l_str} & {s_str} \\\\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# -------------------------------------------------------------
# 메인 함수: 실행 및 통합 LaTeX 파일 생성
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=15, help="표 생성을 위한 반복 평가 횟수")
    args = parser.parse_args()

    print("=" * 70)
    print("LaTeX 표 자동 생성 파이프라인 가동")
    print("=" * 70)

    t1_data = evaluate_3way_benchmark(episodes=args.episodes)
    t2_data = evaluate_stress_benchmark(episodes=args.episodes)
    t3_data = evaluate_history_ablation(episodes=args.episodes)
    t4_data = evaluate_dr_ablation(episodes=args.episodes)

    tex_content = []
    tex_content.append("% ===========================================================")
    tex_content.append("% 필요 패키지: \\usepackage{booktabs}, \\usepackage{multirow}")
    tex_content.append("% ===========================================================\n")

    t1_tex = format_latex_table_3way(t1_data)
    t2_tex = format_latex_table_stress(t2_data)
    t3_tex = format_latex_table_history(t3_data)
    t4_tex = format_latex_table_dr(t4_data)

    tex_content.extend([t1_tex, "\n", t2_tex, "\n", t3_tex, "\n", t4_tex])
    full_tex = "\n".join(tex_content)

    os.makedirs(os.path.dirname(OUTPUT_TEX_PATH), exist_ok=True)
    with open(OUTPUT_TEX_PATH, "w", encoding="utf-8") as f:
        f.write(full_tex)

    print("\n" + "=" * 70)
    print(f"[성공] LaTeX 표 코드가 저장되었습니다: {OUTPUT_TEX_PATH}")
    print("=" * 70)
    print("\n--- [생성된 LaTeX 코드 콘솔 출력] ---\n")
    print(full_tex)


if __name__ == "__main__":
    main()