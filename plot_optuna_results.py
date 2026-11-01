# plot_optuna_results.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances
import joblib


DB_URL = "sqlite:///results/optuna_study.db"
PKL_PATH = "./results/optuna_study.pkl"
SAVE_PATH = "./results/optuna_optimization_analysis.png"


def load_study_safe():
    """SQLite DB 또는 Pickle 파일로부터 Optuna study 객체 로드"""
    if os.path.exists("./results/optuna_study.db"):
        print(f"[Info] SQLite DB로부터 스터디를 로드합니다: {DB_URL}")
        try:
            return optuna.load_study(study_name="quadrotor_reward_tuning", storage=DB_URL)
        except Exception as e:
            print(f"[경고] SQLite 로드 실패 ({e}), PKL 대체를 시도합니다.")

    if os.path.exists(PKL_PATH):
        print(f"[Info] Pickle 파일로부터 스터디를 로드합니다: {PKL_PATH}")
        return joblib.load(PKL_PATH)

    raise FileNotFoundError(
        "Optuna 탐색 결과 파일이 존재하지 않습니다.\n"
        "1) optimize_reward.py를 실행했는지 확인하세요.\n"
        "2) ./results/optuna_study.db 또는 ./results/optuna_study.pkl 파일이 필요합니다."
    )


def plot_optuna_analysis(study):
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) == 0:
        raise ValueError("완료된 Trial 데이터가 존재하지 않습니다.")

    trial_numbers = [t.number for t in completed_trials]
    trial_values = [t.value for t in completed_trials]
    running_best = np.minimum.accumulate(trial_values)

    best_trial = study.best_trial
    importances = get_param_importances(study)

    # 3분할 종합 분석 플롯 생성 (1행 3열)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # ---------------------------------------------------------
    # 1. 최적화 수렴 과정 (Convergence History)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.scatter(trial_numbers, trial_values, color='gray', alpha=0.6, edgecolors='black', s=45, label='Trial Cost')
    ax1.step(trial_numbers, running_best, color='crimson', where='post', lw=2.2, label='Best Cost Curve')
    ax1.scatter([best_trial.number], [best_trial.value], color='gold', marker='*', s=200, edgecolors='black', zorder=5, label=f'Best (#{best_trial.number})')
    
    ax1.set_title("Optimization Convergence History", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Trial Number", fontsize=11)
    ax1.set_ylabel("Composite Objective Cost", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)

    # ---------------------------------------------------------
    # 2. 보상 파라미터 중요도 (Parameter Importance)
    # ---------------------------------------------------------
    ax2 = axes[1]
    param_names = list(importances.keys())
    param_scores = [importances[k] * 100.0 for k in param_names]  # 백분율 변환

    # 점수 오름차순 정렬 (수평 막대 플롯용)
    sorted_idx = np.argsort(param_scores)
    y_pos = np.arange(len(param_names))

    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(param_names)))
    bars = ax2.barh(y_pos, [param_scores[i] for i in sorted_idx], color=colors, edgecolor='black', alpha=0.85)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([param_names[i] for i in sorted_idx], fontsize=10)
    ax2.set_title("Hyperparameter Importance (%)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Relative Importance (%)", fontsize=11)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{width:.1f}%",
                 va='center', ha='left', fontsize=9, fontweight='bold')

    # ---------------------------------------------------------
    # 3. RMSE vs Jitter 파레토 트레이드오프 (Pareto Front)
    # ---------------------------------------------------------
    ax3 = axes[2]
    rmses = [t.user_attrs.get("mean_rmse", np.nan) for t in completed_trials]
    jitters = [t.user_attrs.get("mean_jitter", np.nan) for t in completed_trials]

    scatter = ax3.scatter(rmses, jitters, c=trial_values, cmap='viridis_r', s=60, edgecolors='black', alpha=0.85)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Composite Cost", fontsize=10)

    # 최적 모델 강조
    b_rmse = best_trial.user_attrs.get("mean_rmse", None)
    b_jit = best_trial.user_attrs.get("mean_jitter", None)
    if b_rmse and b_jit:
        ax3.scatter([b_rmse], [b_jit], color='crimson', marker='*', s=220, edgecolors='black', zorder=6, label=f'Best Trial (#{best_trial.number})')

    ax3.set_title("RMSE vs Jittering Trade-off", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Position Tracking RMSE (m)", fontsize=11)
    ax3.set_ylabel("Control Action Jitter", fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"\n[성공] Optuna 최적화 분석 플롯이 저장되었습니다: {SAVE_PATH}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Optuna Study Results")
    parser.add_argument("--save-path", type=str, default=SAVE_PATH, help="플롯 이미지 저장 경로")
    args = parser.parse_args()

    print("=" * 70)
    print("Optuna 보상 함수 자율 최적화 결과 시각화 분석")
    print("=" * 70)

    study = load_study_safe()
    print(f">> 총 Trial 수 : {len(study.trials)}회")
    print(f">> 최적 Trial 번호 : #{study.best_trial.number}")
    print(f">> 최저 목적 Cost  : {study.best_value:.4f}")

    plot_optuna_analysis(study)


if __name__ == "__main__":
    main()