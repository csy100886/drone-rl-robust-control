# run_all.py
import os
import sys
import time
import argparse
import subprocess


def print_banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def run_command(cmd_list, description):
    """현재 가상환경의 파이썬 인터프리터로 서브프로세스 실행"""
    start_time = time.time()
    print(f"\n>> [진행 중] {description}...")
    print(f"   명령어: {' '.join(cmd_list)}")

    result = subprocess.run(cmd_list)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[오류 발생] {description} 단계에서 실패했습니다. (반환 코드: {result.returncode})")
        sys.exit(result.returncode)
    else:
        print(f">> [완료] {description} (소요 시간: {elapsed:.1f}초)")


def main():
    parser = argparse.ArgumentParser(description="Drone RL Graduation Thesis - End-to-End Pipeline Runner")
    parser.add_argument("--skip-train", action="store_true", help="기존 저장된 가중치를 사용하여 학습 단계를 건너뜀")
    parser.add_argument("--timesteps", type=int, default=400_000, help="모델당 학습 스텝 수")
    parser.add_argument("--eval-episodes", type=int, default=20, help="벤치마크 평가 에피소드 수")
    parser.add_argument("--visualize", action="store_true", help="모든 평가 완료 후 3D 실시간 애니메이션 실행")
    args = parser.parse_args()

    python_bin = sys.executable
    total_start = time.time()

    print_banner("드론 심층 강화학습(PPO) 연구 파이프라인 자동화")
    print(f"실행 인터프리터 : {python_bin}")
    print(f"학습 건너뛰기   : {args.skip_train}")
    print(f"평가 에피소드 수: {args.eval_episodes}회")

    # -------------------------------------------------------------
    # [Stage 1] 가상 환경 및 물리 연산 무결성 검증
    # -------------------------------------------------------------
    print_banner("Stage 1. 가상 환경 및 물리 연산 무결성 검증")
    run_command([python_bin, "test_env.py"], "6-DoF 환경 및 관찰 이력 적층 검증")

    # -------------------------------------------------------------
    # [Stage 2] 핵심 제어 모델 학습 (Vanilla PPO & DR-PPO)
    # -------------------------------------------------------------
    print_banner("Stage 2. 강화학습 제어기 학습")
    vanilla_ckpt = "./models/best_model/best_model.zip"
    dr_ckpt = "./models/dr_best_model/best_model.zip"

    if args.skip_train:
        print(">> [--skip-train] 옵션에 의해 모델 학습을 건너뜁니다.")
    else:
        # Vanilla PPO 학습 (가중치가 없거나 재학습 필요 시)
        if not os.path.exists(vanilla_ckpt):
            run_command([python_bin, "train.py"], "Vanilla PPO 모델 학습")
        else:
            print(f">> [확인] Vanilla PPO 가중치 기보유: {vanilla_ckpt}")

        # DR-PPO 학습
        if not os.path.exists(dr_ckpt):
            run_command([python_bin, "train_dr.py"], "DR-PPO (도메인 무작위화) 모델 학습")
        else:
            print(f">> [확인] DR-PPO 가중치 기보유: {dr_ckpt}")

    # -------------------------------------------------------------
    # [Stage 3] 핵심 성능 벤치마크 평가 및 그래프 생성
    # -------------------------------------------------------------
    print_banner("Stage 3. 제어기 성능 벤치마크 및 시각화 플롯 생성")
    
    # 3.1 PID vs Vanilla vs DR-PPO 3자 비교
    run_command([python_bin, "benchmark_3way.py"], "3자 제어기(PID vs Vanilla vs DR-PPO) 벤치마크")

    # 3.2 극한 스트레스 테스트 (Mass +30%, Delay 50ms)
    run_command([python_bin, "benchmark_stress.py"], "극한 스트레스 조건(질량 +30%, 50ms 지연) 벤치마크")

    # 3.3 풍속 감도 분석 (2, 5, 8 m/s)
    run_command([python_bin, "benchmark_wind_sensitivity.py"], "풍속 감도(2/5/8 m/s) 벤치마크")

    # -------------------------------------------------------------
    # [Stage 4] 소거 실험 (Ablation Studies)
    # -------------------------------------------------------------
    print_banner("Stage 4. 제어 메커니즘 소거 실험 (Ablation Studies)")
    
    # 4.1 과거 관찰 이력 크기(K=0, 2, 4) 소거 실험
    k_eval_flag = ["--eval_only"] if args.skip_train else []
    run_command([python_bin, "ablation_history.py"] + k_eval_flag, "관찰 이력 버퍼(K=0, 2, 4) 소거 실험")

    # 4.2 도메인 무작위화 파라미터(질량/지연) 소거 실험
    dr_eval_flag = ["--eval_only"] if args.skip_train else []
    run_command([python_bin, "ablation_dr.py"] + dr_eval_flag, "도메인 무작위화 파라미터 소거 실험")

    # -------------------------------------------------------------
    # [Stage 5] 논문용 LaTeX 표 자동 생성
    # -------------------------------------------------------------
    print_banner("Stage 5. 학부 졸업논문 제출용 LaTeX 표 취합 및 생성")
    run_command([python_bin, "generate_latex_tables.py", "--episodes", str(args.eval_episodes)], "LaTeX 통합 결과 표 생성")

    # -------------------------------------------------------------
    # [선택] 3D 비행 시각화 애니메이션 재생
    # -------------------------------------------------------------
    if args.visualize:
        print_banner("선택 실행. 3D 실시간 비행 애니메이션")
        run_command([python_bin, "visualizer.py"], "3D 실시간 비행 및 외란 복원 애니메이션")

    # -------------------------------------------------------------
    # 최종 결과 요약
    # -------------------------------------------------------------
    total_elapsed = time.time() - total_start
    print_banner("전체 파이프라인 수행 완료")
    print(f"총 소요 시간: {total_elapsed / 60:.1f}분 ({total_elapsed:.1f}초)")
    print("\n[생성된 논문 결과물 목록]")
    print(f"1. LaTeX 표 파일 : ./results/thesis_tables.tex")
    print(f"2. 결과 그래프들 : ./results/*.png")
    
    results_dir = "./results"
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith(".png")]
        for i, f in enumerate(files, 1):
            print(f"   ({i}) {f}")
    print("=" * 75)


if __name__ == "__main__":
    main()