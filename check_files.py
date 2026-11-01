# check_files.py
import os
import py_compile

REQUIRED_FILES = [
    # Root
    "config.py",
    "test_env.py",
    "train.py",
    "train_dr.py",
    "evaluate.py",
    "benchmark_stress.py",
    "benchmark_wind_sensitivity.py",
    "benchmark_3way.py",
    "visualizer.py",
    "ablation_history.py",
    "ablation_dr.py",
    "generate_latex_tables.py",
    "run_all.py",
    # envs/
    "envs/__init__.py",
    "envs/disturbance.py",
    "envs/quadrotor_env.py",
    "envs/wrappers.py",
    # models/
    "models/ppo_agent.py",
    # baselines/
    "baselines/pid_controller.py",
    # utils/
    "utils/domain_randomizer.py",
    "utils/logger.py",
]

REQUIRED_DIRS = ["envs", "models", "baselines", "utils", "results", "logs"]


def main():
    print("=" * 70)
    print("드론 심층 강화학습 프로젝트 전체 파일 구성 및 문법 무결성 점검")
    print("=" * 70)

    # 1. 필수 디렉터리 점검
    print("\n[1] 필수 디렉터리 점검:")
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"  [V] 디렉터리 확인: ./{d}/")

    # 2. 필수 소스코드 파일 존재 및 문법 검사
    print("\n[2] 필수 소스코드 파일 및 문법(Syntax) 검사:")
    missing_files = []
    syntax_error_files = []

    for rel_path in REQUIRED_FILES:
        if not os.path.exists(rel_path):
            print(f"  [X] 누락된 파일: {rel_path}")
            missing_files.append(rel_path)
        else:
            try:
                py_compile.compile(rel_path, doraise=True)
                print(f"  [V] 정상: {rel_path:<32} (문법 오류 없음)")
            except py_compile.PyCompileError as e:
                print(f"  [!] 문법 오류 발견: {rel_path}")
                print(f"      -> {e.msg}")
                syntax_error_files.append(rel_path)

    # 3. 결과 요약
    print("\n" + "=" * 70)
    total = len(REQUIRED_FILES)
    success_count = total - len(missing_files) - len(syntax_error_files)
    print(f"점검 결과: 총 {total}개 파일 중 정상 {success_count}개 / 누락 {len(missing_files)}개 / 문법 오류 {len(syntax_error_files)}개")

    if not missing_files and not syntax_error_files:
        print("\n>> 축하합니다! 모든 필수 파일이 완벽하게 구성되었으며 실행 준비가 완료되었습니다.")
        print(">> 이제 다음 명령어로 전체 파이프라인을 구동할 수 있습니다:")
        print("   python run_all.py")
    else:
        print("\n>> 조치가 필요한 파일이 존재합니다. 위의 [X] 또는 [!] 표시 항목을 확인하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()