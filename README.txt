================================================================================
  자율 비행 쿼드로터 강인 제어 연구 프로젝트 (DR-PPO vs PID) 실행 가이드
  파일명: README.txt
  작성 기준: C:\Users\csy10\drone_rl_project (Windows CMD / Python 3.10)
================================================================================

[목차]
1. 가상환경 활성화 및 필수 패키지 관리
2. 시뮬레이션 환경 무결성 검증
3. Optuna 기반 보상 함수 자율 최적화 및 분석
4. 모델 학습 및 기존 가중치 관리 (단일 / 순차 백업 학습)
5. 단일 모델 비행 안정성 정밀 평가 (3D 궤적, 자세, 모터 파형)
6. 전체 벤치마크 평가 및 논문용 산출물(LaTeX/플롯) 일괄 생성
7. 학습 실시간 모니터링 (TensorBoard)

--------------------------------------------------------------------------------
1. 가상환경 활성화 및 필수 패키지 관리
--------------------------------------------------------------------------------
# [상황] 새 터미널 창을 열었을 때 가장 먼저 가상환경을 활성화
# [설명] 프로젝트 전용 Python 패키지들이 설치된 독립 가상환경 진입
drone_rl_env\Scripts\activate

# [상황] 보상 가중치 베이지안 최적화 스크립트 실행 전 패키지가 없을 때
# [설명] TPE 기반 하이퍼파라미터 최적화 프레임워크 설치
pip install optuna

# [상황] Optuna study 객체 직렬화(저장/불러오기) 모듈 누락 오류 발생 시
# [설명] 파이썬 객체 저장 및 고속 로드를 위한 직렬화 라이브러리 설치
pip install joblib

# [상황] PC의 PyTorch가 GPU(CUDA)를 지원하는지 확인할 때
# [설명] CUDA 가속 지원 여부(True/False)를 콘솔에 즉시 출력
python -c "import torch; print('CUDA 사용 가능 여부:', torch.cuda.is_available())"

# [상황] NVIDIA 외장 그래픽 카드를 활용한 CUDA 가속 학습 환경을 구성할 때 (선택사항)
# [설명] 기존 CPU 버전 PyTorch 제거 후 CUDA 12.1 전용 PyTorch 재설치
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121


--------------------------------------------------------------------------------
2. 시뮬레이션 환경 무결성 검증
--------------------------------------------------------------------------------
# [상황] 프로젝트 파일 구조나 필수 모듈 누락 여부를 전체적으로 점검할 때
# [설명] 18개 이상의 핵심 소스 코드 및 설정 파일 존재 여부 일괄 검증
python check_files.py

# [상황] 보상 함수(_compute_reward)나 환경 코드를 수정한 직후 오류 여부를 확인할 때
# [설명] 쿼드로터 환경을 100스텝 구동하여 관찰 벡터(90차원) 및 물리 적분 정상 동작 확인
python test_env.py


--------------------------------------------------------------------------------
3. Optuna 기반 보상 함수 자율 최적화 및 분석
--------------------------------------------------------------------------------
# [상황] RMSE와 모터 Jittering을 동시에 최소화하는 최적 보상 가중치를 탐색할 때
# [설명] 15회 Trial 동안 10만 스텝 단축 학습 및 6m/s 돌풍 검증을 반복하여 최적 가중치 JSON 도출
python optimize_reward.py --n-trials 15 --timesteps 100000

# [상황] Optuna 탐색 완료 후 수렴 곡선과 파라미터 중요도를 논문용 그래프로 뽑을 때
# [설명] optuna_study.db 데이터를 읽어 3분할 고해상도 분석 이미지(results/optuna_optimization_analysis.png) 생성
python plot_optuna_results.py


--------------------------------------------------------------------------------
4. 모델 학습 및 기존 가중치 관리 (단일 / 순차 백업 학습)
--------------------------------------------------------------------------------
# [상황] 현재 models 폴더 내에 저장된 기존 모델 폴더 및 백업 목록을 확인할 때
# [설명] best_model, dr_best_model 등 저장 디렉터리 구조 조회
dir models

# [상황] 대조군인 Vanilla PPO 모델(도메인 무작위화 미적용)만 단독 학습할 때
# [설명] 공칭 물리 파라미터 고정 환경에서 50만 스텝 학습 진행
python train.py

# [상황] 제안 기법인 DR-PPO 모델(도메인 무작위화 적용)만 단독 학습할 때
# [설명] 질량(±20%), 모터지연(15~40ms), 관성 변동 환경에서 50만 스텝 학습 진행
python train_dr.py

# [상황] [★가장 추천★] 새로운 보상 함수로 Vanilla와 DR-PPO를 처음부터 안전하게 다시 학습할 때
# [설명] 기존 가중치 폴더를 *_old로 백업한 후 Vanilla PPO 완료 즉시 DR-PPO 순차 자동 학습 (약 25~35분 소요)
if exist models\best_model move /y models\best_model models\best_model_old && if exist models\dr_best_model move /y models\dr_best_model models\dr_best_model_old && python train.py && python train_dr.py


--------------------------------------------------------------------------------
5. 단일 모델 비행 안정성 정밀 평가 (3D 궤적, 자세, 모터 파형)
--------------------------------------------------------------------------------
# [상황] 학습된 최신 모델의 3D 궤적, RPY 자세 복원, 모터 제어 입력 곡선을 확인할 때
# [설명] 돌풍 타격 시 단일 에이전트 상세 응답 플롯(results/evaluation_plots.png) 생성
python evaluate.py

# [상황] 특정 가중치 파일(예: DR-PPO 최적 체크포인트)을 지정하여 평가할 때
# [설명] 경로를 직접 전달하여 지정된 에이전트의 제어 성능 평가 수행
python evaluate.py --model-path ./models/dr_best_model/best_model.zip


--------------------------------------------------------------------------------
6. 전체 벤치마크 평가 및 논문용 산출물(LaTeX/플롯) 일괄 생성
--------------------------------------------------------------------------------
# [상황] [★논문 작성 시 일상적 사용★] 학습된 가중치로 모든 벤치마크 플롯과 LaTeX 표를 뽑을 때
# [설명] 모델 학습을 건너뛰고 3자 비교, 스트레스, 풍속 감도, 소거 실험 및 thesis_tables.tex 일괄 갱신 (약 2분 소요)
python run_all.py --skip-train

# [상황] 벤치마크 수행 완료 후 3D 비행 시각화 GUI 창까지 바로 띄워 확인할 때
# [설명] 고속 평가 수행 후 돌풍 타격 시 기체 거동을 실시간 3D 화면으로 재생
python run_all.py --skip-train --visualize

# [상황] 모델 학습부터 전체 소거 실험, LaTeX 표 생성까지 완전 자동 무인 실행할 때
# [설명] 7개 모델 전체 훈련 및 모든 벤치마크/플롯 완전 생성 (약 40~50분 소요)
python run_all.py

# [단독 벤치마크 실행 명령어 모음 (개별 디버깅용)]
# PID vs Vanilla PPO vs DR-PPO 3자 비교 플롯 생성
python benchmark_3way.py

# 질량 +30%, 모터 지연 50ms 극한 환경 평가 플롯 생성
python benchmark_stress.py

# 2.0 / 5.0 / 8.0 m/s 강풍 감도 분석 플롯 생성
python benchmark_wind_sensitivity.py

# 이력 버퍼 크기(K=0, 2, 4) 성능 소거 실험 수행
python ablation_history.py

# DR 물리 파라미터(Mass, Tau) 개별 기여도 소거 실험 수행
python ablation_dr.py

# 취합된 결과를 논문용 Overleaf 양식 LaTeX 표(.tex)로 추출
python generate_latex_tables.py


--------------------------------------------------------------------------------
7. 학습 실시간 모니터링 (TensorBoard)
--------------------------------------------------------------------------------
# [상황] 모델 훈련 중 에피소드 평균 보상(ep_rew_mean)과 손실(loss) 수렴 추이를 웹으로 볼 때
# [설명] 텐서보드 로컬 서버 실행 후 웹 브라우저(http://localhost:6006)에서 그래프 확인
tensorboard --logdir=./logs


--------------------------------------------------------------------------------
8. 신규 팀원 환경 세팅 및 빠른 시작 가이드 (Git Clone 후 실행)
--------------------------------------------------------------------------------
# [Step 1] GitHub 원격 저장소 클론 및 디렉터리 이동
git clone https://github.com/<본인-GitHub-아이디>/<저장소명>.git
cd <저장소명>

# [Step 2] 가상환경 생성 및 활성화 (Windows CMD 기준)
python -m venv drone_rl_env
drone_rl_env\Scripts\activate

# [Step 3] 프로젝트 의존성 라이브러리 일괄 설치
pip install -r requirements.txt

# [Step 4] 시뮬레이션 환경 무결성 1초 검증
# 출력: '환경 구동 성공!' 확인
python test_env.py

# [Step 5] 사전 학습된 최적 모델 기반 전체 벤치마크 및 논문 플롯 일괄 생성
# 별도 모델 재학습 없이 results/ 내 그래프 및 LaTeX 표가 즉시 생성됨 (약 2분 소요)
python run_all.py --skip-train

# (선택) 3D 실시간 비행 시각화 GUI 창 확인
python run_all.py --skip-train --visualize