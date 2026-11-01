# config.py
import numpy as np

# 시간 및 제어 주기
CONTROL_FREQ = 50.0                # 제어 주기 50Hz (dt = 0.02s)
SIM_DT = 0.005                     # 물리 적분 주기 (200Hz, 서브스텝 4회)
DT = 1.0 / CONTROL_FREQ
MAX_STEPS = 500                    # 에피소드당 최대 스텝 (10초 비행)

# 물리 제원 (Crazyflie 2.1 실측치 기준)
MASS = 0.033                       # 기체 질량 (kg)
ARM_LENGTH = 0.046                 # 중심-모터 거리 (m)
I_XX = 1.4e-5                      # 관성 모멘트 X (kg*m^2)
I_YY = 1.4e-5                      # 관성 모멘트 Y (kg*m^2)
I_ZZ = 2.17e-5                     # 관성 모멘트 Z (kg*m^2)
GRAVITY = 9.81                     # 중력가속도 (m/s^2)

# 모터 제원
MAX_THRUST_TOTAL = 0.60            # 모터 4개 최대 총 추력 (N)
HOVER_THRUST = MASS * GRAVITY      # 호버링 필요 추력 (~0.323 N)
MOTOR_TAU = 0.02                   # 모터 1차 지연 시상수 (s)

# 외란 파라미터
OU_THETA = 1.0                     # 난류 평균 회귀 속도
OU_SIGMA = 0.6                     # 난류 변동 강도 (풍속 노이즈)
STEP_GUST_MAGNITUDE = np.array([4.0, 0.0, 0.0], dtype=np.float32)  # 계단형 돌풍 세기 (m/s)

# 관찰 및 한계 조건
HISTORY_LENGTH = 4                 # 관찰 이력 적층 수 (K=4, 총 5프레임 적층)
MAX_TILT_ANGLE = np.deg2rad(45.0)  # 추락 판정 임계 각도 (45도)