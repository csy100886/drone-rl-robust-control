# utils/domain_randomizer.py
import numpy as np
import config

class DomainRandomizer:
    """
    쿼드로터 물리 제원 및 액추에이터 특성을 에피소드마다 무작위화
    - 질량: 기준값 대비 ±20%
    - 모터 1차 지연 시상수: 0.015s ~ 0.040s (기준값 0.020s)
    - 관성 모멘트: 질량 변화율에 비례하여 동적 스케일링 (±15% 추가 노이즈)
    - 공기 저항 계수: 0.05 ~ 0.20
    """
    def __init__(self,
                 mass_range=(0.80, 1.20),
                 tau_range=(0.015, 0.040),
                 drag_range=(0.05, 0.20)):
        self.mass_range = mass_range
        self.tau_range = tau_range
        self.drag_range = drag_range

    def sample_parameters(self):
        # 1. 질량 샘플링
        mass_scale = np.random.uniform(self.mass_range[0], self.mass_range[1])
        random_mass = float(config.MASS * mass_scale)

        # 2. 관성 모멘트 (I = m * r^2 특성을 반영하여 질량 스케일에 연동 + 미세 편차)
        inertia_noise = np.random.uniform(0.85, 1.15, size=3)
        random_I = np.array([
            config.I_XX * mass_scale * inertia_noise[0],
            config.I_YY * mass_scale * inertia_noise[1],
            config.I_ZZ * mass_scale * inertia_noise[2]
        ], dtype=np.float32)

        # 3. 모터 시상수 (응답 지연 시간)
        random_tau = float(np.random.uniform(self.tau_range[0], self.tau_range[1]))

        # 4. 공기 항력 계수
        random_drag = float(np.random.uniform(self.drag_range[0], self.drag_range[1]))

        return {
            "mass": random_mass,
            "inertia": random_I,
            "motor_tau": random_tau,
            "drag_coeff": random_drag
        }