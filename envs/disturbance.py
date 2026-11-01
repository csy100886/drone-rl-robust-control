# envs/disturbance.py
import numpy as np
import config

class WindDisturbance:
    """대기 난류(Ornstein-Uhlenbeck) 및 계단형 돌풍(Step Gust) 생성기"""
    def __init__(self, dt=config.DT):
        self.dt = dt
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(3, dtype=np.float32)

    def step_turbulence(self):
        """dW_t = theta * (mean - W_t)dt + sigma * sqrt(dt) * N(0, 1)"""
        noise = np.random.normal(size=3).astype(np.float32)
        dx = config.OU_THETA * (-self.state) * self.dt + \
             config.OU_SIGMA * np.sqrt(self.dt) * noise
        self.state += dx
        return self.state.copy()

    def get_step_gust(self, current_time, start_time=3.0, duration=0.8):
        """특정 시점에 유입되는 순간 돌풍"""
        if start_time <= current_time <= (start_time + duration):
            return config.STEP_GUST_MAGNITUDE.copy()
        return np.zeros(3, dtype=np.float32)