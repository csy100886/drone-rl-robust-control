# envs/quadrotor_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
import config
from envs.disturbance import WindDisturbance
from utils.domain_randomizer import DomainRandomizer

class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, enable_dr=False):
        super().__init__()
        self.dt = config.DT
        self.substeps = int(config.DT / config.SIM_DT)
        self.disturbance = WindDisturbance(dt=self.dt)
        self.target_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # 도메인 무작위화 설정
        self.enable_dr = enable_dr
        self.randomizer = DomainRandomizer() if enable_dr else None

        # 기본 물리 제원 초기화
        self.mass = config.MASS
        self.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32)
        self.motor_tau = config.MOTOR_TAU
        self.drag_coeff = 0.1

        # 행동/관찰 공간
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

        self.state = None
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.disturbance.reset()
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        # DR 활성화 시 에피소드마다 물리 파라미터 무작위 샘플링
        if self.enable_dr and self.randomizer is not None:
            params = self.randomizer.sample_parameters()
            self.mass = params["mass"]
            self.I = params["inertia"]
            self.motor_tau = params["motor_tau"]
            self.drag_coeff = params["drag_coeff"]
        else:
            self.mass = config.MASS
            self.I = np.array([config.I_XX, config.I_YY, config.I_ZZ], dtype=np.float32)
            self.motor_tau = config.MOTOR_TAU
            self.drag_coeff = 0.1

        pos = self.target_pos + np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
        vel = np.zeros(3, dtype=np.float32)
        rpy = np.random.uniform(-0.02, 0.02, size=3).astype(np.float32)
        rot = R.from_euler('xyz', rpy).as_matrix().astype(np.float32)
        ang_vel = np.zeros(3, dtype=np.float32)

        self.state = {
            "pos": pos,
            "vel": vel,
            "rot": rot,
            "ang_vel": ang_vel,
            "actual_motor_thrust": np.ones(4, dtype=np.float32) * ((self.mass * config.GRAVITY) / 4.0)
        }
        return self._get_obs(), {}

    def _get_obs(self):
        pos_err = self.target_pos - self.state["pos"]
        vel = self.state["vel"]
        rot_flat = self.state["rot"].flatten()
        ang_vel = self.state["ang_vel"]
        return np.concatenate([pos_err, vel, rot_flat, ang_vel], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        curr_time = self.step_count * self.dt

        target_thrust = (action + 1.0) * 0.5 * (config.MAX_THRUST_TOTAL / 4.0)

        wind = self.disturbance.step_turbulence() + self.disturbance.get_step_gust(curr_time)
        for _ in range(self.substeps):
            self._physics_step(target_thrust, wind, config.SIM_DT)

        obs = self._get_obs()
        reward = self._compute_reward(action)

        pos_err = np.linalg.norm(self.target_pos - self.state["pos"])
        tilt_angle = np.arccos(np.clip(self.state["rot"][2, 2], -1.0, 1.0))
        terminated = bool(tilt_angle > config.MAX_TILT_ANGLE or pos_err > 1.5)
        truncated = bool(self.step_count >= config.MAX_STEPS)

        self.prev_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def _physics_step(self, target_thrust, wind, dt):
        # 동적으로 설정된 motor_tau 적용
        alpha = dt / (self.motor_tau + dt)
        self.state["actual_motor_thrust"] += alpha * (target_thrust - self.state["actual_motor_thrust"])
        t = self.state["actual_motor_thrust"]

        total_thrust = np.sum(t)
        f_b = np.array([0.0, 0.0, total_thrust], dtype=np.float32)

        L = config.ARM_LENGTH
        tau_x = L * (t[1] - t[3])
        tau_y = L * (t[2] - t[0])
        tau_z = 0.01 * (t[0] - t[1] + t[2] - t[3])
        torque_b = np.array([tau_x, tau_y, tau_z], dtype=np.float32)

        # 동적으로 설정된 mass 및 drag_coeff 적용
        acc = (self.state["rot"] @ f_b) / self.mass - np.array([0.0, 0.0, config.GRAVITY])
        acc += self.drag_coeff * (wind - self.state["vel"])
        self.state["vel"] += acc * dt
        self.state["pos"] += self.state["vel"] * dt

        # 동적으로 설정된 관성 텐서(I) 적용
        I_mat = np.diag(self.I)
        w = self.state["ang_vel"]
        w_dot = np.linalg.inv(I_mat) @ (torque_b - np.cross(w, I_mat @ w))
        self.state["ang_vel"] += w_dot * dt

        w_skew = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        self.state["rot"] += (self.state["rot"] @ w_skew) * dt
        u, _, vt = np.linalg.svd(self.state["rot"])
        self.state["rot"] = u @ vt

    def _compute_reward(self, action):
            pos_err = np.linalg.norm(self.target_pos - self.state["pos"])
            vel_norm = np.linalg.norm(self.state["vel"])
            ang_vel_norm = np.linalg.norm(self.state["ang_vel"])

            # 1. 자세 오차: Roll/Pitch(70%) 및 Yaw(30%) 독립 정렬
            rpy = R.from_matrix(self.state["rot"]).as_euler('xyz')
            roll_pitch_err = np.sum(rpy[:2]**2)
            yaw_err = rpy[2]**2
            r_att = np.exp(-4.0 * roll_pitch_err) * 0.70 + np.exp(-2.0 * yaw_err) * 0.30

            # 2. 위치 수렴 지수 보상 (지수 3.0)
            r_pos = np.exp(-3.0 * pos_err**2)

            # 3. 속도/각속도 감쇠 (Damping)
            r_vel = np.exp(-0.5 * vel_norm**2)
            r_ang = np.exp(-0.5 * ang_vel_norm**2)

            # 4. 제어 평활화 (Jittering 억제: 가중치 0.40)
            action_diff = np.sum((action - self.prev_action)**2)
            action_mag = np.sum(action**2)
            r_smooth = 0.40 * action_diff + 0.06 * action_mag

            reward = (1.90 * r_pos) + (1.60 * r_att) + (0.4 * r_vel) + (0.3 * r_ang) - r_smooth + 0.5
            return float(reward)