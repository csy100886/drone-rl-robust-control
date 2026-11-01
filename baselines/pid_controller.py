# baselines/pid_controller.py
import numpy as np
from scipy.spatial.transform import Rotation as R
import config

class CascadedPIDController:
    """
    쿼드로터 6자유도 다단(Cascaded) PID 제어기
    - 외부 루프: 3차원 위치(XYZ) 오차 -> 목표 가속도 -> 목표 Roll/Pitch 자세각 & 총 추력
    - 내부 루프: 자세각(RPY) 오차 및 각속도 감쇠 -> 3축 모멘트(Torque) 도출
    - 모터 분배기(Mixer): 4개 모터의 개별 추력 계산 및 정규화 [-1, 1] 액션 변환
    """
    def __init__(self, dt=config.DT):
        self.dt = dt

        # 1. 위치 루프(Position Loop) 게인
        self.kp_pos_xy = 2.0
        self.kd_pos_xy = 1.6
        self.ki_pos_xy = 0.05

        self.kp_pos_z = 10.0
        self.kd_pos_z = 5.0
        self.ki_pos_z = 2.0

        # 2. 자세 루프(Attitude Loop) 게인
        self.kp_att_rp = 8.0
        self.kd_att_rp = 0.8
        self.ki_att_rp = 0.1

        self.kp_att_yaw = 3.0
        self.kd_att_yaw = 0.4

        # 적분 오차 버퍼
        self.reset()

    def reset(self):
        self.pos_integral = np.zeros(3, dtype=np.float32)
        self.att_integral = np.zeros(3, dtype=np.float32)

    def compute_action(self, state, target_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
        """
        QuadrotorEnv의 state 딕셔너리를 받아 정규화된 [-1, 1] 모터 액션 4개 반환
        """
        pos = state["pos"]
        vel = state["vel"]
        rot_mat = state["rot"]
        ang_vel = state["ang_vel"]

        # 현재 오일러 각도 (Roll, Pitch, Yaw)
        rpy = R.from_matrix(rot_mat).as_euler('xyz')

        # ---------------------------------------------------------
        # [Step 1] 외부 위치 제어 루프 (Position -> Desired Accel)
        # ---------------------------------------------------------
        pos_err = target_pos - pos
        self.pos_integral += pos_err * self.dt
        # 적분 윈드업(Windup) 방지
        self.pos_integral = np.clip(self.pos_integral, -1.0, 1.0)

        # 수평 가속도 명령 (X, Y)
        a_x = self.kp_pos_xy * pos_err[0] - self.kd_pos_xy * vel[0] + self.ki_pos_xy * self.pos_integral[0]
        a_y = self.kp_pos_xy * pos_err[1] - self.kd_pos_xy * vel[1] + self.ki_pos_xy * self.pos_integral[1]

        # 수직 가속도 명령 (Z) + 중력 보상
        a_z = self.kp_pos_z * pos_err[2] - self.kd_pos_z * vel[2] + self.ki_pos_z * self.pos_integral[2] + config.GRAVITY

        # 소각 근사(Small Angle Approximation) 기반 목표 Roll/Pitch 자세각 도출
        # X 가속도는 양의 Pitch, Y 가속도는 음의 Roll을 유발
        pitch_des = np.clip(a_x / config.GRAVITY, -config.MAX_TILT_ANGLE, config.MAX_TILT_ANGLE)
        roll_des = np.clip(-a_y / config.GRAVITY, -config.MAX_TILT_ANGLE, config.MAX_TILT_ANGLE)
        yaw_des = 0.0

        # 총 요구 추력 (기체 Z축 투영 보정)
        total_thrust_des = config.MASS * a_z / max(rot_mat[2, 2], 0.3)
        total_thrust_des = np.clip(total_thrust_des, 0.0, config.MAX_THRUST_TOTAL)

        # ---------------------------------------------------------
        # [Step 2] 내부 자세 제어 루프 (Attitude Error -> Desired Torques)
        # ---------------------------------------------------------
        att_err = np.array([roll_des - rpy[0], pitch_des - rpy[1], yaw_des - rpy[2]], dtype=np.float32)
        self.att_integral += att_err * self.dt
        self.att_integral = np.clip(self.att_integral, -0.5, 0.5)

        # 축별 요구 제어 토크 (Body Frame)
        tau_x = config.I_XX * (self.kp_att_rp * att_err[0] - self.kd_att_rp * ang_vel[0] + self.ki_att_rp * self.att_integral[0])
        tau_y = config.I_YY * (self.kp_att_rp * att_err[1] - self.kd_att_rp * ang_vel[1] + self.ki_att_rp * self.att_integral[1])
        tau_z = config.I_ZZ * (self.kp_att_yaw * att_err[2] - self.kd_att_yaw * ang_vel[2])

        # ---------------------------------------------------------
        # [Step 3] 모터 믹서(Motor Mixer) 역변환
        # tau_x = L*(t1 - t3), tau_y = L*(t2 - t0), tau_z = 0.01*(t0 - t1 + t2 - t3)
        # ---------------------------------------------------------
        L = config.ARM_LENGTH
        c_tau = 0.01

        base_t = total_thrust_des / 4.0
        dt_x = tau_x / (2.0 * L)
        dt_y = tau_y / (2.0 * L)
        dt_z = tau_z / (4.0 * c_tau)

        t0 = base_t - dt_y + dt_z
        t1 = base_t + dt_x - dt_z
        t2 = base_t + dt_y + dt_z
        t3 = base_t - dt_x - dt_z

        thrusts = np.array([t0, t1, t2, t3], dtype=np.float32)
        thrusts = np.clip(thrusts, 0.0, config.MAX_THRUST_TOTAL / 4.0)

        # 모터 추력을 Gymnasium 액션 공간 [-1, 1] 범위로 정규화
        actions = (thrusts / (config.MAX_THRUST_TOTAL / 4.0)) * 2.0 - 1.0
        return np.clip(actions, -1.0, 1.0)