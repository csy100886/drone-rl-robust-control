# visualizer.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO

import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper
from baselines.pid_controller import CascadedPIDController


# -------------------------------------------------------------
# 시각화 파라미터 설정
# -------------------------------------------------------------
TEST_WIND_SPEED = 7.5              # 돌풍 풍속 (m/s)
GUST_START = 3.0                   # 돌풍 시작 시점 (초)
GUST_DURATION = 0.8                # 돌풍 지속 시간 (초)
DRONE_ARM_VIS_LENGTH = 0.12        # 시각적 식별을 위해 기체 암 길이를 확대(0.12m)
TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def get_models():
    vanilla_path = "./models/best_model/best_model.zip"
    if not os.path.exists(vanilla_path):
        vanilla_path = "./models/ppo_quadrotor_final.zip"

    dr_path = "./models/dr_best_model/best_model.zip"
    if not os.path.exists(dr_path):
        dr_path = "./models/dr_model/ppo_dr_final.zip"

    if not os.path.exists(vanilla_path) or not os.path.exists(dr_path):
        raise FileNotFoundError("학습된 모델 파일(.zip)이 없습니다. train.py와 train_dr.py를 먼저 실행하세요.")

    return PPO.load(vanilla_path), PPO.load(dr_path)


def simulate_controller(controller_type, model_obj, seed=2026):
    """지정된 제어기의 전체 비행 시계열 데이터(위치, 회전행렬, 생존 상태) 사전 연산"""
    raw_env = QuadrotorEnv(enable_dr=False)
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    config.STEP_GUST_MAGNITUDE = np.array([TEST_WIND_SPEED, 0.0, 0.0], dtype=np.float32)

    pid = CascadedPIDController(dt=config.DT) if controller_type == 'pid' else None

    obs, _ = env.reset(seed=seed)
    if pid:
        pid.reset()

    positions = []
    rotations = []
    survived = True

    for step in range(config.MAX_STEPS):
        if survived:
            if controller_type == 'pid':
                action = pid.compute_action(raw_env.state, target_pos=TARGET_POS)
            else:
                action, _ = model_obj.predict(obs, deterministic=True)

            positions.append(raw_env.state["pos"].copy())
            rotations.append(raw_env.state["rot"].copy())

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated:
                survived = False
        else:
            # 추락 후에는 마지막 지점에 정지 유지
            positions.append(positions[-1].copy())
            rotations.append(rotations[-1].copy())

    return {
        "pos": np.array(positions),
        "rot": np.array(rotations),
        "survived": survived
    }


def get_drone_cross_points(pos, rot, arm_len=DRONE_ARM_VIS_LENGTH):
    """회전 행렬과 위치를 반영한 쿼드로터 십자(+) 프레임 좌표 계산"""
    # Body Frame 기준 암 끝단 모터 좌표 (X축: 0-2, Y축: 1-3)
    p_front = np.array([arm_len, 0.0, 0.0])
    p_back = np.array([-arm_len, 0.0, 0.0])
    p_left = np.array([0.0, arm_len, 0.0])
    p_right = np.array([0.0, -arm_len, 0.0])

    # World Frame 변환
    front_w = pos + rot @ p_front
    back_w = pos + rot @ p_back
    left_w = pos + rot @ p_left
    right_w = pos + rot @ p_right

    arm_x = np.array([back_w, front_w])
    arm_y = np.array([right_w, left_w])
    rotors = np.array([front_w, left_w, back_w, right_w])
    return arm_x, arm_y, rotors


def main():
    print("=" * 70)
    print("3자 제어기 비행 궤적 시뮬레이션 연산 중...")
    vanilla_model, dr_model = get_models()

    data_pid = simulate_controller('pid', None)
    data_vanilla = simulate_controller('vanilla', vanilla_model)
    data_dr = simulate_controller('dr', dr_model)
    print("시뮬레이션 완료. 3D 애니메이션 렌더링을 시작합니다.")
    print("=" * 70)

    # ---------------------------------------------------------
    # 3D 애니메이션 시각화 구성
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 스타일 설정
    colors = {
        'pid': 'forestgreen',
        'vanilla': 'crimson',
        'dr': 'royalblue'
    }

    # 궤적 라인 객체
    trail_pid, = ax.plot([], [], [], color=colors['pid'], lw=1.5, alpha=0.6, label='Cascaded PID')
    trail_van, = ax.plot([], [], [], color=colors['vanilla'], lw=1.5, alpha=0.6, label='Vanilla PPO')
    trail_dr, = ax.plot([], [], [], color=colors['dr'], lw=2.0, alpha=0.9, label='DR-PPO (Ours)')

    # 드론 프레임 객체 (십자선 2개 + 로터 4개)
    def create_drone_actors(color):
        line_x, = ax.plot([], [], [], color=color, lw=3)
        line_y, = ax.plot([], [], [], color=color, lw=3)
        rotors = ax.scatter([], [], [], color='black', s=25, depthshade=False)
        return line_x, line_y, rotors

    drone_actors = {
        'pid': create_drone_actors(colors['pid']),
        'vanilla': create_drone_actors(colors['vanilla']),
        'dr': create_drone_actors(colors['dr'])
    }

    # 목표 지점 및 기준면 표시
    ax.scatter([TARGET_POS[0]], [TARGET_POS[1]], [TARGET_POS[2]], color='gold', marker='*', s=120, label='Target [0, 0, 1.0]')

    # HUD 텍스트
    info_text = ax.text2D(0.03, 0.94, "", transform=ax.transAxes, fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.85))
    gust_text = ax.text2D(0.03, 0.86, "", transform=ax.transAxes, fontsize=12, fontweight='bold', color="red")

    # 뷰포트 범위 설정
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([0.0, 1.8])
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    ax.set_title(f"3D Flight & Attitude Recovery Animation (Gust: {TEST_WIND_SPEED} m/s)", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.view_init(elev=22, azim=-55)
    ax.grid(True)

    # ---------------------------------------------------------
    # 애니메이션 업데이트 함수 (프레임 스킵 적용하여 30 FPS 유지)
    # ---------------------------------------------------------
    step_skip = 2  # 50Hz 데이터에서 2스텝씩 건너뛰어 약 25~30 FPS 속도로 렌더링
    total_frames = config.MAX_STEPS // step_skip

    def update(frame):
        idx = min(frame * step_skip, config.MAX_STEPS - 1)
        curr_time = idx * config.DT

        # 1. 궤적(Trails) 갱신
        trail_pid.set_data(data_pid["pos"][:idx+1, 0], data_pid["pos"][:idx+1, 1])
        trail_pid.set_3d_properties(data_pid["pos"][:idx+1, 2])

        trail_van.set_data(data_vanilla["pos"][:idx+1, 0], data_vanilla["pos"][:idx+1, 1])
        trail_van.set_3d_properties(data_vanilla["pos"][:idx+1, 2])

        trail_dr.set_data(data_dr["pos"][:idx+1, 0], data_dr["pos"][:idx+1, 1])
        trail_dr.set_3d_properties(data_dr["pos"][:idx+1, 2])

        # 2. 드론 3D 프레임 자세 갱신
        for key, data in [('pid', data_pid), ('vanilla', data_vanilla), ('dr', data_dr)]:
            pos = data["pos"][idx]
            rot = data["rot"][idx]
            arm_x, arm_y, rotors = get_drone_cross_points(pos, rot)

            line_x, line_y, scat_rotors = drone_actors[key]
            line_x.set_data(arm_x[:, 0], arm_x[:, 1])
            line_x.set_3d_properties(arm_x[:, 2])

            line_y.set_data(arm_y[:, 0], arm_y[:, 1])
            line_y.set_3d_properties(arm_y[:, 2])

            scat_rotors._offsets3d = (rotors[:, 0], rotors[:, 1], rotors[:, 2])

        # 3. HUD 상태 표시 갱신
        info_text.set_text(
            f"Time: {curr_time:4.2f}s / 10.00s\n"
            f"PID Pos    : [{data_pid['pos'][idx, 0]:.2f}, {data_pid['pos'][idx, 1]:.2f}, {data_pid['pos'][idx, 2]:.2f}]\n"
            f"Vanilla Pos: [{data_vanilla['pos'][idx, 0]:.2f}, {data_vanilla['pos'][idx, 1]:.2f}, {data_vanilla['pos'][idx, 2]:.2f}]\n"
            f"DR-PPO Pos : [{data_dr['pos'][idx, 0]:.2f}, {data_dr['pos'][idx, 1]:.2f}, {data_dr['pos'][idx, 2]:.2f}]"
        )

        # 돌풍 인가 구간 (3.0s ~ 3.8s) 시각화
        if GUST_START <= curr_time <= (GUST_START + GUST_DURATION):
            gust_text.set_text(f">>> STEP GUST ACTIVE ({TEST_WIND_SPEED} m/s +X) <<<")
        else:
            gust_text.set_text("")

        return trail_pid, trail_van, trail_dr

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=35, blit=False, repeat=True
    )

    plt.show()


if __name__ == "__main__":
    main()