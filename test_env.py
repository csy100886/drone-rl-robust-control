# test_env.py
import config
from envs.quadrotor_env import QuadrotorEnv
from envs.wrappers import HistoryBufferWrapper

def main():
    raw_env = QuadrotorEnv()
    env = HistoryBufferWrapper(raw_env, history_length=config.HISTORY_LENGTH)
    
    obs, _ = env.reset()
    print("=" * 50)
    print(f"관찰 벡터 차원: {obs.shape} (기대값: (90,))")
    print(f"행동 벡터 차원: {env.action_space.shape} (기대값: (4,))")
    
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"에피소드 종료 Step: {step+1}")
            break
            
    print(f"환경 구동 성공! 100스텝 누적 보상: {total_reward:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()