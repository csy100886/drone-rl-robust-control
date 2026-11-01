# envs/wrappers.py
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HistoryBufferWrapper(gym.Wrapper):
    """최근 K개 프레임을 쌓아 (18 * (K+1)) 차원으로 확장하는 래퍼"""
    def __init__(self, env, history_length=4):
        super().__init__(env)
        self.history_length = history_length
        self.buffer = deque(maxlen=history_length + 1)
        
        orig_space = env.observation_space
        low = np.tile(orig_space.low, history_length + 1)
        high = np.tile(orig_space.high, history_length + 1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.buffer.clear()
        for _ in range(self.history_length + 1):
            self.buffer.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self):
        return np.concatenate(list(self.buffer), axis=0, dtype=np.float32)