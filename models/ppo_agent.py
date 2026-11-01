# models/ppo_agent.py
import torch
from stable_baselines3 import PPO


def create_ppo_agent(env, log_dir=None, tensorboard_log=None, learning_rate=3e-4, device="auto"):
    """
    프로젝트 전반에서 동일한 네트워크 아키텍처와 PPO 파라미터를 일관되게 생성하는 팩토리 함수
    """
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(
            pi=[256, 256, 128],  # Actor
            vf=[256, 256, 128]   # Critic
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=tensorboard_log,
        device=device
    )
    return model