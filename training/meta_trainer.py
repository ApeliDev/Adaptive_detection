import torch
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from typing import List, Dict, Any
import random
from core.marl_env import CarlaMARLEnv

class MetaPPOTrainer:
    def __init__(self, env_config: Dict, meta_config: Dict):
        self.env = DummyVecEnv([lambda: CarlaMARLEnv(env_config)])
        self.env = VecMonitor(self.env)
        
        # Meta-learning parameters
        self.meta_batch_size = meta_config.get('meta_batch_size', 16)
        self.inner_lr = meta_config.get('inner_lr', 1e-4)
        self.outer_lr = meta_config.get('outer_lr', 3e-5)
        self.inner_steps = meta_config.get('inner_steps', 5)
        
        # Initialize policy
        self.policy = PPO(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            learning_rate=self.inner_lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./logs/"
        )
        
        # Outer optimizer
        self.outer_optimizer = optim.Adam(self.policy.parameters(), lr=self.outer_lr)
        
        # Setup callbacks
        self.callbacks = [
            EvalCallback(
                self.env,
                best_model_save_path="./models/",
                log_path="./logs/",
                eval_freq=1000,
                deterministic=True,
                render=False
            )
        ]
    
    def meta_update(self, episodes: List[Dict]):
        """Perform meta-learning update"""
        # Inner loop adaptation
        for episode in episodes[:self.inner_steps]:
            # Train on episode
            self.policy.learn(
                total_timesteps=len(episode['observations']),
                callback=self.callbacks,
                reset_num_timesteps=False
            )
        
        # Outer loop update
        self.outer_optimizer.zero_grad()
        
        # Compute meta-loss across all episodes
        meta_loss = 0.0
        for episode in episodes[self.inner_steps:self.meta_batch_size]:
            # Evaluate adapted policy on new episodes
            obs = torch.tensor(episode['observations'])
            actions, values, log_probs = self.policy.policy(obs)
            
            # Compute advantage
            rewards = torch.tensor(episode['rewards'])
            advantages = rewards - values
            
            # Compute policy gradient loss
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, rewards)
            entropy_loss = -self.policy.ent_coef * log_probs.mean()
            
            meta_loss += policy_loss + value_loss + entropy_loss
        
        # Backpropagate through inner loop
        meta_loss.backward()
        self.outer_optimizer.step()
    
    def train(self, total_timesteps: int):
        """Main training loop"""
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            tb_log_name="meta_ppo"
        )

        