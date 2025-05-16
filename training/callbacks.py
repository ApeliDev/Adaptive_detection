import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

class AdaptiveDetectionCallback(BaseCallback):
    """
    Custom callback for tracking adaptive detection performance metrics.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.detection_history = []
        self.switch_counts = {'yolo': 0, 'detr': 0, 'ensemble': 0}
        self.current_mode = None
        self.episode_metrics = {
            'mean_fps': [],
            'mean_mAP': [],
            'collisions': 0,
            'mode_switches': 0
        }

    def _on_step(self) -> bool:
        # Track detector switching
        if 'computational' in self.locals['infos'][0]:
            current_mode = self.locals['infos'][0]['computational']['detector_mode']
            if current_mode != self.current_mode:
                self.episode_metrics['mode_switches'] += 1
                self.current_mode = current_mode
                mode_name = {0: 'yolo', 1: 'detr', 2: 'ensemble'}[current_mode]
                self.switch_counts[mode_name] += 1

        # Track collisions
        if 'collision_occurred' in self.locals['infos'][0]:
            if self.locals['infos'][0]['collision_occurred']:
                self.episode_metrics['collisions'] += 1

        # Track FPS and mAP
        if 'perception' in self.locals['infos'][0]:
            self.episode_metrics['mean_fps'].append(
                self.locals['infos'][0]['perception']['fps']
            )
            self.episode_metrics['mean_mAP'].append(
                self.locals['infos'][0]['perception']['avg_confidence']
            )

        return True

    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout"""
        if len(self.episode_metrics['mean_fps']) > 0:
            self.logger.record(
                "rollout/mean_fps",
                np.mean(self.episode_metrics['mean_fps'])
            )
            self.logger.record(
                "rollout/mean_mAP",
                np.mean(self.episode_metrics['mean_mAP'])
            )
            self.logger.record(
                "rollout/collisions",
                self.episode_metrics['collisions']
            )
            self.logger.record(
                "rollout/mode_switches",
                self.episode_metrics['mode_switches']
            )
        
        # Reset episode metrics
        self.episode_metrics = {
            'mean_fps': [],
            'mean_mAP': [],
            'collisions': 0,
            'mode_switches': 0
        }

    def _on_training_end(self) -> None:
        """Save final detector switch counts"""
        self.logger.record(
            "train/yolo_usage",
            self.switch_counts['yolo']
        )
        self.logger.record(
            "train/detr_usage",
            self.switch_counts['detr']
        )
        self.logger.record(
            "train/ensemble_usage",
            self.switch_counts['ensemble']
        )

class VideoRecordCallback(EvalCallback):
    """
    Extended EvalCallback that records videos of evaluation episodes.
    """
    def __init__(
        self,
        eval_env: VecEnv,
        video_length: int = 100,
        n_eval_episodes: int = 3,
        **kwargs
    ):
        super().__init__(eval_env, **kwargs)
        self.video_length = video_length
        self.n_eval_episodes = n_eval_episodes
        self.video_frames = []
        self.current_episode = 0

    def _on_step(self) -> bool:
        if self.eval_env is not None and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if needed
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset video frames
            self.video_frames = []
            self.current_episode = 0
            
            # Run evaluation
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                callback=self._log_video_callback,
            )

            # Log video
            if len(self.video_frames) > 0:
                video = np.stack(self.video_frames)
                self.logger.record(
                    "eval/video",
                    Video(video, fps=20),
                    exclude=("stdout", "log", "json", "csv"),
                )

        return True

    def _log_video_callback(self, locals_: Dict, globals_: Dict) -> None:
        """Callback for collecting video frames during evaluation"""
        if 'rgb_array' in locals_ and self.current_episode < self.n_eval_episodes:
            if len(self.video_frames) < self.video_length:
                frame = locals_['rgb_array']
                if frame is not None:
                    self.video_frames.append(frame)
            else:
                self.current_episode += 1
                self.video_frames = []