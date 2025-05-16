import os
import yaml
import numpy as np
import torch
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from core.marl_env import CarlaMARLEnv
from training.callbacks import AdaptiveDetectionCallback

def evaluate_model(
    model_path: str,
    config_path: str,
    n_episodes: int = 10,
    render: bool = True,
    save_video: bool = False,
    video_length: int = 1000,
    video_name: str = "eval_video"
) -> Dict[str, float]:
    """
    Evaluate a trained model on the specified scenario.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to the scenario config file
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        save_video: Whether to save evaluation videos
        video_length: Maximum length of video
        video_name: Base name for saved videos
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create environment
    env = DummyVecEnv([lambda: CarlaMARLEnv(config)])
    
    # Wrap for video recording if needed
    if save_video:
        env = VecVideoRecorder(
            env,
            video_folder="videos/",
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=video_name
        )

    # Load model
    model = PPO.load(model_path, env=env)

    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'detection_metrics': {
            'mean_ap': [],
            'mean_fps': [],
            'mode_switches': []
        },
        'safety_metrics': {
            'collisions': 0,
            'lane_invasions': 0
        }
    }

    # Evaluation loop
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        # Initialize callback for this episode
        callback = AdaptiveDetectionCallback()
        callback.init_callback(model)
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward[0]
            episode_length += 1
            
            # Update callback
            callback.locals = {
                'infos': info,
                'done': done
            }
            callback._on_step()
            
            if render:
                env.render()

        # Store episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        
        # Get detection metrics from callback
        if len(callback.episode_metrics['mean_mAP']) > 0:
            metrics['detection_metrics']['mean_ap'].append(
                np.mean(callback.episode_metrics['mean_mAP'])
            )
            metrics['detection_metrics']['mean_fps'].append(
                np.mean(callback.episode_metrics['mean_fps'])
            )
            metrics['detection_metrics']['mode_switches'].append(
                callback.episode_metrics['mode_switches']
            )
        
        # Get safety metrics from env
        if 'collision_occurred' in info[0]:
            metrics['safety_metrics']['collisions'] += int(info[0]['collision_occurred'])
        if 'lane_invasion' in info[0]:
            metrics['safety_metrics']['lane_invasions'] += int(info[0]['lane_invasion'])

    # Compute aggregate metrics
    results = {
        'mean_reward': np.mean(metrics['episode_rewards']),
        'std_reward': np.std(metrics['episode_rewards']),
        'mean_episode_length': np.mean(metrics['episode_lengths']),
        'mean_mAP': np.mean(metrics['detection_metrics']['mean_ap']),
        'mean_fps': np.mean(metrics['detection_metrics']['mean_fps']),
        'mean_mode_switches': np.mean(metrics['detection_metrics']['mode_switches']),
        'total_collisions': metrics['safety_metrics']['collisions'],
        'total_lane_invasions': metrics['safety_metrics']['lane_invasions'],
        'success_rate': np.mean([
            1 if r > config['environment']['reward_config']['success_reward'] * 0.7 else 0 
            for r in metrics['episode_rewards']
        ])
    }

    # Print summary
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} steps")
    print(f"Detection mAP: {results['mean_mAP']:.3f}")
    print(f"Average FPS: {results['mean_fps']:.1f}")
    print(f"Average Mode Switches: {results['mean_mode_switches']:.1f}")
    print(f"Total Collisions: {results['total_collisions']}")
    print(f"Total Lane Invasions: {results['total_lane_invasions']}")
    print(f"Success Rate: {results['success_rate']:.1%}")

    env.close()
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, required=True, help="Scenario config file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--video", action="store_true", help="Save evaluation video")
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes,
        render=not args.no_render,
        save_video=args.video
    )