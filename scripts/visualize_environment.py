import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2

def plot_episode_frames(frames, episode_num, output_dir):
    """Plot key frames from an episode"""
    n_frames = len(frames)
    fig, axes = plt.subplots(1, n_frames, figsize=(5*n_frames, 5))
    
    for i, frame in enumerate(frames):
        if n_frames > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'episode_{episode_num}_frames.png')
    plt.close()

def plot_agent_trajectory(trajectory, episode_num, output_dir):
    """Plot agent's trajectory during an episode"""
    plt.figure(figsize=(10, 10))
    
    # Plot trajectory
    x, y = zip(*trajectory)
    plt.plot(x, y, 'b-', label='Trajectory')
    plt.plot(x[0], y[0], 'go', label='Start')
    plt.plot(x[-1], y[-1], 'ro', label='End')
    
    plt.title(f'Agent Trajectory - Episode {episode_num}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / f'episode_{episode_num}_trajectory.png')
    plt.close()

def plot_episode_metrics(metrics, episode_num, output_dir):
    """Plot episode-specific metrics"""
    plt.figure(figsize=(15, 5))
    
    # Plot reward
    plt.subplot(131)
    plt.plot(metrics['rewards'], label='Reward')
    plt.title('Episode Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot speed
    plt.subplot(132)
    plt.plot(metrics['speed'], label='Speed')
    plt.title('Vehicle Speed')
    plt.xlabel('Step')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    
    # Plot steering
    plt.subplot(133)
    plt.plot(metrics['steering'], label='Steering')
    plt.title('Steering Angle')
    plt.xlabel('Step')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'episode_{episode_num}_metrics.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Load episode data
    episode_data_path = f'logs/episode_{args.episode}_data.npy'
    if Path(episode_data_path).exists():
        episode_data = np.load(episode_data_path, allow_pickle=True).item()
        
        # Plot frames
        if 'frames' in episode_data:
            plot_episode_frames(episode_data['frames'], args.episode, output_dir)
        
        # Plot trajectory
        if 'trajectory' in episode_data:
            plot_agent_trajectory(episode_data['trajectory'], args.episode, output_dir)
        
        # Plot metrics
        if 'metrics' in episode_data:
            plot_episode_metrics(episode_data['metrics'], args.episode, output_dir)
    else:
        print(f"Warning: No data found for episode {args.episode}")

if __name__ == "__main__":
    main() 