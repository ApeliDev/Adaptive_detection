import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_scenario_performance(metrics, scenario, output_dir):
    """Plot performance metrics for a scenario"""
    plt.figure(figsize=(15, 5))
    
    # Plot rewards
    plt.subplot(131)
    plt.plot(metrics['rewards'], label='Reward')
    plt.title(f'{scenario} - Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot detection accuracy
    plt.subplot(132)
    plt.plot(metrics['detection_accuracy'], label='Accuracy')
    plt.title(f'{scenario} - Detection Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot FPS
    plt.subplot(133)
    plt.plot(metrics['fps'], label='FPS')
    plt.title(f'{scenario} - FPS')
    plt.xlabel('Episode')
    plt.ylabel('FPS')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{scenario}_performance.png')
    plt.close()

def plot_scenario_comparison(metrics, output_dir):
    """Create comparison plots between scenarios"""
    scenarios = list(metrics.keys())
    
    # Bar chart comparison
    plt.figure(figsize=(10, 6))
    metrics_to_compare = ['mean_reward', 'mean_accuracy', 'mean_fps']
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    for i, scenario in enumerate(scenarios):
        values = [metrics[scenario][m] for m in metrics_to_compare]
        plt.bar(x + i*width, values, width, label=scenario)

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Scenario Comparison')
    plt.xticks(x + width/2, metrics_to_compare)
    plt.legend()
    plt.savefig(output_dir / 'scenario_comparison.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=str, nargs='+', default=['urban', 'highway'])
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Load metrics for each scenario
    metrics = {}
    for scenario in args.scenarios:
        metrics_path = f'logs/{scenario}_metrics.npy'
        if Path(metrics_path).exists():
            metrics[scenario] = np.load(metrics_path, allow_pickle=True).item()
            plot_scenario_performance(metrics[scenario], scenario, output_dir)
        else:
            print(f"Warning: No metrics found for {scenario}")

    if len(metrics) > 1:
        plot_scenario_comparison(metrics, output_dir)
    else:
        print("Need at least two scenarios for comparison")

if __name__ == "__main__":
    main() 