import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_model_comparison(metrics, output_dir):
    """Create comparison plots for all models"""
    # Bar chart
    plt.figure(figsize=(10, 6))
    models = list(metrics.keys())
    rewards = [m['mean_reward'] for m in metrics.values()]
    plt.bar(models, rewards)
    plt.title('Model Comparison - Mean Rewards')
    plt.xlabel('Model')
    plt.ylabel('Mean Reward')
    plt.savefig(output_dir / 'model_comparison_bar.png')
    plt.close()

    # Radar plot
    plt.figure(figsize=(8, 8))
    metrics_to_plot = ['mean_reward', 'mean_mAP', 'mean_fps']
    angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # complete the circle

    ax = plt.subplot(111, polar=True)
    for model, values in metrics.items():
        values = [values[m] for m in metrics_to_plot]
        values = np.concatenate((values, [values[0]]))  # complete the circle
        ax.plot(angles, values, label=model)
        ax.fill(angles, values, alpha=0.25)

    plt.xticks(angles[:-1], metrics_to_plot)
    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
    plt.title('Model Comparison - Radar Plot')
    plt.savefig(output_dir / 'model_comparison_radar.png')
    plt.close()

    # Box plot
    plt.figure(figsize=(10, 6))
    data = [m['rewards'] for m in metrics.values()]
    plt.boxplot(data, labels=models)
    plt.title('Model Comparison - Reward Distribution')
    plt.xlabel('Model')
    plt.ylabel('Reward')
    plt.savefig(output_dir / 'model_comparison_boxplot.png')
    plt.close()

    # Detector usage comparison
    plt.figure(figsize=(10, 6))
    detector_usage = {
        'SSD': [m.get('mean_ssd_usage', 0) for m in metrics.values()],
        'YOLO': [m.get('mean_yolo_usage', 0) for m in metrics.values()]
    }
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, detector_usage['SSD'], width, label='SSD')
    plt.bar(x + width/2, detector_usage['YOLO'], width, label='YOLO')
    plt.title('Model Comparison - Detector Usage')
    plt.xlabel('Model')
    plt.ylabel('Usage Count')
    plt.xticks(x, models)
    plt.legend()
    plt.savefig(output_dir / 'model_comparison_detector_usage.png')
    plt.close()

    # Inference time comparison
    plt.figure(figsize=(10, 6))
    inference_times = {
        'SSD': [m.get('mean_ssd_time', 0) for m in metrics.values()],
        'YOLO': [m.get('mean_yolo_time', 0) for m in metrics.values()]
    }
    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, inference_times['SSD'], width, label='SSD')
    plt.bar(x + width/2, inference_times['YOLO'], width, label='YOLO')
    plt.title('Model Comparison - Inference Times')
    plt.xlabel('Model')
    plt.ylabel('Time (ms)')
    plt.xticks(x, models)
    plt.legend()
    plt.savefig(output_dir / 'model_comparison_inference_times.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='comparison')
    parser.add_argument("--models", type=str, nargs='+', default=['ssd', 'yolo', 'multi_detector'])
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Load metrics for each model
    metrics = {}
    for model in args.models:
        metrics_path = f'logs/{model}_metrics.npy'
        if Path(metrics_path).exists():
            metrics[model] = np.load(metrics_path, allow_pickle=True).item()
        else:
            print(f"Warning: No metrics found for {model}")

    if metrics:
        plot_model_comparison(metrics, output_dir)
    else:
        print("No metrics found for any model")

if __name__ == "__main__":
    main() 