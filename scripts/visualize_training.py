import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_metrics(metrics, model_name, output_dir):
    """Plot training metrics for a model"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(231)
    plt.plot(metrics['rewards'], label='Reward')
    plt.title(f'{model_name} - Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Plot FPS
    plt.subplot(232)
    plt.plot(metrics['fps'], label='FPS')
    plt.title(f'{model_name} - FPS')
    plt.xlabel('Episode')
    plt.ylabel('FPS')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(233)
    plt.plot(metrics['accuracy'], label='Accuracy')
    plt.title(f'{model_name} - Detection Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot SSD vs YOLO usage
    plt.subplot(234)
    ssd_usage = metrics.get('ssd_usage', [])
    yolo_usage = metrics.get('yolo_usage', [])
    if ssd_usage and yolo_usage:
        plt.plot(ssd_usage, label='SSD Usage')
        plt.plot(yolo_usage, label='YOLO Usage')
        plt.title(f'{model_name} - Detector Usage')
        plt.xlabel('Episode')
        plt.ylabel('Usage Count')
        plt.legend()
    
    # Plot inference times
    plt.subplot(235)
    ssd_time = metrics.get('ssd_time', [])
    yolo_time = metrics.get('yolo_time', [])
    if ssd_time and yolo_time:
        plt.plot(ssd_time, label='SSD Time')
        plt.plot(yolo_time, label='YOLO Time')
        plt.title(f'{model_name} - Inference Times')
        plt.xlabel('Episode')
        plt.ylabel('Time (ms)')
        plt.legend()
    
    # Plot detection counts
    plt.subplot(236)
    ssd_dets = metrics.get('ssd_detections', [])
    yolo_dets = metrics.get('yolo_detections', [])
    if ssd_dets and yolo_dets:
        plt.plot(ssd_dets, label='SSD Detections')
        plt.plot(yolo_dets, label='YOLO Detections')
        plt.title(f'{model_name} - Detection Counts')
        plt.xlabel('Episode')
        plt.ylabel('Number of Detections')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_{model_name}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='performance')
    parser.add_argument("--metrics", type=str, nargs='+', default=['reward', 'fps', 'detection_accuracy'])
    args = parser.parse_args()

    # Create output directory
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    # Load metrics for multi-detector model
    metrics_path = 'logs/multi_detector_metrics.npy'
    if Path(metrics_path).exists():
        metrics = np.load(metrics_path, allow_pickle=True).item()
        plot_training_metrics(metrics, 'multi_detector', output_dir)
    else:
        print("Warning: No metrics found for multi-detector model")

if __name__ == "__main__":
    main() 