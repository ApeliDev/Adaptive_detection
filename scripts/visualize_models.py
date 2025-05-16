import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_curves(metrics, model_name, output_dir):
    """Plot training curves for a model"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(131)
    plt.plot(metrics['loss'], label='Training Loss')
    if 'val_loss' in metrics:
        plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(132)
    plt.plot(metrics['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in metrics:
        plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot learning rate
    plt.subplot(133)
    plt.plot(metrics['learning_rate'], label='Learning Rate')
    plt.title(f'{model_name} - Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_training.png')
    plt.close()

def plot_model_comparison(metrics, output_dir):
    """Create comparison plots between models"""
    models = list(metrics.keys())
    
    # Bar chart comparison
    plt.figure(figsize=(10, 6))
    metrics_to_compare = ['final_loss', 'final_accuracy', 'training_time']
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    for i, model in enumerate(models):
        values = [metrics[model][m] for m in metrics_to_compare]
        plt.bar(x + i*width, values, width, label=model)

    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Model Comparison')
    plt.xticks(x + width/2, metrics_to_compare)
    plt.legend()
    plt.savefig(output_dir / 'model_comparison.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs='+', default=['dqn', 'ppo', 'sac'])
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
            plot_training_curves(metrics[model], model, output_dir)
        else:
            print(f"Warning: No metrics found for {model}")

    if len(metrics) > 1:
        plot_model_comparison(metrics, output_dir)
    else:
        print("Need at least two models for comparison")

if __name__ == "__main__":
    main() 