import argparse
import yaml
from evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=['urban', 'highway'])
    parser.add_argument("--models", type=str, default='all', choices=['all', 'yolo', 'detr', 'meta', 'hybrid'])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    # Load config
    config_path = f"configs/{args.scenario}.yaml"

    # Evaluate selected models
    models_to_evaluate = ['yolo', 'detr', 'meta', 'hybrid'] if args.models == 'all' else [args.models]
    
    results = {}
    for model in models_to_evaluate:
        print(f"Evaluating {model} model...")
        model_path = f"models/{args.scenario}_{model}"
        results[model] = evaluate_model(
            model_path=model_path,
            config_path=config_path,
            n_episodes=args.episodes,
            save_video=args.video,
            video_name=f"{args.scenario}_{model}"
        )

    # Print summary
    print("\nEvaluation Summary:")
    print("------------------")
    for model, result in results.items():
        print(f"\n{model.upper()} Model:")
        print(f"Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"Mean mAP: {result['mean_mAP']:.3f}")
        print(f"Average FPS: {result['mean_fps']:.1f}")

if __name__ == "__main__":
    main() 