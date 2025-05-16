import argparse
import yaml
from training.meta_trainer import MetaPPOTrainer
from training.curriculum import CurriculumManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=True, choices=['urban', 'highway'])
    args = parser.parse_args()

    # Load configs
    with open(f"configs/{args.scenario}.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Train SSD model
    print("Training SSD model...")
    trainer = MetaPPOTrainer(config['environment'], config['meta_learning'])
    trainer.train(config['training']['total_timesteps'])

if __name__ == "__main__":
    main() 