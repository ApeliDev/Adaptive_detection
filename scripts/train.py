import yaml
from training.meta_trainer import MetaPPOTrainer
from core.marl_env import CarlaMARLEnv

def main():
    # Load configs
    with open("configs/defaults.yaml", 'r') as f:
        config = yaml.safe_load(f)

    with open('configs/urban.yaml') as f:
        urban_config = yaml.safe_load(f)

    with open('configs/highway.yaml') as f:
        highway_config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = MetaPPOTrainer(
        env_config=config['environment'],
        meta_config=config['meta_learning']
    )
    
    # Train
    trainer.train(config['training']['total_timesteps'])
    
    # Save final model
    trainer.policy.save("models/final_meta_ppo")

if __name__ == "__main__":
    main()