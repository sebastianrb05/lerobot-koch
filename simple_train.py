#!/usr/bin/env python

import sys
import os
sys.path.insert(0, 'src')

import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting simplified training script")
    
    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")
    
    # Dataset path
    dataset_root = Path("recorded_data/throw_paper")
    if not dataset_root.exists():
        logger.error(f"Dataset not found at {dataset_root}")
        return
    
    logger.info(f"Dataset found at {dataset_root}")
    
    # Try to import and load dataset
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logger.info("Successfully imported LeRobotDataset")
        
        # Load dataset
        dataset = LeRobotDataset(
            repo_id="koch_test",
            root=dataset_root,
        )
        logger.info(f"Dataset loaded successfully. Total episodes: {dataset.meta.total_episodes}")
        logger.info(f"Dataset features: {list(dataset.features.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Try to import policy
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        logger.info("Successfully imported ACT policy")
        
        # Create policy config
        # We'll need to determine the input/output features from the dataset
        input_features = {}
        output_features = {}
        
        for key, feature in dataset.features.items():
            if feature.get("type") == "action":
                output_features[key] = feature
            else:
                input_features[key] = feature
        
        config = ACTConfig(
            input_features=input_features,
            output_features=output_features,
        )
        
        # Create policy
        policy = ACTPolicy(config, dataset_stats=dataset.meta.stats)
        policy.to(device)
        logger.info("ACT policy created and moved to device")
        
    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        return
    
    logger.info("Training setup completed successfully!")
    logger.info("You can now run the full training script with:")
    logger.info("set PYTHONPATH=src && python -m lerobot.scripts.train --dataset.repo_id=koch_test --dataset.root=recorded_data/throw_paper --policy.type=act --output_dir=outputs/train/act_koch_test --job_name=act_koch_test --policy.device=cuda --wandb.enable=true")

if __name__ == "__main__":
    main() 