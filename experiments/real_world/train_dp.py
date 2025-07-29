# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script demonstrates how to train Diffusion Policy on the xarm-gello-teleop dataset.

Train a Diffusion Policy model using the teleoperation demonstrations collected
with xArm robot and Gello devices.
"""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/xarm_gello_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Number of offline training steps
    # Adjust based on dataset size - more episodes may need more steps
    training_steps = 10000
    log_freq = 50

    # Load our converted xarm-gello-teleop dataset
    # Update this path to match your converted dataset location
    dataset_repo_id = "tao_dataset"
    dataset_root = Path("/home/yolandazhu/xarm-gello-teleop/local_datasets").resolve()
    
    # When starting from scratch, we need to specify input/output shapes and dataset stats
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    print("Input features:", list(input_features.keys()))
    print("Output features:", list(output_features.keys()))

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features, device=device)

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Configure delta_timestamps for our xarm-gello-teleop dataset
    # Use the default configuration from DiffusionPolicy but adapt to our feature names
    delta_timestamps = {
        # Camera observations: use historical frames for temporal context
        "observation.images.cam_0": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.images.cam_1": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        
        # Combined state observation: use historical states for temporal context  
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        
        # Combined action: predict multiple future actions
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    #We can then instantiate the dataset with these delta_timestamps configuration. 
    print(f"Loading dataset from: {dataset_root / dataset_repo_id}")
    # Use larger tolerance for timestamp validation due to discretized timestamps
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, delta_timestamps=delta_timestamps, tolerance_s=0.04)

    # Print dataset information
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of episodes: {dataset.num_episodes}")
    print(f"Dataset FPS: {dataset_metadata.fps}")
    
    # Create optimizer and dataloader for offline training
    # Adjust batch size based on GPU memory and dataset size
    batch_size = min(32, len(dataset) // 10)  # Adaptive batch size
    print(f"Using batch size: {batch_size}")
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=2,  # Reduced for stability
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # Debug: print batch keys to understand the structure
            if step == 0:
                print("Batch keys:", list(batch.keys()))
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
