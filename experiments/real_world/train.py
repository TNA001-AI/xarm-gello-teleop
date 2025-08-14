#!/usr/bin/env python3

"""
Training script using LeRobot's built-in wandb integration with custom tolerance.
"""
from pathlib import Path
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs.train import TrainPipelineConfig  
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from lerobot.datasets.utils import dataset_to_policy_features, cycle
from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger
## --------------------------------------- ##
TOLERANCE_S = 0.05
STEPS = 100000
BATCH_SIZE = 64
NUM_WORKERS = 8
DATASET_ROOT = "/data/lerobot_datasets"
## --------------------------------------- ##
class SafeDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for LeRobotDataset that excludes the last frame of each episode
    to avoid frame index out of bounds errors.
    """
    def __init__(self, dataset: LeRobotDataset, frames_to_exclude: int = 1):
        self.dataset = dataset
        self.frames_to_exclude = frames_to_exclude
        
        # Build a list of valid indices (excluding last N frames of each episode)
        self.valid_indices = []
        
        for ep_idx in range(dataset.num_episodes):
            ep_start = dataset.episode_data_index["from"][ep_idx].item()
            ep_end = dataset.episode_data_index["to"][ep_idx].item()
            
            # Exclude the last N frames of this episode
            safe_end = ep_end - frames_to_exclude
            
            # Add valid indices for this episode
            for idx in range(ep_start, safe_end):
                self.valid_indices.append(idx)
        
        print(f"SafeDatasetWrapper: Original dataset has {dataset.num_frames} frames")
        print(f"SafeDatasetWrapper: After excluding {frames_to_exclude} frame(s) per episode, {len(self.valid_indices)} frames remain")
        
        # Copy attributes from original dataset
        self.meta = dataset.meta
        self.features = dataset.features
        self.num_episodes = dataset.num_episodes
        self.num_frames = len(self.valid_indices)
        self.hf_dataset = dataset.hf_dataset
        self.episode_data_index = dataset.episode_data_index
        # Only copy stats if it exists
        self.stats = getattr(dataset, 'stats', None)
        self.fps = dataset.fps
        self.video_backend = getattr(dataset, 'video_backend', None)
        self.delta_timestamps = getattr(dataset, 'delta_timestamps', None)
        # Copy tolerance_s if it exists
        self.tolerance_s = getattr(dataset, 'tolerance_s', 0.05)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map from wrapper index to original dataset index
        original_idx = self.valid_indices[idx]
        return self.dataset[original_idx]

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def custom_train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend="torchcodec",
        tolerance_s= TOLERANCE_S,  
    )
    
    logging.info(f"Dataset created successfully with tolerance_s={dataset.tolerance_s}")
    
    # Wrap dataset to exclude last frames of each episode (prevents frame index errors)
    # This is necessary because video frames are 0-indexed and the last frame might
    # cause index out of bounds when delta_timestamps look ahead
    dataset = SafeDatasetWrapper(dataset, frames_to_exclude=2)  # Exclude last 2 frames for safety
    logging.info(f"Applied SafeDatasetWrapper: {dataset.num_frames} frames available for training")


    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers        = cfg.num_workers,
        batch_size         = cfg.batch_size,
        shuffle            = True,
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 6,         
        drop_last          = True,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            # Removed wandb checkpoint upload

    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)

def main():
    # Dataset configuration with resize transform
    from lerobot.datasets.transforms import ImageTransformsConfig, ImageTransformConfig
    
    image_transforms_config = ImageTransformsConfig(
        enable=True,
        max_num_transforms=1,
        random_order=False,
    )
    
    dataset_config = DatasetConfig(
        repo_id="tao_dataset",
        root=DATASET_ROOT,
        image_transforms=image_transforms_config
    )
    
    # Wandb configuration
    wandb_config = WandBConfig(
        enable=True,
        project="xarm-gello-teleop",  # Your project name
        entity=None,  # Your wandb username/team (optional)
        notes="Diffusion Policy training on xArm-Gello teleop data"
    )
    
    # Get dataset metadata to configure policy
    dataset_metadata = LeRobotDatasetMetadata("tao_dataset", root=Path(DATASET_ROOT))
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Policy configuration
    policy_config = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        device="cuda",
        repo_id="xarm-gello-diffusion-policy",  # Required for hub upload
        crop_shape= (224,224),
        vision_backbone="resnet34"
    )
    
    # Main training configuration
    config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        wandb=wandb_config,
        output_dir=Path("/data/xarm_orca_diffusion"),
        job_name="xarm-gello-diffusion",
        seed=42,
        steps=STEPS,  # Total training steps
        log_freq=50,  # Log every 50 steps
        save_freq=5000,  # Save checkpoint every 1000 steps
        save_checkpoint=True,
        batch_size=BATCH_SIZE,  # Increased for better GPU utilization
        num_workers = NUM_WORKERS,  # Set to 0 to disable multiprocessing
    )
    
    # Start training with wandb logging and custom tolerance
    custom_train(config)

if __name__ == "__main__":
    init_logging()
    main()