#!/usr/bin/env python3

"""
Direct conversion from raw xarm-gello-teleop recordings to LeRobot dataset format.

This script uses the timestamp alignment method from teleop_postprocess_with_hand.py
but directly converts to LeRobot format without creating intermediate processed files.

Usage:
    python experiments/real_world/direct_to_lerobot.py --data_id 2 --output_repo_id my_dataset
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Add paths for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))


from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils import get_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root: Path = get_root(__file__)


def infer_features_from_raw_data(recording_dirs: Dict[str, List[str]], num_cams: int) -> Dict:
    """Infer dataset features from raw recording data across all episodes."""
    features = {}
    
    # Check all recordings to see what features are consistently available
    has_action_hand = True
    has_action_gello = True
    has_obs_hand = True
    
    for recording_name, action_name_list in recording_dirs.items():
        recording_dir = Path("/home/yolandazhu/xarm-gello-teleop/experiments/log/data") / recording_name
        
        # Check for xarm7 data (required)
        obs_gello_dir = recording_dir / "obs" / "xarm7"
        if obs_gello_dir.exists():
            joint_files = sorted([f for f in os.listdir(obs_gello_dir) if f.endswith('.txt')])
            if joint_files:
                joint_data = np.loadtxt(obs_gello_dir / joint_files[0])
                joint_dim = joint_data.shape[0] if joint_data.ndim == 1 else joint_data.shape[0]
                features["observation.xarm_joint_pos"] = {"dtype": "float32", "shape": (joint_dim,), "names": [f"xarm_joint_{i}" for i in range(joint_dim)]}
        
        # Check if action hand data exists in ALL recordings
        act_hand_dir = recording_dir / "action" / "hand"
        if not (act_hand_dir.exists() and list(act_hand_dir.glob("*.txt"))):
            has_action_hand = False
            
        # Check if action gello data exists in ALL recordings  
        act_gello_dir = recording_dir / "action" / "gello"
        if not (act_gello_dir.exists() and list(act_gello_dir.glob("*.txt"))):
            has_action_gello = False
            
        # Check if observation hand data exists in ALL recordings
        obs_hand_dir = recording_dir / "obs" / "hand"
        if not (obs_hand_dir.exists() and list(obs_hand_dir.glob("*.txt"))):
            has_obs_hand = False
    
    # Only add features that exist in ALL recordings
    if has_action_hand:
        # Get sample to determine dimensions
        for recording_name, action_name_list in recording_dirs.items():
            recording_dir = Path("/home/yolandazhu/xarm-gello-teleop/experiments/log/data") / recording_name
            act_hand_dir = recording_dir / "action" / "hand"
            hand_files = sorted([f for f in act_hand_dir.glob("*.txt") if f.name != "metadata.txt"])
            if hand_files:
                sample_hand_data = np.loadtxt(hand_files[0])
                hand_dim = sample_hand_data.shape[0] if sample_hand_data.ndim == 1 else sample_hand_data.shape[0]
                features['action.hand_joint_pos'] = {"dtype": "float32", "shape": (hand_dim,), "names": [f"act_hand_joint_{i}" for i in range(hand_dim)]}
                logger.info(f"Added action hand joint features with dimension: {hand_dim}")
                break
    else:
        logger.info("Action hand data not available in all recordings, skipping")
        
    if has_action_gello:
        # Get sample to determine dimensions
        for recording_name, action_name_list in recording_dirs.items():
            recording_dir = Path("/home/yolandazhu/xarm-gello-teleop/experiments/log/data") / recording_name
            act_gello_dir = recording_dir / "action" / "gello"
            gello_files = sorted([f for f in act_gello_dir.glob("*.txt") if f.name != "metadata.txt"])
            if gello_files:
                sample_gello_data = np.loadtxt(gello_files[0])
                gello_dim = sample_gello_data.shape[0] if sample_gello_data.ndim == 1 else sample_gello_data.shape[0]
                features['action.gello_joint_pos'] = {"dtype": "float32", "shape": (gello_dim,), "names": [f"gello_joint_{i}" for i in range(gello_dim)]}
                logger.info(f"Added action gello joint features with dimension: {gello_dim}")
                break
    else:
        logger.info("Action gello data not available in all recordings, skipping")
        
    if has_obs_hand:
        # Get sample to determine dimensions
        for recording_name, action_name_list in recording_dirs.items():
            recording_dir = Path("/home/yolandazhu/xarm-gello-teleop/experiments/log/data") / recording_name
            obs_hand_dir = recording_dir / "obs" / "hand"
            hand_files = sorted([f for f in obs_hand_dir.glob("*.txt") if f.name != "metadata.txt"])
            if hand_files:
                sample_hand_data = np.loadtxt(hand_files[0])
                hand_dim = sample_hand_data.shape[0] if sample_hand_data.ndim == 1 else sample_hand_data.shape[0]
                features['observation.hand_joint_pos'] = {"dtype": "float32", "shape": (hand_dim,), "names": [f"obs_hand_joint_{i}" for i in range(hand_dim)]}
                logger.info(f"Added observation hand joint features with dimension: {hand_dim}")
                break
    else:
        logger.info("Observation hand data not available in all recordings, skipping")
    
    # Add camera features - check first recording for image dimensions
    for recording_name, action_name_list in recording_dirs.items():
        if action_name_list:
            recording_dir = Path("/home/yolandazhu/xarm-gello-teleop/experiments/log/data") / recording_name
            obs_base_dir = recording_dir / "obs"
            if obs_base_dir.exists():
                obs_dirs = [d for d in obs_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if obs_dirs:
                    obs_dir = obs_dirs[0]  # Use first obs directory
                    for cam_idx in range(num_cams):
                        camera_dir = obs_dir / f"camera_{cam_idx}"
                        
                        # Check RGB image dimensions
                        rgb_dir = camera_dir / "rgb"
                        if rgb_dir.exists():
                            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])
                            if rgb_files:
                                rgb_image = Image.open(rgb_dir / rgb_files[0])
                                height, width = rgb_image.size[1], rgb_image.size[0]  # PIL uses (width, height)
                                features[f'observation.images.cam_{cam_idx}'] = {
                                    "dtype": "image", 
                                    "shape": (height, width, 3), 
                                    "names": ["height", "width", "channels"]
                                }
                    
                    # Skip depth images for now due to channel issues with LeRobot image writer
                    # TODO: Add depth support when LeRobot supports single-channel images
                    logger.info(f"Skipping depth images (LeRobot requires 3-channel images)")
                    break  # Only need to check first recording for image dimensions
    
    logger.info(f"Inferred features: {features}")
    return features

def convert_recording_to_lerobot(
    recording_name: str,
    action_name: str,
    recording_dir: Path,
    dataset: LeRobotDataset,
    num_cams: int = 2,
    task_name: str = "xarm_orca",
    tolerance_s: float = 0.03
) -> bool:
    """Convert a single recording directly to LeRobot format using timestamp alignment."""
    
    logger.info(f"Converting recording: {recording_name}/{action_name}")
    
    # Load xarm7 joint data (already saved as files)
    obs_gello_dir = recording_dir / "obs" / "xarm7"
    obs_xarm7_timestamps = []
    joint_data_lookup = {}
    
    if obs_gello_dir.exists():
        joint_files = sorted([f for f in os.listdir(obs_gello_dir) if f.endswith('.txt')])
        obs_xarm7_timestamps = [float(f[:-4]) for f in joint_files]
        # Create lookup table for joint data
        for joint_file in joint_files:
            timestamp = float(joint_file[:-4])
            joint_data_lookup[timestamp] = obs_gello_dir / joint_file
        logger.info(f"Loaded {len(joint_files)} joint files")
    else:
        logger.warning(f"No joint directory found in {recording_dir}")
        return False
    
    # Load camera data from obs directory
    obs_dir = recording_dir / "obs" / str(action_name)
    if not obs_dir.exists():
        logger.warning(f"Observation directory {obs_dir} not found")
        return False
        
    with open(obs_dir / 'timestamps.txt', 'r') as f:
        cams_timestamps = f.readlines()
    cams_timestamps = [[float(tt) for tt in t.split()[-num_cams:]] for t in cams_timestamps]
    
    # Load action hand data from existing txt files (extracted from ROS2 bags)
    act_hand_dir = recording_dir / "action" / "hand"
    act_hand_data_list = []
    act_hand_timestamps = []
    
    if act_hand_dir.exists() and list(act_hand_dir.glob("*.txt")):
        hand_files = sorted([f for f in act_hand_dir.glob("*.txt")])
        # Filter out metadata.txt if it exists
        hand_files = [f for f in hand_files if f.name != "metadata.txt"]
        
        for hand_file in hand_files:
            try:
                timestamp = float(hand_file.stem)
                hand_data = np.loadtxt(hand_file)
                act_hand_data_list.append((timestamp, hand_data))
                act_hand_timestamps.append(timestamp)
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to load action hand file {hand_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(act_hand_data_list)} action hand data files")
    else:
        logger.info("No action hand txt files found, skipping action hand data")
    
    # Load action gello data from existing txt files (extracted from ROS2 bags)
    act_gello_dir = recording_dir / "action" / "gello"
    act_gello_data_list = []
    act_gello_timestamps = []
    
    if act_gello_dir.exists() and list(act_gello_dir.glob("*.txt")):
        gello_files = sorted([f for f in act_gello_dir.glob("*.txt")])
        # Filter out metadata.txt if it exists
        gello_files = [f for f in gello_files if f.name != "metadata.txt"]
        
        for gello_file in gello_files:
            try:
                timestamp = float(gello_file.stem)
                gello_data = np.loadtxt(gello_file)
                act_gello_data_list.append((timestamp, gello_data))
                act_gello_timestamps.append(timestamp)
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to load action gello file {gello_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(act_gello_data_list)} action gello data files")
    else:
        logger.info("No action gello txt files found, skipping action gello data")
    
    # Load observation hand data from existing txt files (extracted from ROS2 bags)
    obs_hand_dir = recording_dir / "obs" / "hand"
    obs_hand_data_list = []
    obs_hand_timestamps = []
    
    if obs_hand_dir.exists() and list(obs_hand_dir.glob("*.txt")):
        hand_files = sorted([f for f in obs_hand_dir.glob("*.txt")])
        # Filter out metadata.txt if it exists
        hand_files = [f for f in hand_files if f.name != "metadata.txt"]
        
        for hand_file in hand_files:
            try:
                timestamp = float(hand_file.stem)
                hand_data = np.loadtxt(hand_file)
                obs_hand_data_list.append((timestamp, hand_data))
                obs_hand_timestamps.append(timestamp)
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to load observation hand file {hand_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(obs_hand_data_list)} observation hand data files")
    else:
        logger.info("No observation hand txt files found, skipping observation hand data")
    
    # Hand timestamps are already extracted above
    
    # Find the stable overlap period where all data is available
    # Get the latest start time and earliest end time across all data streams
    all_start_times = [min(cams_timestamps)[0] if cams_timestamps else float('inf')]
    all_end_times = [max(cams_timestamps)[0] if cams_timestamps else 0]
    
    if obs_xarm7_timestamps:
        all_start_times.append(min(obs_xarm7_timestamps))
        all_end_times.append(max(obs_xarm7_timestamps))
    if act_hand_timestamps:
        all_start_times.append(min(act_hand_timestamps))
        all_end_times.append(max(act_hand_timestamps))
    if act_gello_timestamps:
        all_start_times.append(min(act_gello_timestamps))
        all_end_times.append(max(act_gello_timestamps))
    if obs_hand_timestamps:
        all_start_times.append(min(obs_hand_timestamps))
        all_end_times.append(max(obs_hand_timestamps))
    
    # The stable period is from the latest start to the earliest end
    stable_start = max(all_start_times)
    stable_end = min(all_end_times)
    
    logger.info(f"Data overlap period: {stable_start:.3f} to {stable_end:.3f} ({stable_end - stable_start:.3f}s)")
    
    # Filter camera frames to only include the stable period
    stable_cam_frames = []
    for t, cam_timestamp in enumerate(cams_timestamps):
        if stable_start <= cam_timestamp[0] <= stable_end:
            stable_cam_frames.append((t, cam_timestamp))
    
    logger.info(f"Processing {len(stable_cam_frames)} stable camera frames (out of {len(cams_timestamps)} total)...")
    successful_matches = 0
    
    for t, cam_timestamp in stable_cam_frames:
        master_timestamp = cam_timestamp[0]
        
        if t % 50 == 0:  # Progress indicator every 50 frames
            logger.info(f"Processing frame {t}/{len(cams_timestamps)}, timestamp: {master_timestamp}")
        
        # Find corresponding joint data
        obs_xarm7_idx = -1
        if obs_xarm7_timestamps:
            min_dist = tolerance_s
            for jt, xarm7_timestamp in enumerate(obs_xarm7_timestamps):
                t_diff = abs(master_timestamp - xarm7_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    obs_xarm7_idx = jt
        
        # Find corresponding action hand data
        act_hand_idx = -1
        if act_hand_timestamps:
            min_dist = tolerance_s
            for ht, hand_timestamp in enumerate(act_hand_timestamps):
                t_diff = abs(master_timestamp - hand_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    act_hand_idx = ht
        
        # Find corresponding action gello data
        act_gello_idx = -1
        if act_gello_timestamps:
            min_dist = tolerance_s
            for gt, gello_timestamp in enumerate(act_gello_timestamps):
                t_diff = abs(master_timestamp - gello_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    act_gello_idx = gt
        
        # Find corresponding observation hand data
        obs_hand_idx = -1
        if obs_hand_timestamps:
            min_dist = tolerance_s
            for ht, hand_timestamp in enumerate(obs_hand_timestamps):
                t_diff = abs(master_timestamp - hand_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    obs_hand_idx = ht
        # Skip frame if no joint data
        if obs_xarm7_idx < 0:
            logger.warning(f"No obs xarm7 data found for frame {t}")
            continue
        # Skip frame if missing required hand/gello data (strict alignment required)
        if act_hand_idx < 0:
            logger.warning(f"No act hand data found for frame {t}")
            continue
        if act_gello_idx < 0:
            logger.warning(f"No act gello data found for frame {t}")
            continue  
        if obs_hand_idx < 0:
            logger.warning(f"No joint data found for frame {t}")
            continue
        
        # Prepare frame data
        frame_data = {}
        
        # Load xarm joint data (required) - ensure correct dtype
        xarm7_timestamp = obs_xarm7_timestamps[obs_xarm7_idx]
        source_joint_file = joint_data_lookup[xarm7_timestamp]
        joint_data = np.loadtxt(source_joint_file).astype(np.float32)
        frame_data['observation.xarm_joint_pos'] = joint_data
        
        # Load action hand joint data (required) - ensure correct dtype
        _, act_hand_data = act_hand_data_list[act_hand_idx]
        frame_data['action.hand_joint_pos'] = act_hand_data.astype(np.float32)
    
        # Load action gello joint data (required) - ensure correct dtype
        _, act_gello_data = act_gello_data_list[act_gello_idx]
        frame_data['action.gello_joint_pos'] = act_gello_data.astype(np.float32)
        
        # Load observation hand joint data (required) - ensure correct dtype
        _, obs_hand_data = obs_hand_data_list[obs_hand_idx]
        frame_data['observation.hand_joint_pos'] = obs_hand_data.astype(np.float32)
        
        # Load camera images from obs directory
        for cam_idx in range(num_cams):
            camera_dir = obs_dir / f"camera_{cam_idx}"
            
            # RGB image
            rgb_file = camera_dir / "rgb" / f"{t:06d}.jpg"
            if rgb_file.exists():
                rgb_image = Image.open(rgb_file)
                frame_data[f'observation.images.cam_{cam_idx}'] = np.array(rgb_image)
            else:
                logger.warning(f"Missing RGB image for camera {cam_idx}, frame {t}")
                
            # Skip depth images for now (LeRobot image writer requires 3-channel images)
        
        # Add frame to dataset
        try:
            dataset.add_frame(frame_data, task=task_name, timestamp=master_timestamp)
            successful_matches += 1
        except Exception as e:
            logger.error(f"Error adding frame {t}: {e}")
            continue
    
    # Check if we have sufficient valid frames
    if successful_matches == 0:
        logger.warning(f"No valid frames found for {recording_name}/{action_name} - skipping episode (timing misalignment)")
        return False
    
    logger.info(f"Successfully processed {successful_matches}/{len(cams_timestamps)} frames")
    return True

def direct_convert_to_lerobot(
    recording_dirs: Dict[str, List[str]],
    output_repo_id: str,
    output_root: Optional[Path] = None,
    num_cams: int = 2,
    fps: int = 30,
    robot_type: str = "xarm",
    task_name: str = "manipulation",
    tolerance_s:float = 0.03
) -> bool:
    """Direct conversion from raw recordings to LeRobot format."""
    
    logger.info(f"Starting direct conversion to LeRobot dataset: {output_repo_id}")
    
    # Infer features from all recordings to ensure consistency
    features = infer_features_from_raw_data(recording_dirs, num_cams)
    
    # Create LeRobot dataset

    # Create dataset metadata first
    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        root=output_root,
        use_videos=True,  # Use videos for efficient storage
        tolerance_s=tolerance_s  # Relaxed tolerance for teleoperation data (30ms)
    )
    

    
    # Convert each recording as separate episodes
    successful_episodes = 0
    total_episodes = sum(len(action_list) for action_list in recording_dirs.values())
    
    for recording_name, action_name_list in recording_dirs.items():
        recording_dir = root / "log" / "data" / recording_name
        
        if not recording_dir.exists():
            logger.warning(f"Recording directory not found: {recording_dir}")
            continue
        
        for action_name in action_name_list:
            try:
                if convert_recording_to_lerobot(
                    recording_name, action_name, recording_dir, dataset,
                    num_cams, task_name, tolerance_s
                ):
                    # Save the episode if conversion was successful
                    dataset.save_episode()
                    successful_episodes += 1
                    logger.info(f"Saved episode for {recording_name}/{action_name}")
                else:
                    logger.warning(f"Failed to convert {recording_name}/{action_name}")
            except Exception as e:
                logger.error(f"Error converting {recording_name}/{action_name}: {e}")
    
    logger.info(f"Successfully converted {successful_episodes}/{total_episodes} episodes")
    
    if successful_episodes > 0:
        logger.info("Dataset conversion completed successfully!")
        logger.info(f"Output location: {dataset.root}")
        return True
    else:
        logger.error("No episodes were successfully converted!")
        return False

def discover_episodes(data_root: Path) -> Dict[str, List[str]]:
    """Automatically discover all episodes under tao directory."""
    tao_dir = data_root / "tao"
    episodes = {}
    
    if not tao_dir.exists():
        logger.warning(f"Directory not found: {tao_dir}")
        return episodes
    
    # Scan all recording directories (r0, r1, r2, etc.)
    for recording_dir in sorted(tao_dir.iterdir()):
        if recording_dir.is_dir() and recording_dir.name.startswith('r'):
            obs_dir = recording_dir / "obs"
            if obs_dir.exists():
                # Find all timestamp directories
                timestamp_dirs = [d.name for d in obs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                if timestamp_dirs:
                    recording_name = f"tao/{recording_dir.name}"
                    episodes[recording_name] = sorted(timestamp_dirs)
                    logger.info(f"Found {len(timestamp_dirs)} episodes in {recording_name}: {timestamp_dirs}")
    
    return episodes

def main():
    parser = argparse.ArgumentParser(description="Direct conversion from raw recordings to LeRobot format")
    parser.add_argument('--data_id', type=int, default=0,
                       help='Data configuration ID (0, 1, 2, or 99)')
    parser.add_argument('--output_repo_id', type=str, default="xarm_manipulation_direct",
                       help='Repository ID for the output LeRobot dataset')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Root directory for output dataset')
    parser.add_argument('--num_cams', type=int, default=2,
                       help='Number of cameras (default: 2)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second of the dataset (default: 30)')
    parser.add_argument('--robot_type', type=str, default="xarm",
                       help='Robot type identifier (default: xarm)')
    parser.add_argument('--task_name', type=str, default="manipulation",
                       help='Task name for the dataset (default: manipulation)')
    parser.add_argument('--bimanual', default="False", action='store_true',
                       help='Enable bimanual mode')
    
    args = parser.parse_args()
    
    # Configuration mapping (same as teleop_postprocess_with_hand.py)
    ii = args.data_id
    

    if ii == 0:
        bimanual = False
        num_cams = 2
        # Automatically discover all episodes under tao directory
        data_root = root / "log" / "data"
        dirs = discover_episodes(data_root)
        if not dirs:
            logger.error("No episodes found under tao directory")
            exit(1)
    else:
        logger.error(f"No configuration found for data_id {ii}")
        logger.error("Available data_ids: 0")
        exit(1)
    
    # Override with command line arguments
    if args.num_cams:
        num_cams = args.num_cams
    if args.bimanual:
        bimanual = args.bimanual
    
    # Use provided repo_id or default name
    repo_id = args.output_repo_id
    
    output_root = Path(args.output_root).resolve() if args.output_root else Path("./local_datasets").resolve()
    
    logger.info(f"Configuration: data_id={ii}, num_cams={num_cams}, bimanual={bimanual}")
    logger.info(f"Output repo_id: {repo_id}")
    logger.info(f"Recording directories: {dirs}")
    
    success = direct_convert_to_lerobot(
        recording_dirs=dirs,
        output_repo_id=repo_id,
        output_root=output_root,
        num_cams=num_cams,
        fps=args.fps,
        robot_type=args.robot_type,
        task_name=args.task_name,
        tolerance_s=0.05
    )
    
    if success:
        logger.info("Dataset conversion completed successfully!")
    else:
        logger.error("Dataset conversion failed!")
        exit(1)
        


if __name__ == "__main__":
    main()