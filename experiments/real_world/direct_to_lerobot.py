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

# SQLite imports for reading ROS2 bags directly
import sqlite3
import struct

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from utils import get_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root: Path = get_root(__file__)

def read_ros2_bag_sqlite(bag_path: str, topic_name: str):
    """
    Read messages from ROS2 bag SQLite database directly
    Returns list of (timestamp, parsed_data) tuples
    """
    db_files = list(Path(bag_path).glob("*.db3"))
    if not db_files:
        logger.warning(f"No .db3 files found in {bag_path}")
        return []
    
    db_file = db_files[0]
    logger.info(f"Reading from database: {db_file}")
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Get topics
        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()
        logger.info(f"Available topics:")
        for topic_id, name, msg_type in topics:
            logger.info(f"  {topic_id}: {name} [{msg_type}]")
        
        # Find target topic
        target_topic_id = None
        for topic_id, name, msg_type in topics:
            if name == topic_name:
                target_topic_id = topic_id
                logger.info(f"Found target topic: {name} (ID: {topic_id})")
                break
        
        if target_topic_id is None:
            logger.warning(f"Topic {topic_name} not found")
            conn.close()
            return []
        
        # Get messages
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (target_topic_id,))
        messages = cursor.fetchall()
        logger.info(f"Found {len(messages)} messages")
        
        # Parse messages and convert to list of (timestamp_seconds, parsed_data)
        result = []
        for timestamp_ns, data in messages:
            timestamp_s = timestamp_ns / 1e9
            parsed_data = parse_joint_state_message(data)
            if parsed_data is not None:
                result.append((timestamp_s, parsed_data))
        
        conn.close()
        logger.info(f"Successfully parsed {len(result)} messages")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to read ROS2 bag {bag_path}: {e}")
        return []


def parse_joint_state_message(data):
    """
    Parser for JointState messages from ROS2 bags
    JointState contains: header, name[], position[], velocity[], effort[]
    """
    try:
        offset = 4  # Skip CDR header
        
        # Skip ROS header - stamp (8 bytes) + frame_id
        offset += 8  # stamp.sec + stamp.nanosec
        
        # Read frame_id length and skip frame_id
        if offset + 4 <= len(data):
            frame_id_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4 + frame_id_len
            # Align to 4-byte boundary
            offset = (offset + 3) & ~3
        
        # Read joint names array
        joint_names = []
        if offset + 4 <= len(data):
            names_count = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            for _ in range(names_count):
                if offset + 4 <= len(data):
                    name_len = struct.unpack('<I', data[offset:offset+4])[0]
                    offset += 4
                    if offset + name_len <= len(data):
                        name = data[offset:offset+name_len].decode('utf-8').rstrip('\x00')
                        joint_names.append(name)
                        offset += name_len
                        offset = (offset + 3) & ~3  # Align to 4-byte boundary
        
        # Read positions array
        if offset + 4 <= len(data):
            pos_count = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            if pos_count == len(joint_names) and offset + pos_count * 8 <= len(data):
                positions = []
                for i in range(pos_count):
                    pos = struct.unpack('<d', data[offset:offset+8])[0]
                    positions.append(pos)
                    offset += 8
                
                # Validate positions are reasonable
                if all(not np.isnan(pos) and abs(pos) < 100 for pos in positions):
                    return np.array(positions)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error parsing JointState message: {e}")
        return None

def extract_hand_data(hand_messages) -> List[tuple]:
    """
    Extract hand data from parsed ROS2 messages
    Returns list of (timestamp, hand_data_array) tuples
    """
    # hand_messages already contains (timestamp, parsed_data) tuples
    return hand_messages

def infer_features_from_raw_data(recording_dir: Path, num_cams: int) -> Dict:
    """Infer dataset features from raw recording data."""
    features = {}
    
    # Check for joint data
    joint_dir = recording_dir / "joint"
    if joint_dir.exists():
        joint_files = sorted([f for f in os.listdir(joint_dir) if f.endswith('.txt')])
        if joint_files:
            joint_data = np.loadtxt(joint_dir / joint_files[0])
            joint_dim = joint_data.shape[0] if joint_data.ndim == 1 else joint_data.shape[0]
            features['xarm_joint_pos'] = {"dtype": "float32", "shape": (joint_dim,), "names": [f"xarm_joint_{i}" for i in range(joint_dim)]}
    
    # Check for hand data - first try existing txt files, then ROS2 bags
    hand_dir = recording_dir / "hand"
    if hand_dir.exists() and list(hand_dir.glob("*.txt")):
        hand_files = sorted([f for f in hand_dir.glob("*.txt")])
        if hand_files:
            sample_hand_data = np.loadtxt(hand_files[0])
            hand_dim = sample_hand_data.shape[0] if sample_hand_data.ndim == 1 else sample_hand_data.shape[0]
            features['hand_joint_pos'] = {"dtype": "float32", "shape": (hand_dim,), "names": [f"hand_joint_{i}" for i in range(hand_dim)]}
            logger.info(f"Added hand joint features from txt files with dimension: {hand_dim}")
    else:
        # Skip ROS2 bag parsing for now due to complexity
        logger.info("No hand txt files found, skipping hand data for now")
        # TODO: Add ROS2 bag parsing later if needed
    
    # Add camera features - check first action directory for image dimensions
    action_dirs = [d for d in recording_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if action_dirs:
        action_dir = action_dirs[0]  # Use first action directory
        for cam_idx in range(num_cams):
            camera_dir = action_dir / f"camera_{cam_idx}"
            
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
            
            # Check depth image (optional)
            depth_dir = camera_dir / "depth"
            if depth_dir.exists():
                depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
                if depth_files:
                    depth_image = Image.open(depth_dir / depth_files[0])
                    height, width = depth_image.size[1], depth_image.size[0]
                    features[f'observation.images.cam_{cam_idx}_depth'] = {
                        "dtype": "image", 
                        "shape": (height, width, 1), 
                        "names": ["height", "width", "channels"]
                    }
    
    logger.info(f"Inferred features: {features}")
    return features

def convert_recording_to_lerobot(
    recording_name: str,
    action_name: str,
    recording_dir: Path,
    dataset: LeRobotDataset,
    num_cams: int = 2,
    hand_topic: str = "/orca_hand/joint_angles",
    task_name: str = "manipulation"
) -> bool:
    """Convert a single recording directly to LeRobot format using timestamp alignment."""
    
    logger.info(f"Converting recording: {recording_name}/{action_name}")
    
    # Load joint data (already saved as files)
    joint_dir = recording_dir / "joint"
    joint_timestamps = []
    joint_data_lookup = {}
    
    if joint_dir.exists():
        joint_files = sorted([f for f in os.listdir(joint_dir) if f.endswith('.txt')])
        joint_timestamps = [float(f[:-4]) for f in joint_files]
        # Create lookup table for joint data
        for joint_file in joint_files:
            timestamp = float(joint_file[:-4])
            joint_data_lookup[timestamp] = joint_dir / joint_file
        logger.info(f"Loaded {len(joint_files)} joint files")
    else:
        logger.warning(f"No joint directory found in {recording_dir}")
        return False
    
    # Load camera data
    action_dir = recording_dir / str(action_name)
    if not action_dir.exists():
        logger.warning(f"Action directory {action_dir} not found")
        return False
        
    with open(action_dir / 'timestamps.txt', 'r') as f:
        action_timestamps = f.readlines()
    action_timestamps = [[float(tt) for tt in t.split()[-num_cams:]] for t in action_timestamps]
    
    # Load hand data - first try txt files, then ROS2 bags if needed
    hand_dir = recording_dir / "hand"
    hand_data_list = []
    hand_timestamps = []
    
    if hand_dir.exists() and list(hand_dir.glob("*.txt")):
        hand_files = sorted([f for f in hand_dir.glob("*.txt")])
        for hand_file in hand_files:
            timestamp = float(hand_file.stem)
            hand_data = np.loadtxt(hand_file)
            hand_data_list.append((timestamp, hand_data))
            hand_timestamps.append(timestamp)
        logger.info(f"Loaded {len(hand_data_list)} hand data files")
    else:
        logger.info("No hand txt files found, skipping hand data for now")
        # TODO: Add ROS2 bag parsing later if needed
    
    # Hand timestamps are already extracted above
    
    # Match timestamps and add frames to dataset
    logger.info(f"Processing {len(action_timestamps)} camera frames...")
    successful_matches = 0
    
    for t, action_timestamp in enumerate(action_timestamps):
        master_timestamp = action_timestamp[0]
        
        if t % 50 == 0:  # Progress indicator every 50 frames
            logger.info(f"Processing frame {t}/{len(action_timestamps)}, timestamp: {master_timestamp}")
        
        # Find corresponding joint data
        joint_idx = -1
        if joint_timestamps:
            min_dist = 999
            for jt, joint_timestamp in enumerate(joint_timestamps):
                t_diff = abs(master_timestamp - joint_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    joint_idx = jt
        
        # Skip frame if no joint data
        if joint_idx < 0:
            logger.warning(f"No joint data found for frame {t}")
            continue
        
        # Find corresponding hand data
        hand_idx = -1
        if hand_timestamps:
            min_dist = 999
            for ht, hand_timestamp in enumerate(hand_timestamps):
                t_diff = abs(master_timestamp - hand_timestamp)
                if t_diff < min_dist:
                    min_dist = t_diff
                    hand_idx = ht
        
        # Prepare frame data
        frame_data = {}
        
        # Load xarm joint data (required)
        joint_timestamp = joint_timestamps[joint_idx]
        source_joint_file = joint_data_lookup[joint_timestamp]
        joint_data = np.loadtxt(source_joint_file)
        frame_data['xarm_joint_pos'] = joint_data
        
        # Load hand joint data (optional)
        if hand_idx >= 0:
            _, hand_data = hand_data_list[hand_idx]
            frame_data['hand_joint_pos'] = hand_data
        
        # Load camera images
        for cam_idx in range(num_cams):
            camera_dir = action_dir / f"camera_{cam_idx}"
            
            # RGB image
            rgb_file = camera_dir / "rgb" / f"{t:06d}.jpg"
            if rgb_file.exists():
                rgb_image = Image.open(rgb_file)
                frame_data[f'observation.images.cam_{cam_idx}'] = np.array(rgb_image)
            else:
                logger.warning(f"Missing RGB image for camera {cam_idx}, frame {t}")
                
            # Depth image (optional)
            depth_file = camera_dir / "depth" / f"{t:06d}.png"
            if depth_file.exists():
                depth_image = Image.open(depth_file)
                frame_data[f'observation.images.cam_{cam_idx}_depth'] = np.array(depth_image)
        
        # Add frame to dataset
        try:
            dataset.add_frame(frame_data, task=task_name, timestamp=master_timestamp)
            successful_matches += 1
        except Exception as e:
            logger.error(f"Error adding frame {t}: {e}")
            continue
    
    # Save the episode
    try:
        dataset.save_episode()
        logger.info(f"Successfully converted {successful_matches}/{len(action_timestamps)} frames")
        return True
    except Exception as e:
        logger.error(f"Error saving episode: {e}")
        return False

def direct_convert_to_lerobot(
    recording_dirs: Dict[str, List[str]],
    output_repo_id: str,
    output_root: Optional[Path] = None,
    num_cams: int = 2,
    fps: int = 30,
    robot_type: str = "xarm",
    task_name: str = "manipulation",
    hand_topic: str = "/orca_hand/joint_angles"
) -> bool:
    """Direct conversion from raw recordings to LeRobot format."""
    
    logger.info(f"Starting direct conversion to LeRobot dataset: {output_repo_id}")
    
    # Find first recording to infer features
    first_recording = None
    for recording_name, action_name_list in recording_dirs.items():
        if action_name_list:
            first_recording = root / "log" / "data" / recording_name
            break
    
    if not first_recording or not first_recording.exists():
        logger.error("No valid recordings found")
        return False
    
    # Infer features from first recording
    features = infer_features_from_raw_data(first_recording, num_cams)
    
    # Create LeRobot dataset

    # Create dataset metadata
    LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        root=output_root,
        use_videos=True  # Use videos for efficient storage
    )
    
    # Create actual dataset
    dataset = LeRobotDataset(
        repo_id=output_repo_id,
        root=output_root
    )
    

    
    # Convert each recording
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
                    num_cams, hand_topic, task_name
                ):
                    successful_episodes += 1
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

def main():
    parser = argparse.ArgumentParser(description="Direct conversion from raw recordings to LeRobot format")
    parser.add_argument('--data_id', type=int, required=True, default=0,
                       help='Data configuration ID (0, 1, 2, or 99)')
    parser.add_argument('--output_repo_id', type=str, default="xarm_manipulation_direct",
                       help='Repository ID for the output LeRobot dataset')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Root directory for output dataset (default: ~/.cache/huggingface/lerobot)')
    parser.add_argument('--num_cams', type=int, default=2,
                       help='Number of cameras (default: 2)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second of the dataset (default: 30)')
    parser.add_argument('--robot_type', type=str, default="xarm",
                       help='Robot type identifier (default: xarm)')
    parser.add_argument('--task_name', type=str, default="manipulation",
                       help='Task name for the dataset (default: manipulation)')
    parser.add_argument('--hand_topic', type=str, default='/orca_hand/joint_angles',
                       help='ROS2 topic name for hand data (default: /orca_hand/joint_angles)')
    parser.add_argument('--bimanual', default="False", action='store_true',
                       help='Enable bimanual mode')
    
    args = parser.parse_args()
    
    # Configuration mapping (same as teleop_postprocess_with_hand.py)
    ii = args.data_id
    

    if ii == 0:
        bimanual = False
        num_cams = 2
        dirs = {
            'tao/recording_2': ['1753122120'],  # Actual action directory name
        }
    else:
        logger.error(f"No configuration found for data_id {ii}")
        logger.error("Available data_ids: 1")
        exit(1)
    
    # Override with command line arguments
    if args.num_cams:
        num_cams = args.num_cams
    if args.bimanual:
        bimanual = args.bimanual
    
    # Use provided repo_id or default name
    repo_id = args.output_repo_id
    
    output_root = Path(args.output_root) if args.output_root else None
    
    logger.info(f"Configuration: data_id={ii}, num_cams={num_cams}, bimanual={bimanual}")
    logger.info(f"Output repo_id: {repo_id}")
    logger.info(f"Recording directories: {dirs}")
    
    try:
        success = direct_convert_to_lerobot(
            recording_dirs=dirs,
            output_repo_id=repo_id,
            output_root=output_root,
            num_cams=num_cams,
            fps=args.fps,
            robot_type=args.robot_type,
            task_name=args.task_name,
            hand_topic=args.hand_topic
        )
        
        if success:
            logger.info("Dataset conversion completed successfully!")
        else:
            logger.error("Dataset conversion failed!")
            exit(1)
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        with open(f'{repo_id}_conversion_failed.txt', 'w') as f:
            f.write(f"Error: {e}\n{error_details}\n")
        logger.error(f"Error during conversion: {e}")
        logger.error(f"Full traceback saved to {repo_id}_conversion_failed.txt")
        exit(1)

if __name__ == "__main__":
    main()