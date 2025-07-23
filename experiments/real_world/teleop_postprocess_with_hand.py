from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import time
import kornia
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# ROS2 bag imports
try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 dependencies not available. ROS2 bag functionality will be disabled.")
    ROS2_AVAILABLE = False

from utils import get_root, mkdir
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))
# sys.path.append(str(root / "../third-party/sam2"))


def read_ros2_bag(bag_path: str, topic_name: str):
    """
    Read messages from a ROS2 bag file for a specific topic
    Returns list of (timestamp, message) tuples
    """
    if not ROS2_AVAILABLE:
        print("ROS2 not available, skipping bag reading")
        return []
    
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
    
    messages = []
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        if topic == topic_name:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            messages.append((timestamp / 1e9, msg))  # Convert nanoseconds to seconds
    
    return messages


def process_hand_data(hand_messages, save_dir: Path):
    """
    Process hand data from ROS2 bag and save to files
    """
    hand_save_dir = save_dir / "hand"
    mkdir(hand_save_dir, overwrite=True, resume=False)
    
    hand_timestamps = []
    for i, (timestamp, msg) in enumerate(hand_messages):
        hand_timestamps.append(timestamp)
        
        # Extract hand data (JointState message from /orca_hand/joint_angles)
        if hasattr(msg, 'position'):
            hand_data = np.array(msg.position)
        elif hasattr(msg, 'data'):
            hand_data = np.array(msg.data)
        else:
            print(f"Warning: Unknown hand message format at timestamp {timestamp}")
            print(f"Message attributes: {dir(msg)}")
            continue
            
        np.savetxt(hand_save_dir / f"{i:06d}.txt", hand_data)
    
    return hand_timestamps


def match_timestamps_with_bags(name: str, recording_dirs: dict, num_cams: int = 2, 
                              hand_topic: str = "/orca_hand/joint_angles"):
    """
    Enhanced postprocessing that includes ROS2 bag data for joints and hand
    """
    count = 0
    for recording_name, action_name_list in recording_dirs.items():
        for action_name in action_name_list:
            print(f"Processing {recording_name} {action_name}")
            
            save_dir = root / "log" / "data" / name
            mkdir(save_dir, overwrite=False, resume=True)

            episode_save_dir = save_dir / f"episode_{count:04d}"
            mkdir(episode_save_dir, overwrite=True, resume=False)

            episode_save_dir_cam_list = []
            for cam in range(num_cams):
                episode_save_dir_cam = episode_save_dir / f"camera_{cam}"
                episode_save_dir_cam_rgb = episode_save_dir_cam / "rgb"
                episode_save_dir_cam_depth = episode_save_dir_cam / "depth"
                mkdir(episode_save_dir_cam_rgb, overwrite=True, resume=False)
                mkdir(episode_save_dir_cam_depth, overwrite=True, resume=False)
                episode_save_dir_cam_list.append(episode_save_dir_cam)

            episode_save_dir_robot = episode_save_dir / "robot"
            mkdir(episode_save_dir_robot, overwrite=True, resume=False)

            count += 1

            # Load the recording
            recording_dir = root / "log" / "data" / recording_name
            calibration_dir = recording_dir / "calibration"
            subprocess.run(f'cp -r {calibration_dir} {episode_save_dir}', shell=True)
            
            # Load robot data (existing functionality)
            robot_dir = recording_dir / "robot"
            robot_timestamps = sorted([float(d[:-4]) for d in os.listdir(robot_dir)])
            
            # Load joint data (already saved as files)
            joint_dir = recording_dir / "joint"
            joint_timestamps = []
            joint_data_lookup = {}
            # Create joint directory (always create, even if no joint data exists)
            episode_joint_dir = episode_save_dir / "joint"
            
            if joint_dir.exists():
                joint_files = sorted([f for f in os.listdir(joint_dir) if f.endswith('.txt')])
                joint_timestamps = [float(f[:-4]) for f in joint_files]
                # Create lookup table for joint data
                for i, joint_file in enumerate(joint_files):
                    timestamp = float(joint_file[:-4])
                    joint_data_lookup[timestamp] = joint_dir / joint_file
                # Create joint directory (files will be copied with integer names later)
                mkdir(episode_joint_dir, overwrite=True, resume=False)
                print(f"Loaded {len(joint_files)} joint files for processing")
            
            # Load camera data (existing functionality)
            action_dir = recording_dir / str(action_name)
            with open(action_dir / 'timestamps.txt', 'r') as f:
                action_timestamps = f.readlines()
            action_timestamps = [[float(tt) for tt in t.split()[-num_cams:]] for t in action_timestamps]

            # Process ROS2 bag data for hand data only
            bag_path = recording_dir / "ros2_bag"
            hand_timestamps = []
            
            if bag_path.exists() and ROS2_AVAILABLE:
                print(f"Processing ROS2 bag data from {bag_path}")
                
                # Find the actual bag directory (it has timestamp suffix)
                bag_dirs = [d for d in bag_path.iterdir() if d.is_dir()]
                if bag_dirs:
                    actual_bag_path = bag_dirs[0]  # Use first bag directory
                    
                    # Read hand data from ROS2 bag
                    hand_messages = read_ros2_bag(str(actual_bag_path), hand_topic)
                    if hand_messages:
                        hand_timestamps = process_hand_data(hand_messages, episode_save_dir)
                        print(f"Processed {len(hand_messages)} hand messages")
                else:
                    print("No ROS2 bag directories found")
            
            # Match timestamps (existing logic with enhancements)
            print(f"Processing {len(action_timestamps)} camera frames...")
            successful_matches = 0
            for t, action_timestamp in enumerate(action_timestamps):
                master_timestamp = action_timestamp[0]
                timestamps = [master_timestamp]
                
                if t % 50 == 0:  # Progress indicator every 50 frames
                    print(f"Processing frame {t}/{len(action_timestamps)}, timestamp: {master_timestamp}")
                
                # Match camera timestamps (existing logic)
                for cam in range(1, num_cams):
                    min_dist = 999
                    min_dist_tt = -100
                    for tt in range(max(t-1, 0), min(t+1, len(action_timestamps))):
                        t_diff = abs(action_timestamps[tt][cam] - master_timestamp)
                        if t_diff < min_dist:
                            min_dist = t_diff
                            min_dist_tt = tt
                    assert min_dist_tt != -100
                    timestamps.append(action_timestamps[min_dist_tt][cam])
                
                # Find corresponding robot data (existing logic)
                min_dist = 999
                min_dist_tt = -1
                for tt in range(len(robot_timestamps)):
                    t_diff = abs(master_timestamp - robot_timestamps[tt])
                    if t_diff < min_dist:
                        min_dist = t_diff
                        min_dist_tt = tt
                
                if min_dist_tt == -1:
                    print(f'[Warning] No matching robot timestamp found for camera timestamp {master_timestamp}')
                    continue
                    
                # Ensure we can interpolate (need at least one timestamp before or after)
                if min_dist_tt == 0 and len(robot_timestamps) == 1:
                    print('[Warning] Only one robot timestamp available, cannot interpolate')
                    continue
                elif min_dist_tt >= len(robot_timestamps) - 1 and len(robot_timestamps) == 1:
                    print('[Warning] Only one robot timestamp available, cannot interpolate')
                    continue
                
                # Find corresponding joint data
                joint_idx = -1
                if joint_timestamps:
                    min_dist = 999
                    for jt, joint_timestamp in enumerate(joint_timestamps):
                        t_diff = abs(master_timestamp - joint_timestamp)
                        if t_diff < min_dist:
                            min_dist = t_diff
                            joint_idx = jt
                
                # Find corresponding hand data
                hand_idx = -1
                if hand_timestamps:
                    min_dist = 999
                    for ht, hand_timestamp in enumerate(hand_timestamps):
                        t_diff = abs(master_timestamp - hand_timestamp)
                        if t_diff < min_dist:
                            min_dist = t_diff
                            hand_idx = ht
                
                # Save matched timestamps with joint and hand indices
                with open(episode_save_dir / "timestamps.txt", 'a') as f:
                    timestamp_line = ' '.join([str(tt) for tt in timestamps])
                    if joint_idx >= 0:
                        timestamp_line += f' joint:{joint_idx}'
                    if hand_idx >= 0:
                        timestamp_line += f' hand:{hand_idx}'
                    f.write(timestamp_line + '\n')
                
                # Copy camera data (existing logic)
                for cam in range(num_cams):
                    source_dir = action_dir / f"camera_{cam}" / "rgb" / f"{t:06d}.jpg"
                    target_dir = episode_save_dir_cam_list[cam] / "rgb" / f"{t:06d}.jpg"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)

                    source_dir = action_dir / f"camera_{cam}" / "depth" / f"{t:06d}.png"
                    target_dir = episode_save_dir_cam_list[cam] / "depth" / f"{t:06d}.png"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)
                
                # Interpolate robot motion (existing logic)
                if min_dist_tt == 0:
                    # Use first two timestamps
                    tt1 = 0
                    tt2 = min(1, len(robot_timestamps) - 1)
                elif min_dist_tt >= len(robot_timestamps) - 1:
                    # Use last two timestamps  
                    tt2 = len(robot_timestamps) - 1
                    tt1 = max(0, tt2 - 1)
                elif master_timestamp > robot_timestamps[min_dist_tt]:
                    # Interpolate forward
                    tt1 = min_dist_tt
                    tt2 = min(min_dist_tt + 1, len(robot_timestamps) - 1)
                else:
                    # Interpolate backward
                    tt1 = max(0, min_dist_tt - 1)
                    tt2 = min_dist_tt
                
                # Skip if same timestamp (no interpolation possible)
                if tt1 == tt2:
                    weight = 0
                else:
                    weight = (master_timestamp - robot_timestamps[tt1]) / (robot_timestamps[tt2] - robot_timestamps[tt1] + 1e-6)
                robot_data1 = np.loadtxt(robot_dir / f"{robot_timestamps[tt1]:.3f}.txt")
                robot_data2 = np.loadtxt(robot_dir / f"{robot_timestamps[tt2]:.3f}.txt")

                @torch.no_grad()
                def interpolate_matrices(mat1, mat2, weight):
                    mat1 = torch.tensor(mat1)
                    mat2 = torch.tensor(mat2)
                    quat1 = kornia.geometry.conversions.rotation_matrix_to_quaternion(mat1)
                    quat2 = kornia.geometry.conversions.rotation_matrix_to_quaternion(mat2)
                    quat1 = kornia.geometry.quaternion.Quaternion(quat1)
                    quat2 = kornia.geometry.quaternion.Quaternion(quat2)
                    quat = quat1.slerp(quat2, weight).data
                    mat = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat)
                    mat = mat.numpy()
                    return mat

                assert robot_data1.shape[0] in [1, 4, 5, 9]  # Support 4-row format (1 pos + 3 rot), bi-manual (2 * (1 pos + 3 rot) + 1 gripper) or single arm (1 pos + 3 rot + 1 gripper or 1 pos)
                if robot_data1.shape[0] == 4:  # 4x3 format: translation + rotation matrix (no gripper)
                    robot_data1_trans = robot_data1[0:1]  # First row is translation
                    robot_data1_rot = robot_data1[1:4]    # Next 3 rows are rotation matrix
                    
                    robot_data2_trans = robot_data2[0:1]
                    robot_data2_rot = robot_data2[1:4]
                    
                    robot_data_trans = robot_data1_trans * (1 - weight) + robot_data2_trans * weight
                    robot_data_rot = interpolate_matrices(robot_data1_rot[None], robot_data2_rot[None], weight)[0]
                    
                    robot_data = np.concatenate([robot_data_trans, robot_data_rot], axis=0)  # (4, 3)
                    
                elif robot_data1.shape[0] > 4:  # 5 or 9 (with gripper)
                    gripper1 = robot_data1[-1]
                    robot_data1 = robot_data1[:-1]
                    robot_data1 = robot_data1.reshape(-1, 4, 3)
                    robot_data1_trans = robot_data1[:, 0]
                    robot_data1_rot = robot_data1[:, 1:]

                    gripper2 = robot_data2[-1]
                    robot_data2 = robot_data2[:-1]
                    robot_data2 = robot_data2.reshape(-1, 4, 3)
                    robot_data2_trans = robot_data2[:, 0]
                    robot_data2_rot = robot_data2[:, 1:]

                    robot_data_trans = robot_data1_trans * (1 - weight) + robot_data2_trans * weight
                    robot_data_rot = interpolate_matrices(robot_data1_rot, robot_data2_rot, weight)
                    robot_data_gripper = gripper1 * (1 - weight) + gripper2 * weight

                    robot_data = np.concatenate([robot_data_trans[:, None], robot_data_rot], axis=1)  # (-1, 4, 3)
                    robot_data = robot_data.reshape(-1, 3)
                    robot_data = np.concatenate([robot_data, robot_data_gripper.reshape(1, 3)], axis=0)
                
                else:
                    robot_data = robot_data1 * (1 - weight) + robot_data2 * weight

                np.savetxt(episode_save_dir_robot / f"{t:06d}.txt", robot_data)
                
                # Copy joint data with integer filename
                if joint_idx >= 0 and joint_timestamps:
                    joint_timestamp = joint_timestamps[joint_idx]
                    source_joint_file = joint_data_lookup[joint_timestamp]
                    target_joint_file = episode_joint_dir / f"{t:06d}.txt"
                    subprocess.run(f'cp {source_joint_file} {target_joint_file}', shell=True)
                
                successful_matches += 1
                
            print(f"Successfully processed {successful_matches}/{len(action_timestamps)} frames")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=2)
    parser.add_argument('--num_cams', type=int, default=2)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--hand_topic', type=str, default='/orca_hand/joint_angles',
                       help='ROS2 topic name for hand data')
    args = parser.parse_args()

    if True:
        ii = int(args.data_id)

        eef_T = [0, 0, -0.01]
        cameras = None
        dirs = None

        if ii == 99:
            args.name = 'teleop_test_rl_with_hand_processed'
            args.bimanual = True
            args.num_cams = 2
            dirs = {
                'teleop_test_rl/rel_traj_mode1': ['1739216010'],
                'teleop_test_rl/rel_traj_mode2': ['1739216080'],
                'teleop_test_rl/rel_traj_mode3': ['1739216498'],
            }

        elif ii == 0:
            args.name = 'test_with_hand_processed'
            args.bimanual = False
            args.num_cams = 4
            dirs = {
                'test/recording_1': ['1740075589', '1740075610', '1740075626', '1740075645', '1740075660', '1740075681'],
                'test/recording_3': ['1740075844', '1740075853', '1740075863', '1740075873', '1740075883', '1740075898', '1740075910', '1740075923', '1740075972']
            }
        
        elif ii == 1:
            args.name = 'test_run_with_hand_processed'
            args.bimanual = False
            args.num_cams = 4
            dirs = {
                'test_run/recording_1': ['1740094945', '1740094960'],
            }

        elif ii == 2:
            args.name = 'tao_recording_2_processed'
            args.bimanual = False
            args.num_cams = 2
            dirs = {
                'tao/recording_2': ['1753122120'],  # Actual action directory name
            }

        else:
            print(f"Error: No configuration found for data_id {ii}")
            print("Available data_ids: 0, 1, 2, 99")
            exit(1)

        try:
            match_timestamps_with_bags(args.name, dirs, num_cams=args.num_cams,
                                     hand_topic=args.hand_topic)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            with open(f'{args.name}_failed.txt', 'a') as f:
                f.write(f"{ii}: {e}\n{error_details}\n")
            print(f"Error in {args.name}: {e}")
            print(f"Full traceback: {error_details}")
#   camera0_timestamp camera1_timestamp joint:index hand:index