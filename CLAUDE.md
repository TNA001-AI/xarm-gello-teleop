# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a teleoperation system for xArm robots using Gello devices and Intel RealSense cameras. The system supports both single-arm and bimanual robot control with real-time camera capture and data recording. The project includes full LeRobot framework integration for advanced robotics policy training and evaluation.

## Key Commands

### Calibration
```bash
# Single arm calibration
python experiments/real_world/calibrate.py --calibrate

# Bimanual calibration  
python experiments/real_world/calibrate.py --calibrate_bimanual
```

### Teleoperation
```bash
# Single arm teleoperation
python experiments/real_world/teleop.py --name exp_name/recording_1

# Bimanual teleoperation
python experiments/real_world/teleop.py --bimanual --name exp_name/recording_1
```

### Data Processing
```bash
# Process recorded data (match timestamps between cameras and robot)
python experiments/real_world/teleop_postprocess.py

# Enhanced processing with ROS2 bag support (includes hand data)
python experiments/real_world/teleop_postprocess_with_hand.py --data_id 2
```

### LeRobot Integration
```bash
# Train policies using LeRobot framework
python experiments/real_world/lerobot/scripts/train.py

# Evaluate trained policies
python experiments/real_world/lerobot/scripts/eval.py

# Visualize datasets
python experiments/real_world/lerobot/scripts/visualize_dataset.py

# Record data using LeRobot interface
python experiments/real_world/lerobot/record.py

# Teleop using LeRobot teleoperators
python experiments/real_world/lerobot/teleoperate.py
```

### Testing
```bash
# Run existing tests (minimal test coverage)
pytest experiments/real_world/gello/dynamixel/tests/test_driver.py
```

## Architecture

### Core Components

**Robot Control (`modules/`):**
- `xarm_wrapper.py` - xArm robot interface and control
- `robot_env.py` - Single arm robot environment
- `robot_env_bimanual.py` - Bimanual robot environment  
- `perception_module.py` - Camera and perception handling
- `planner.py` - Motion planning utilities

**Teleoperation (`modules_teleop/`):**
- `robot_env_teleop.py` - Main teleoperation environment with multiprocessing
- `xarm_controller.py` - Real-time robot control with UDP communication
- `teleop_keyboard.py` - Keyboard-based teleoperation interface
- `perception.py` - Real-time perception during teleop
- `kinematics_utils.py` - Kinematic calculations and transformations
- `udp_util.py` - UDP communication utilities

**Gello Device Integration (`gello/`):**
- `dynamixel/driver.py` - Dynamixel servo control for Gello devices
- `agents/gello_agent.py` - Gello device agent for teleoperation input
- `robots/` - Robot abstractions for Gello integration
- `teleop_gello.py` - Gello teleoperation interface
- `zmq_core/` - ZeroMQ communication components

**Camera System (`camera/`):**
- `multi_realsense.py` - Multi-camera RealSense capture
- `single_realsense.py` - Single camera interface
- `shared_memory/` - Shared memory system for high-performance camera data transfer
- `utils.py` - Camera utility functions

**LeRobot Framework (`lerobot/`):**
- Complete HuggingFace LeRobot integration for robotics ML
- Policy implementations: ACT, Diffusion, TDMPC, VQBet, Pi0, SAC
- Dataset management and processing utilities
- Robot configurations for Koch, Aloha, SO100, SO101, ViperX, Stretch3
- Teleoperator interfaces including gamepad and keyboard control
- Training and evaluation pipeline with extensive configuration support

### Communication Architecture

The system uses multiple communication patterns for different performance requirements:

- **UDP**: Real-time robot control and command distribution (~30Hz)
- **Shared Memory**: High-speed camera data transfer between processes
- **ZeroMQ**: Inter-process communication for Gello devices
- **ROS2**: Integration with robotics ecosystem (optional, for hand data)
- **Multiprocessing**: Concurrent camera capture and robot control

### Data Flow

1. **Calibration**: Place calibration board → run calibration script → generates camera extrinsics/intrinsics
2. **Teleoperation**: Start teleop script → Gello device provides input → robot moves → cameras record → data saved to timestamped files
3. **Processing**: Run postprocess script → matches camera/robot timestamps → creates final dataset
4. **LeRobot Training**: Convert to LeRobot format → train policies → evaluate performance

### Dependencies

The project has no requirements.txt but uses these key dependencies:
- **xarm-python-sdk** - xArm robot control
- **pyrealsense2** - Intel RealSense cameras  
- **dynamixel-sdk** - Gello device control
- **opencv-python** - Computer vision
- **open3d** - 3D point cloud processing
- **torch** - Deep learning framework
- **rclpy, std_msgs** - ROS 2 integration
- **numpy, transforms3d** - Math and transformations
- **kornia** - Differentiable computer vision
- **multiprocess** - High-performance multiprocessing
- **huggingface-hub** - Model and dataset management (LeRobot)

### Hardware Setup Requirements

- Intel RealSense cameras (typically 4 cameras for full coverage)
- xArm robot(s) (single or bimanual setup)
- Gello teleoperation device(s) with Dynamixel servos
- Calibration board for camera-robot calibration

## Data Format

Robot data is saved in two separate directories with timestamped files:

### End-Effector Data (`robot/` directory)
- **Single arm**: 5×3 array (translation + rotation matrix + gripper state)
- **Bimanual**: 9×3 array (two 4×3 robot poses + 1×3 gripper states)

### Joint Position Data (`joint/` directory)
- **Single arm**: 7-element array (joint angles in radians)
- **Bimanual**: 14-element array (left arm 7 joints + right arm 7 joints)

### Hand Data (`hand/` directory)
- Hand joint angles from ROS2 bags (when available)
- Processed from `/orca_hand/joint_angles` topic

Camera data includes RGB/depth streams with matched timestamps.

### Final Dataset Structure
```
experiments/log/data/{exp_name_processed}/
├── episode_XXXX/
│   ├── timestamps.txt          # Camera synchronization with joint/hand indices
│   ├── calibration/           # Camera calibration parameters
│   ├── robot/                 # End-effector poses (timestamped .txt files)
│   ├── joint/                 # Joint angles (timestamped .txt files)
│   ├── hand/                  # Hand data (when available)
│   └── camera_X/
│       ├── rgb/               # RGB images
│       └── depth/             # Depth images
```

## Development Notes

- Working directory should be project root when running scripts
- Calibration must be done before each recording session  
- The system uses shared memory for high-performance camera data transfer
- Robot control happens in real-time with ~30Hz control loop
- All file paths in scripts are relative to `experiments/real_world/`
- LeRobot integration provides extensive configuration options through YAML files
- For ROS2 functionality, ensure ROS2 dependencies are installed and sourced