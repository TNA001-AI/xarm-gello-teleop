# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a teleoperation system for xArm robots using Gello devices and Intel RealSense cameras. The system supports both single-arm and bimanual robot control with real-time camera capture and data recording.

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
- `robot_env_teleop.py` - Main teleoperation environment
- `xarm_controller.py` - Real-time robot control during teleop
- `teleop_keyboard.py` - Keyboard-based teleoperation interface
- `perception.py` - Real-time perception during teleop

**Gello Device Integration (`gello/`):**
- `dynamixel/driver.py` - Dynamixel servo control for Gello devices
- `agents/gello_agent.py` - Gello device agent for teleoperation input
- `robots/` - Robot abstractions for Gello integration

**Camera System (`camera/`):**
- `multi_realsense.py` - Multi-camera RealSense capture
- `single_realsense.py` - Single camera interface
- `shared_memory/` - Shared memory system for high-performance camera data transfer

### Data Flow

1. **Calibration**: Place calibration board → run calibration script → generates camera extrinsics/intrinsics
2. **Teleoperation**: Start teleop script → Gello device provides input → robot moves → cameras record → data saved to timestamped files
3. **Processing**: Run postprocess script → matches camera/robot timestamps → creates final dataset

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

### Hardware Setup Requirements

- Intel RealSense cameras (typically 4 cameras for full coverage)
- xArm robot(s) (single or bimanual setup)
- Gello teleoperation device(s) with Dynamixel servos
- Calibration board for camera-robot calibration

## Data Format

Robot data is saved in two separate directories with timestamped files:

### End-Effector Data (`robot/` directory)
- **Single arm**: 5x3 array (translation + rotation matrix + gripper state)
- **Bimanual**: 9x3 array (two 4x3 robot poses + 1x3 gripper states)

### Joint Position Data (`joint/` directory)
- **Single arm**: 7-element array (joint angles in radians)
- **Bimanual**: 14-element array (left arm 7 joints + right arm 7 joints)

Camera data includes RGB/depth streams with matched timestamps.

## Development Notes

- Working directory should be project root when running scripts
- Calibration must be done before each recording session  
- The system uses shared memory for high-performance camera data transfer
- Robot control happens in real-time with ~30Hz control loop
- All file paths in scripts are relative to `experiments/real_world/`