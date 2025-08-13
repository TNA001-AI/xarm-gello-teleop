#!/usr/bin/env python3
"""
Diffusion Policy Rollout for Hand-Arm Robot Control

This script rolls out a trained diffusion policy to control both xArm and hand simultaneously.
Uses the existing teleoperation infrastructure for camera capture and robot control.

Usage:
    python experiments/real_world/diffusion_policy_rollout.py \
        --policy_path /path/to/policy.safetensors \
        --exp_name my_rollout \
        --robot_ip 192.168.1.196 \
        --duration 120 \
        --device cuda \
        --verbose

Inputs:
    - Two cameras (RGB images)
    - xArm joint states (7 DOF)
    - Hand joint states (from ROS2, 17 DOF)

Outputs:
    - xArm joint commands (via UDP to XarmController child process)
    - Hand joint commands (via ROS2, 17 DOF)
"""

import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
import traceback

import numpy as np
import torch
import cv2
import pygame
from collections import deque

# Add project root to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add root for absolute imports

# Import teleoperation modules
from modules_teleop.xarm_controller import XarmController
from camera.multi_realsense import MultiRealsense
from utils import get_root, mkdir
from multiprocess.managers import SharedMemoryManager
import pickle

# Import kinematics helper
try:
    from modules_teleop.kinematics_utils import KinHelper
    KINEMATICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: KinHelper not available: {e}")
    KINEMATICS_AVAILABLE = False

# NEW: UDP sender and ports to talk to the XarmController child process
from modules_teleop.udp_util import udpSender
from modules_teleop.common.communication import XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R

CHECK_POINT_PATH = "/data/xarm_orca_diffusion/checkpoints/last/pretrained_model"

# ROS2 imports for hand control
try:
    import rclpy
    from rclpy.node import Node
    ROS2_AVAILABLE = True
    from sensor_msgs.msg import JointState
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
except ImportError:
    print("Warning: ROS2 not available, hand control disabled")
    ROS2_AVAILABLE = False

# LeRobot imports
try:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from typing import Optional, Dict, Any
    LEROBOT_AVAILABLE = True
except ImportError:
    print("Warning: LeRobot not available")
    LEROBOT_AVAILABLE = False


class HandController(Node):
    """
    Minimal ROS2 bridge.

    send_hand_command(hand_16, wrist_rad=None)
      - hand_16: numpy array of 16 finger joint radians in RIGHT_NAMES_16 order
      - wrist_rad: optional radians value; published as the 8th joint
        to /gello/act_joint_states

    get_hand_state()
      - returns a dict with:
          {
            "fingers": np.ndarray(16) or None,   # in RIGHT_NAMES_16 order if all present
            "wrist": float or None,              # if present in obs
            "raw": {"names": [...], "positions": [...]},  # raw JointState mirror
          }
    """

    # Exact publish order expected by your hardware on /joint_states (16 dims)
    RIGHT_NAMES_16 = [
        "right_thumb_mcp", "right_thumb_abd", "right_thumb_pip", "right_thumb_dip",
        "right_index_mcp", "right_index_pip", "right_index_abd",
        "right_middle_mcp", "right_middle_pip", "right_middle_abd",
        "right_ring_mcp", "right_ring_pip", "right_ring_abd",
        "right_pinky_mcp", "right_pinky_pip", "right_pinky_abd",
    ]

    def __init__(self):
        super().__init__("hand_controller")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Command publishers
        self.joint_cmd_pub = self.create_publisher(JointState, "/joint_states", qos)
        self.wrist_cmd_pub = self.create_publisher(JointState, "/gello/act_joint_states", qos)

        # Observation subscriber (whatever Orca publishes)
        self.obs_sub = self.create_subscription(
            JointState, "/orca_hand/obs_joint_states", self._obs_cb, qos
        )

        self._last_obs_names: list[str] = []
        self._last_obs_pos: list[float] = []
        self._lock = threading.Lock()

    # -------- Observation --------
    def _obs_cb(self, msg: JointState):
        with self._lock:
            self._last_obs_names = list(msg.name)
            # ensure float
            self._last_obs_pos = [float(p) for p in msg.position]

    def get_hand_state(self) -> Optional[np.ndarray]:
        """Return latest observation as numpy array of 17 DOF (16 fingers + 1 wrist) for policy rollout."""
        with self._lock:
            names = list(self._last_obs_names)
            pos = list(self._last_obs_pos)

        if not names:
            return None

        name_to_pos = {n: p for n, p in zip(names, pos)}

        # Try to assemble fingers in RIGHT_NAMES_16 order
        try:
            fingers = np.array([name_to_pos[n] for n in self.RIGHT_NAMES_16], dtype=np.float32)
        except KeyError:
            # Some finger names missing; return None to use zeros
            return None

        # Wrist is optional; use 0.0 if not present
        wrist = name_to_pos.get("wrist", 0.0)
        wrist = float(wrist)

        # Combine into 17 DOF: 16 fingers + 1 wrist
        hand_state = np.concatenate([fingers, np.array([wrist], dtype=np.float32)])
        
        return hand_state

    # -------- Command --------
    def send_hand_command(self, hand_joints: np.ndarray, wrist_rad: Optional[float] = None):
        """
        Publish 16 finger joints (radians) in RIGHT_NAMES_16 order to /joint_states,
        and optionally publish wrist (as the 8th joint) to /gello/act_joint_states.
        """
        hand = np.asarray(hand_joints, dtype=np.float32).flatten()
        if hand.size != 16:
            raise ValueError(f"hand_joints must have 16 elements (got {hand.size})")

        # Fingers -> /joint_states
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(self.RIGHT_NAMES_16)
        js.position = hand.tolist()
        self.joint_cmd_pub.publish(js)

        # Optional wrist -> /gello/act_joint_states (index 7)
        if wrist_rad is not None:
            wmsg = JointState()
            wmsg.header.stamp = self.get_clock().now().to_msg()
            pos = [0.0] * 8
            pos[7] = float(wrist_rad)
            wmsg.position = pos
            self.wrist_cmd_pub.publish(wmsg)



class DiffusionPolicyRollout:
    """Main class for diffusion policy rollout"""

    def __init__(
        self,
        policy_path: str,
        exp_name: str = "diffusion_rollout",
        robot_ip: str = "192.168.1.196",
        camera_serial_numbers: Optional[list] = None,
        action_horizon: int = 16,
        history_length: int = 2,
        device: str = "cuda",
        verbose: bool = True,
        debug: bool = True
    ):
        self.policy_path = Path(policy_path)
        self.exp_name = exp_name
        self.robot_ip = robot_ip
        self.camera_serial_numbers = camera_serial_numbers
        self.image_width = None
        self.image_height = None
        self.image_resolution = None
        self.action_horizon = action_horizon
        self.history_length = history_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.debug = debug

        self.root = get_root(__file__)

        # Initialize components
        self.policy = None
        self.xarm_controller = None
        self.hand_controller = None
        self.camera_system = None
        self.shm_manager = None

        # UDP sender for xArm commands (child process intake)
        self.xarm_cmd_sender = None

        # Latest robot joints cache for composing absolute targets
        self._last_xarm_joints: Optional[np.ndarray] = None
        self._last_wrist_state: Optional[float] = None

        self.step_count = 0

        # Control flags
        self.running = False
        self.policy_thread = None  # not used in new flow

        # Visualization components
        self.visualization_thread = None
        self.display_queue = deque(maxlen=10)
        self.coordinate_history = deque(maxlen=100)
        self.predicted_coordinates = None
        
        # Camera calibration data for end effector visualization
        self.camera_intrinsics = None
        self.camera_extrinsics = None
        self.robot_base_to_world = None
        
        # Forward kinematics helper
        self.kin_helper = None
        
        # End effector point in gripper frame - 22cm Z offset
        self.eef_point = np.array([[0.0, 0.0, 0.22]])  # 22cm Z offset

        # Latest single-step observation cache
        self.image_feature_keys = None  # to be filled from policy config
        self.latest_obs = None

    # ---------------- Policy / Cameras / Robot setup ----------------

    def load_policy(self):
        """Load the trained diffusion policy using LeRobot factory and align image config."""
        print(f"Loading policy from: {self.policy_path}")

        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is not available. Please install lerobot package.")

        # Check if policy path exists
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy path does not exist: {self.policy_path}")

        try:
            # Load policy config
            self.policy = DiffusionPolicy.from_pretrained(str(self.policy_path))
            self.policy.to(self.device)
            self.policy.eval()

            # Read expected image feature keys and shape (C,H,W)
            self.image_feature_keys = list(self.policy.config.image_features.keys())
            img_shape = next(iter(self.policy.config.image_features.values())).shape  # (C,H,W)
            # Align rollout resize with policy expectation
            self.image_height, self.image_width = img_shape[1], img_shape[2]
            self.image_resolution = (self.image_width, self.image_height)

            if self.verbose:
                print(f"Image feature keys: {self.image_feature_keys}")
                print(f"Policy expects image size (H,W) = ({self.image_height},{self.image_width})")

        except Exception as e:
            raise RuntimeError(f"Failed to load policy: {e}. Please check policy path and files.")

    def setup_robot_control(self):
        """Initialize robot control systems"""
        print("Setting up robot control...")

        # Initialize xArm controller (child process)
        self.xarm_controller = XarmController(
            start_time=time.time(),
            ip=self.robot_ip,
            gripper_enable=False,
            mode="3D",
            command_mode="rollout",  # Use joint control for diffusion policy
            robot_id=-1,            # -1 => use LEFT port in this project setup
            verbose=self.verbose
        )

        # Prepare UDP sender to the controller's intake port
        port = XARM_CONTROL_PORT_L if self.xarm_controller.robot_id <= 0 else XARM_CONTROL_PORT_R
        self.xarm_cmd_sender = udpSender(port=port)

        # Initialize hand controller if ROS2 is available
        if ROS2_AVAILABLE:
            rclpy.init()
            self.hand_controller = HandController()
            print("Hand controller initialized")
        else:
            print("ROS2 not available, hand control disabled")

    def load_camera_calibration(self):
        """Load camera calibration data for end effector visualization"""
        try:
            calibration_dir = self.root / "log" / "latest_calibration"
            
            # Load base calibration (robot base to world transform)
            with open(calibration_dir / "base.pkl", 'rb') as f:
                base = pickle.load(f)
            R_base2world = base['R_base2world']
            t_base2world = base['t_base2world']
            base2world_mat = np.eye(4)
            base2world_mat[:3, :3] = R_base2world
            base2world_mat[:3, 3] = t_base2world
            self.robot_base_to_world = base2world_mat
            
            # Load camera extrinsics
            with open(calibration_dir / "rvecs.pkl", 'rb') as f:
                rvecs = pickle.load(f)
            with open(calibration_dir / "tvecs.pkl", 'rb') as f:
                tvecs = pickle.load(f)
            
            if self.verbose:
                print(f"Loading calibration for camera system with {len(self.camera_system.cameras)} cameras")
                print(f"Available calibration keys: {list(rvecs.keys())}")
            
            extr_list = []
            serial_numbers = list(self.camera_system.cameras.keys())
            for serial in serial_numbers:
                if serial in rvecs:
                    R_world2cam = cv2.Rodrigues(rvecs[serial])[0]
                    t_world2cam = tvecs[serial][:, 0]
                    extr_mat = np.eye(4)
                    extr_mat[:3, :3] = R_world2cam
                    extr_mat[:3, 3] = t_world2cam
                    extr_list.append(extr_mat)
                else:
                    print(f"Warning: No calibration data found for camera {serial}")
                    extr_list.append(np.eye(4))  # Identity as fallback
            
            self.camera_extrinsics = np.stack(extr_list) if extr_list else None
            
            if self.verbose:
                print("Camera calibration loaded successfully")
            
        except Exception as e:
            print(f"Warning: Failed to load camera calibration: {e}")
            print("End effector visualization will be disabled")
            self.camera_extrinsics = None
            self.robot_base_to_world = None

    def setup_cameras(self):
        """Initialize camera system (resolution aligned with policy config)"""
        print("Setting up cameras...")

        # Initialize shared memory manager (critical for camera data)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        # Use auto-detection if serial numbers not specified
        if self.camera_serial_numbers is None:
            self.camera_serial_numbers = None  # Will auto-detect all available cameras

        # Initialize camera system with shared memory manager
        self.camera_system = MultiRealsense(
            shm_manager=self.shm_manager,
            serial_numbers=self.camera_serial_numbers,
            resolution=self.image_resolution,         # align with policy expectation
            capture_fps=30,
            put_fps=30,
            enable_depth=False,
            enable_color=True,
            get_max_k=30,
            verbose=False
        )

        # Set camera parameters (matching teleoperation settings)
        print("Setting camera exposure and white balance...")
        self.camera_system.set_exposure(exposure=100, gain=60)  # 100: bright, 60: dark
        self.camera_system.set_white_balance(3800)
        
        # Load camera calibration after camera system is initialized
        self.load_camera_calibration()
        
        # Initialize forward kinematics helper
        if KINEMATICS_AVAILABLE:
            try:
                self.kin_helper = KinHelper(robot_name='xarm7', headless=True)
                if self.verbose:
                    print("Forward kinematics helper initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize kinematics helper: {e}")
                self.kin_helper = None
        else:
            print("Warning: Kinematics utilities not available, end effector visualization will be disabled")
            self.kin_helper = None
        
        # Note: Camera intrinsics will be loaded after camera system starts

    # ---------------- Observation helpers (single-step) ----------------

    def draw_end_effector_frame(self, images: Dict[str, np.ndarray], joints: np.ndarray, is_predicted: bool = True) -> Dict[str, np.ndarray]:
        """
        Draw end effector frame on camera images using forward kinematics.
        Args:
            images: Dict of camera serial -> {'color': HxWx3 BGR image}
            joints: (7,) array of joint angles in radians
            is_predicted: True for predicted frame (RGB axes), False for current frame (different colors)
        Returns:
            Dict of camera serial -> {'color': HxWx3 BGR image with visualization}
        """
        # Check what components are available
        components_status = {
            'kin_helper': self.kin_helper is not None,
            'camera_extrinsics': self.camera_extrinsics is not None,
            'robot_base_to_world': self.robot_base_to_world is not None,
            'camera_intrinsics': self.camera_intrinsics is not None
        }
        
        if (self.kin_helper is None or 
            self.camera_extrinsics is None or 
            self.robot_base_to_world is None or 
            self.camera_intrinsics is None):
            # Debug: Show which components are missing
            if self.verbose and self.step_count % 200 == 0:
                missing = []
                if self.kin_helper is None: missing.append("kin_helper")
                if self.camera_extrinsics is None: missing.append("extrinsics")
                if self.robot_base_to_world is None: missing.append("base_transform")
                if self.camera_intrinsics is None: missing.append("intrinsics")
                print(f"EE visualization disabled - missing: {missing}")
            return images  # Return original images if calibration not available
        
        try:
            # Compute forward kinematics for joint angles
            eef_pose = self.kin_helper.compute_fk_sapien_links(
                joints, [self.kin_helper.sapien_eef_idx]
            )[0]  # 4x4 transformation matrix
            
            # Transform end effector points to world coordinates (matching teleop.py approach)
            eef_points_gripper = np.concatenate([self.eef_point, np.ones((self.eef_point.shape[0], 1))], axis=1)  # (n, 4)
            eef_points_world = (self.robot_base_to_world @ eef_pose @ eef_points_gripper.T).T[:, :3]  # (n, 3)
            eef_points_world_vis = np.concatenate([eef_points_world, np.ones((eef_points_world.shape[0], 1))], axis=1)  # (n, 4)
            
            # Set colors and style based on frame type
            if is_predicted:
                # Predicted frame: RGB colors, thinner lines
                eef_axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue for X, Y, Z
                circle_color = (0, 255, 255)  # Yellow circle
                circle_radius = 2
                line_thickness = 2
                axis_length = 0.1  # 10cm
            else:
                # Current frame: Different colors, thicker lines
                eef_axis_colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255)]  # BGR: Magenta, Cyan, Yellow for X, Y, Z
                circle_color = (255, 255, 255)  # White circle
                circle_radius = 3
                line_thickness = 3
                axis_length = 0.05  # 5cm (shorter)
            
            # Define end effector axes for visualization (same as teleop.py)
            eef_axis = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # X, Y, Z axes scaled to 10cm
            eef_axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue for X, Y, Z
            
            # Draw on each camera image
            annotated_images = {}
            camera_serials = list(images.keys())
            
            for idx, (serial, img_data) in enumerate(images.items()):
                img = img_data['color'].copy()  # Copy to avoid modifying original
                
                if idx >= len(self.camera_extrinsics) or idx >= len(self.camera_intrinsics):
                    annotated_images[serial] = {'color': img}
                    continue
                
                extr = self.camera_extrinsics[idx]  # World to camera transform
                intr = self.camera_intrinsics[idx]   # Camera intrinsic matrix
                
                # Project end effector point to image
                point = eef_points_world_vis[0]  # (4,) - already has homogeneous coordinate
                point = extr @ point  # Transform to camera coordinates
                
                if point[2] > 0:  # Check if in front of camera
                    point = point[:3] / point[2]  # Perspective division
                    point = intr @ point  # Apply intrinsics
                    
                    # Draw circle with bounds checking for visibility
                    x, y = int(point[0]), int(point[1])
                    
                    # Debug: Print EE position occasionally
                    if self.verbose and self.step_count % 100 == 0:
                        in_bounds = 0 <= x < img.shape[1] and 0 <= y < img.shape[0]
                        print(f"EE frame cam_{idx}: ({x}, {y}) depth={point[2]:.3f} in_bounds={in_bounds}")
                    
                    # Draw the circle even if outside bounds, but clamp for visibility
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        # Normal case: point is within image
                        cv2.circle(img, (x, y), circle_radius, circle_color, -1)
                    else:
                        # Point is outside image - draw at clamped position with different color
                        x_clamped = max(5, min(x, img.shape[1] - 5))
                        y_clamped = max(5, min(y, img.shape[0] - 5))
                        # Use red for out-of-bounds predicted, magenta for out-of-bounds current
                        out_of_bounds_color = (0, 0, 255) if is_predicted else (255, 0, 255)
                        cv2.circle(img, (x_clamped, y_clamped), circle_radius + 1, out_of_bounds_color, -1)
                
                # Draw end effector axes 
                eef_axis_teleop = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # (3, 4) format like teleop
                point_orig = eef_points_gripper[0]  # Original point in gripper frame (4,)
                
                # Always draw a small test circle to verify drawing works  
                cv2.circle(img, (30, 30), 3, (0, 255, 255), -1)  # Yellow test dot every frame
                
                for axis, color in zip(eef_axis_teleop, eef_axis_colors):
                    # Create axis endpoint
                    eef_point_axis = point_orig + axis_length * axis  # Variable axis length
                    eef_point_axis_world = (self.robot_base_to_world @ eef_pose @ eef_point_axis)
                    eef_point_axis_world = extr @ eef_point_axis_world
                    
                    if point[2] > 0 and eef_point_axis_world[2] > 0:
                        eef_point_axis_world = eef_point_axis_world[:3] / eef_point_axis_world[2]
                        eef_point_axis_world = intr @ eef_point_axis_world
                        
                        # Draw line from origin to axis endpoint
                        origin_x, origin_y = int(point[0]), int(point[1])
                        axis_x, axis_y = int(eef_point_axis_world[0]), int(eef_point_axis_world[1])
                        
                        # Clamp line endpoints to image bounds for visibility
                        origin_x = max(0, min(origin_x, img.shape[1] - 1))
                        origin_y = max(0, min(origin_y, img.shape[0] - 1))
                        axis_x = max(0, min(axis_x, img.shape[1] - 1))
                        axis_y = max(0, min(axis_y, img.shape[0] - 1))
                        
                        cv2.line(img, (origin_x, origin_y), (axis_x, axis_y), color, line_thickness)
                                
                # Draw step counter
                cv2.putText(img, f"Step: {self.step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                annotated_images[serial] = {'color': img}
            
            return annotated_images
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: EE frame drawing failed: {e}")
            return images


    def draw_trajectory_3d_window(self, action_sequence: np.ndarray, current_joints: np.ndarray):
        """
        Display trajectory in a BLOCKING Open3D window showing all 16 trajectory points.
        Window will pause execution until closed.
        """
        
        if self.kin_helper is None or self.robot_base_to_world is None:
            return
        
        try:
            import open3d as o3d
            
            # Compute ALL 16 trajectory points (or however many are in action_sequence)
            trajectory_points = []
            trajectory_frames = []
            cumulative_joints = current_joints.copy()
            
            # Add current position as starting point
            current_eef_pose = self.kin_helper.compute_fk_sapien_links(
                current_joints, [self.kin_helper.sapien_eef_idx]
            )[0]
            eef_point_gripper = np.concatenate([self.eef_point[0], [1]])
            current_eef_world = (self.robot_base_to_world @ current_eef_pose @ eef_point_gripper)[:3]
            trajectory_points.append(current_eef_world)
            trajectory_frames.append(self.robot_base_to_world @ current_eef_pose)
            
            # Compute trajectory points
            num_steps = action_sequence.shape[0]  # Use all available steps
            
            for step_idx in range(num_steps):
                gello_deltas = action_sequence[step_idx, 16:24][:7]
                
                
                cumulative_joints = cumulative_joints + gello_deltas
                
                eef_pose = self.kin_helper.compute_fk_sapien_links(
                    cumulative_joints, [self.kin_helper.sapien_eef_idx]
                )[0]
                
                eef_point_world = (self.robot_base_to_world @ eef_pose @ eef_point_gripper)[:3]
                trajectory_points.append(eef_point_world)
                trajectory_frames.append(self.robot_base_to_world @ eef_pose)
            
            trajectory_points = np.array(trajectory_points)
            
            # Create geometries for visualization
            geometries = []
            
           
            # 2. Robot base frame (make it more prominent with larger size and different color)
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            base_frame.transform(self.robot_base_to_world)
            geometries.append(base_frame)
            
            
            
            # 3. Current/Start end effector frame (green)
            current_ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            current_ee_frame.transform(trajectory_frames[0])
            current_ee_frame.paint_uniform_color([0, 1, 0])
            geometries.append(current_ee_frame)
            
        
            
            # 4. Final/End end effector frame (red)
            if len(trajectory_frames) > 1:
                final_ee_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                final_ee_frame.transform(trajectory_frames[-1])
                final_ee_frame.paint_uniform_color([1, 0, 0])
                geometries.append(final_ee_frame)
                
                

            
            # BLOCKING visualization - execution will pause until window is closed
            if self.verbose:
                print(f"[3D TRAJECTORY] Opening BLOCKING Open3D window with {len(geometries)} geometries")
                print(f"[3D TRAJECTORY] Showing all {len(trajectory_points)} trajectory points")
                print("[3D TRAJECTORY] Frame Labels: WORLD=Yellow, BASE=Cyan, START=Green, END=Red")
                print("[3D TRAJECTORY] Close the window to continue execution...")
            
            # Use blocking draw_geometries - this will pause execution
            window_title = (f"Robot Trajectory Visualization | Step {self.step_count} ")
            
            # Create visualizer with custom view settings
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_title, width=1200, height=900, left=100, top=100)
            
            # Add all geometries
            for geom in geometries:
                vis.add_geometry(geom)
            
            # Set camera to look at robot base with proper orientation
            ctr = vis.get_view_control()
            
            # Set the center of view to robot base position
            base_position = self.robot_base_to_world[:3, 3] if self.robot_base_to_world is not None else np.array([0, 0, 0])
            ctr.set_lookat(base_position + [0,0,-0.3])
            
            # Set camera position: X to the right, Z up
            # Camera should be positioned to have a good view with X pointing right and Z up
            # Front vector: looking from diagonal position
            ctr.set_front([0.3, -0.7, 0])  # Look from front-right, slightly above
            
            # Up vector: Z axis points up in view
            ctr.set_up([0, 0, -1])
            
            # Set zoom to see the full trajectory
            ctr.set_zoom(0.9)
            
            # Render and run
            vis.run()
            vis.destroy_window()
            
            
        except Exception as e:
            if self.verbose:
                print(f"[3D TRAJECTORY ERROR] {e}")

    def preprocess_images_single(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Map raw camera dict to policy's image feature keys as single-frame tensors.
        Input: images = { <serial>: {'color': HxWx3 BGR} }
        Output: { <policy_key>: (3,H,W) float32 in [0,1] }  —— no batch dim
        """

        # Prefer explicit serial order from CLI to ensure mapping stability
        camera_serials = self.camera_serial_numbers
        processed = {}

        for idx, pol_key in enumerate(self.image_feature_keys):
            serial = camera_serials[idx]
            img = images[serial]['color']  # HxWx3 (BGR)
            # Resize to policy input size
            img = cv2.resize(img, self.image_resolution)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # To tensor (C,H,W) in [0,1]
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            assert img_t.shape == (3, self.image_height, self.image_width), \
                f"Expected (3,{self.image_height},{self.image_width}), got {img_t.shape}"
            processed[pol_key] = img_t

        return processed

    def create_step_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build a single-step batch for policy.select_action():
          - image keys: (1,C,H,W)
          - state: (1,state_dim)
        """
        if self.latest_obs is None:
            return None
        batch = {}
        # Images
        for k in self.image_feature_keys:
            batch[k] = self.latest_obs[k].unsqueeze(0).to(self.device)  # (1,C,H,W)
        # State
        batch['observation.state'] = self.latest_obs['observation.state'].unsqueeze(0).to(self.device)  # (1,D)
        return batch

    # ---------------- Robot state ----------------

    def get_robot_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current robot state with separate xarm and hand components"""
        # Get xArm joint state (7 DOF) from controller's queue
        xarm_joints = None
        if self.xarm_controller.is_controller_alive and not self.xarm_controller.cur_qpos_q.empty():
            xarm_joints = self.xarm_controller.cur_qpos_q.get()

        if xarm_joints is None:
            return None

        # Assert xArm joint dimensions
        assert isinstance(xarm_joints, np.ndarray), f"Expected numpy array for xarm joints, got {type(xarm_joints)}"
        assert xarm_joints.shape == (7,), f"Expected 7 DOF for xArm, got shape {xarm_joints.shape}"

        # Get hand joint state (17 DOF for observation)
        if self.hand_controller is not None:
            hand_joints = self.hand_controller.get_hand_state()
            if hand_joints is None:
                hand_joints = np.zeros(17, dtype=np.float32)
        else:
            hand_joints = np.zeros(17, dtype=np.float32)

        # Assert hand joint dimensions
        assert isinstance(hand_joints, np.ndarray), f"Expected numpy array for hand joints, got {type(hand_joints)}"
        assert hand_joints.shape == (17,), f"Expected 17 DOF for hand, got shape {hand_joints.shape}"

        robot_state = {
            'observation.xarm_joint_pos': xarm_joints,
            'observation.hand_joint_pos': hand_joints
        }
        return robot_state

    # ---------------- Action execution ----------------

    def execute_actions(self, actions: Dict[str, np.ndarray]):
        """
        Execute predicted actions on robot:
          - Send hand commands over ROS2 (if available)
          - Compose absolute joint target for xArm and send via UDP to the XarmController child process
        """

        assert isinstance(actions, np.ndarray), f"Expected numpy array for actions, got {type(actions)}"
        assert actions.shape == (24,), f"Expected 24-dim action vector (16 hand + 8 gello), got shape {actions.shape}"
        hand_actions = actions[:16]
        gello_actions = actions[16:24]

        # Ensure types
        hand_actions = np.asarray(hand_actions, dtype=np.float32)
        gello_actions = np.asarray(gello_actions, dtype=np.float32)

        gello_deltas = gello_actions[:7] - self._last_xarm_joints
        
        # Visualization: store predicted deltas for first 7 joints
        self.predicted_coordinates = gello_deltas

        # Send hand commands (16 DOF) + wrist via ROS2 (if available)
        if self.hand_controller is not None:
            # Extract wrist command from gello_actions (if available) or use current wrist state
            wrist_cmd = None
            if len(gello_actions) >= 8:  # gello has 8 DOF, 8th might be wrist
                wrist_cmd = float(gello_actions[7])  # Use 8th gello joint as wrist command
            else:
                # Fallback: use current wrist state if available
                if self._last_xarm_joints is not None and hasattr(self, '_last_wrist_state'):
                    wrist_cmd = self._last_wrist_state
            
            self.hand_controller.send_hand_command(hand_actions, wrist_rad=wrist_cmd)

        # Send xArm joint command via UDP to child process (absolute 8-dim: 7 joints + 1 gripper placeholder)
        if self.xarm_cmd_sender is not None:
            # Need current joints to convert predicted delta -> absolute target
            if self._last_xarm_joints is None:
                # No valid current joints yet; skip this action safely
                return
            # Use the processed deltas from above
            abs_target = (self._last_xarm_joints + gello_deltas).astype(np.float32)
            # length-8 vector: joints + gripper placeholder (0.0)
            cmd = np.concatenate([abs_target, np.array([0.0], dtype=np.float32)], axis=0)
            # send over UDP
            self.xarm_cmd_sender.send([cmd.tolist()])

    # ---------------- Visualization ----------------

    def visualization_thread_func(self):
        """Background thread for real-time visualization"""
        try:
            # Initialize pygame with proper display backend
            import os
            os.environ['SDL_VIDEODRIVER'] = 'x11'  # Force X11 for Linux
            pygame.init()
            pygame.font.init()

            # Window dimensions - make bigger to accommodate larger images
            window_width = 1400
            window_height = 800

            screen = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption('Diffusion Policy Rollout - Real-time Visualization')
        except Exception as e:
            print(f"Failed to initialize pygame display: {e}")
            print("Running without visualization...")
            return
        
        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        GRAY = (128, 128, 128)

        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)


        frame_count = 0
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            screen.fill(WHITE)

            # Display camera images
            if len(self.display_queue) > 0:
                display_data = self.display_queue[-1]  # Get latest
                images = display_data.get('images', {})

                # Display cameras side by side - centered
                cam_width, cam_height = 600, 360
                total_cam_width = cam_width * 2 + 20  # Two cameras with 20px gap
                start_x = (screen.get_width() - total_cam_width) // 2  # Center horizontally

                for idx, (cam_key, img_data) in enumerate(list(images.items())[:2]):
                    if 'color' in img_data:
                        img = img_data['color']
                        # Resize and convert for pygame
                        img_resized = cv2.resize(img, (cam_width, cam_height))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        img_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))

                        # Draw camera image - centered
                        x_pos = start_x + idx * (cam_width + 20)
                        screen.blit(img_surface, (x_pos, 20))

                        # Label camera with serial - centered under image
                        serial_text = f"cam_{idx}: {str(cam_key)}"
                        label = small_font.render(serial_text, True, BLACK)
                        label_x = x_pos + (cam_width - label.get_width()) // 2
                        screen.blit(label, (label_x, 20 + cam_height + 10))

            # Display coordinate information - adjust for bigger images
            coord_panel_x = 20
            coord_panel_y = 500

            # Current robot state
            if len(self.coordinate_history) > 0:
                current_joints = self.coordinate_history[-1]

                # Draw coordinate panel
                pygame.draw.rect(screen, GRAY, (coord_panel_x, coord_panel_y, 350, 250), 2)

                title = font.render("Robot State", True, BLACK)
                screen.blit(title, (coord_panel_x + 10, coord_panel_y + 10))

                # Current joint positions
                y_offset = 50
                for i, joint_val in enumerate(current_joints[:7]):
                    joint_text = small_font.render(f"Joint {i}: {joint_val:.3f} rad", True, BLACK)
                    screen.blit(joint_text, (coord_panel_x + 10, coord_panel_y + y_offset + i * 25))

            # Predicted coordinates
            if self.predicted_coordinates is not None:
                pred_panel_x = 400
                pred_panel_y = 500

                # Draw prediction panel
                pygame.draw.rect(screen, BLUE, (pred_panel_x, pred_panel_y, 350, 250), 2)

                title = font.render("Policy Predictions", True, BLUE)
                screen.blit(title, (pred_panel_x + 10, pred_panel_y + 10))

                # Predicted joint deltas
                y_offset = 50
                for i, pred_val in enumerate(self.predicted_coordinates[:7]):
                    pred_text = small_font.render(f"Delta {i}: {pred_val:.3f}", True, BLUE)
                    screen.blit(pred_text, (pred_panel_x + 10, pred_panel_y + y_offset + i * 25))

            # Status information
            status_text = f"Step: {self.step_count}"
            status_surface = small_font.render(status_text, True, BLACK)
            screen.blit(status_surface, (20, screen.get_height() - 30))

            pygame.display.flip()

            frame_count += 1
            clock.tick(30)  # 30 FPS

        pygame.quit()

    # ---------------- Main loop ----------------

    def run_rollout(self, duration: float = 60.0):
        """Run the diffusion policy rollout"""
        # Create data directories
        data_dir = self.root / "log" / "data" / self.exp_name
        mkdir(data_dir, overwrite=True, resume=False)
        print(f"Starting diffusion policy rollout for {duration} seconds...")

        # Setup all systems
        self.setup_robot_control()
        # Load policy BEFORE cameras to align resolution and keys
        self.load_policy()
        self.setup_cameras()

        # Start robot systems (following RobotTeleopEnv pattern)
        print("Starting robot controller...")
        self.xarm_controller.start()

        # Give robot controller time to initialize
        print("Waiting for robot controller to initialize...")
        time.sleep(3)

        # Check if robot state is available
        robot_state_available = self.xarm_controller.get_current_joint() is not None
        if robot_state_available:
            print("Robot state is ready")
        else:
            print("Warning: Robot state not immediately available, will continue trying during rollout")

        print("Starting camera system...")
        try:
            self.camera_system.start()
            # restart_put after starting
            self.camera_system.restart_put(time.time() + 1)
            time.sleep(5)  # Let cameras warm up

            # Check if cameras are ready
            if not self.camera_system.is_ready:
                print("Warning: Camera system not ready, continuing with limited functionality")
            else:
                # Load camera intrinsics now that cameras are ready
                try:
                    self.camera_intrinsics = self.camera_system.get_intrinsics()
                    print("Camera intrinsics loaded successfully")
                except Exception as e:
                    print(f"Warning: Failed to load camera intrinsics: {e}")
                    self.camera_intrinsics = None
        except Exception as e:
            print(f"Warning: Camera initialization failed: {e}")
            print("Continuing without camera data...")

        # Start visualization thread (if enabled)
        self.running = True
        if not getattr(self, 'no_visualization', False):
            print("Starting visualization thread...")
            self.visualization_thread = threading.Thread(target=self.visualization_thread_func)
            self.visualization_thread.start()
        else:
            print("Visualization disabled")

        print("Starting main rollout loop...")
        start_time = time.time()

        try:
            # Reset policy queues at the beginning
            if hasattr(self.policy, 'reset'):
                self.policy.reset()

            while time.time() - start_time < duration:
                loop_start = time.time()

                # Get camera observations with timeout handling
                images = {}
                try:
                    cameras_output = self.camera_system.get(k=1)
                    if cameras_output:
                        camera_serials = list(self.camera_system.cameras.keys())
                        for idx, serial in enumerate(camera_serials):
                            if idx in cameras_output and 'color' in cameras_output[idx]:
                                color_data = cameras_output[idx]['color']
                                if color_data.ndim == 4:
                                    color_data = color_data[-1]

                                # Assert camera data dimensions
                                assert color_data.ndim == 3, f"Expected 3D image data (H,W,C), got shape {color_data.shape}"
                                assert color_data.shape[2] == 3, f"Expected 3 color channels, got {color_data.shape[2]}"

                                images[serial] = {'color': color_data}
                except TimeoutError as e:
                    # Camera timeout - continue without images this step
                    if self.step_count % 100 == 0:  # Print occasionally 
                        print(f"Camera timeout (step {self.step_count}): {e}")
                    images = {}
                except Exception as e:
                    # Other camera errors - continue without images
                    if self.step_count % 100 == 0:
                        print(f"Camera error (step {self.step_count}): {e}")
                    images = {}

                # Get robot state
                robot_state = self.get_robot_state()

                # Always add to display queue if we have images (for visualization)
                if images:
                    display_data = {
                        'images': images,
                        'robot_state': robot_state,
                        'timestamp': time.time()
                    }
                    self.display_queue.append(display_data)

                # If we have both robot state and images, build single-step batch and act
                if robot_state is not None and images:
                    # Store current robot joints for visualization & for composing absolute targets
                    current_joints = robot_state['observation.xarm_joint_pos']
                    self.coordinate_history.append(current_joints)
                    self._last_xarm_joints = current_joints  # cache for execute_actions()
                    
                    # Cache wrist state (17th DOF of hand state)
                    hand_state = robot_state['observation.hand_joint_pos']
                    if len(hand_state) >= 17:
                        self._last_wrist_state = float(hand_state[16])  # 17th element is wrist

                    # Preprocess single-frame observations mapped to policy keys
                    processed_images = self.preprocess_images_single(images)

                    # Combine robot state into single state vector (7 xarm + 17 hand = 24 total)
                    xarm_joints = robot_state['observation.xarm_joint_pos']  # 7 DOF
                    hand_joints = robot_state['observation.hand_joint_pos']  # 17 DOF
                    combined_state = np.concatenate([xarm_joints, hand_joints])  # (24,)
                    state_tensor = torch.from_numpy(combined_state).float()

                    # Cache latest single-step obs
                    self.latest_obs = {**processed_images, 'observation.state': state_tensor}

                    # Select action (policy keeps its own n_obs_steps queue)
                    batch = self.create_step_batch()
                    if batch is not None:
                        with torch.no_grad():
                            action = self.policy.select_action(batch)
                            
                            # Convert to numpy 
                            action = action.detach().cpu().numpy()
                            

                            if action.shape[0] == 1:
                                # Shape: (1, action_dim) - single action
                                current_action = action[0]
                                action_sequence = action[0:1]  # Keep as sequence of 1
                            else:
                                # Shape: (action_horizon, action_dim) - sequence without batch
                                action_sequence = action
                                current_action = action[0]

                            
                            assert current_action.ndim == 1 and current_action.shape[0] == 24, f"Expected 24-dim current action, got {current_action.shape}"
                            
                            # Visualize current frame, predicted trajectory, and immediate next frame
                            if self._last_xarm_joints is not None:
                                # Extract current action for immediate execution
                                hand_actions = current_action[:16]
                                gello_actions = current_action[16:24]
                                # Policy outputs absolute positions, so use them directly
                                predicted_joints = gello_actions[:7]
                                
                                # Start with original images
                                annotated_images = images.copy()
                                
                                # Draw current end effector frame (for debugging)
                                annotated_images = self.draw_end_effector_frame(annotated_images, self._last_xarm_joints, is_predicted=False)
                                
                                # Show trajectory visualization (blocking)
                                # Call the blocking 3D visualization directly
                                if self.debug:
                                    self.draw_trajectory_3d_window(action_sequence, self._last_xarm_joints)
                                
                                # Draw immediate next frame (first predicted step) on top of everything
                                annotated_images = self.draw_end_effector_frame(annotated_images, predicted_joints, is_predicted=True)
                                
                                # Update display queue with fully annotated images
                                if annotated_images:
                                    display_data = {
                                        'images': annotated_images,
                                        'robot_state': robot_state,
                                        'timestamp': time.time()
                                    }
                                    self.display_queue.append(display_data)
                            
                            self.execute_actions(current_action)

                elif not images and self.step_count % 100 == 0:
                    print("Warning: No camera data available, skipping action step")

                self.step_count += 1

                # Print status every 100 steps
                if self.step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    has_images = len(images) > 0
                    has_robot_state = robot_state is not None
                    print(f"Step {self.step_count}, Elapsed: {elapsed:.1f}s"
                          f"Images: {has_images}, Robot state: {has_robot_state}")

                # Maintain loop rate
                loop_time = time.time() - loop_start
                target_dt = 1.0 / 20.0  # 20 Hz
                if loop_time < target_dt:
                    time.sleep(target_dt - loop_time)

        except KeyboardInterrupt:
            print("Rollout interrupted by user")

        finally:
            self.cleanup()

    # ---------------- Cleanup ----------------

    def cleanup(self):
        """Cleanup all systems"""
        print("Cleaning up...")

        # Stop threads
        self.running = False
        if self.policy_thread is not None:
            self.policy_thread.join(timeout=5)
        if self.visualization_thread is not None:
            self.visualization_thread.join(timeout=5)

        # Stop robot systems
        if self.xarm_controller is not None:
            self.xarm_controller.stop()

        if self.camera_system is not None:
            self.camera_system.stop()

        # Stop shared memory manager
        if self.shm_manager is not None:
            self.shm_manager.shutdown()

        # Shutdown ROS2
        if ROS2_AVAILABLE and self.hand_controller is not None:
            try:
                self.hand_controller.destroy_node()
                rclpy.shutdown()
            except RuntimeError as e:
                if "Context must be initialized" in str(e):
                    print("ROS2 was not properly initialized, skipping shutdown")
                else:
                    raise

        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Rollout for Hand-Arm Robot")
    parser.add_argument("--policy_path", type=str, default=CHECK_POINT_PATH,
                       help="Path to trained diffusion policy (.safetensors file)")
    parser.add_argument("--exp_name", type=str, default="diffusion_rollout",
                       help="Experiment name for logging")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.196",
                       help="xArm robot IP address")
    parser.add_argument("--duration", type=float, default=1000,
                       help="Rollout duration in seconds")
    parser.add_argument("--camera_serials", type=str, nargs="+", default=["239222300740", "239222303153"],
                       help="Camera serial numbers, Cam0 and Cam1")
    parser.add_argument("--image_width", type=int, default=424,
                       help="Image width for policy input (default: 424)  [will be overridden by policy config]")
    parser.add_argument("--image_height", type=int, default=240,
                       help="Image height for policy input (default: 240) [will be overridden by policy config]")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for policy inference (cuda/cpu)")
    parser.add_argument("--verbose", type=bool, default= True,
                       help="Enable verbose logging")
    parser.add_argument("--debug", type=bool, default= False,
                    help="Enable debug")
    parser.add_argument("--no_visualization", action="store_true",
                       help="Disable real-time visualization window")

    args = parser.parse_args()

    # Create rollout system
    rollout = DiffusionPolicyRollout(
        policy_path=args.policy_path,
        exp_name=args.exp_name,
        robot_ip=args.robot_ip,
        camera_serial_numbers=args.camera_serials,
        device=args.device,
        verbose=args.verbose,
        debug = args.debug
    )

    # Set visualization flag
    rollout.no_visualization = args.no_visualization

    # Run rollout
    try:
        rollout.run_rollout(duration=args.duration)
    except Exception as e:
        print(f"Error during rollout: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        rollout.cleanup()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


'''
  Observations (inputs to the policy):
  1. observation.images.cam_0 - RGB camera 0 image(240x424x3)
  2. observation.images.cam_1 - RGB camera 1 image(240x424x3)
  3. observation.state - 24-dimensional vector containing:
    - XArm7 joint positions ( 7 values)                 0  1  2  3  4  5  6                            wrist
    - Hand  joint positions (17 values including wrist) 7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22    23
  Actions (outputs from the policy):                                                  
  - XArm joint actions ( 8 values including wrist)     16 17 18 19 20 21 22                               23 
  - Hand joint actions (16 values)                      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 

'''