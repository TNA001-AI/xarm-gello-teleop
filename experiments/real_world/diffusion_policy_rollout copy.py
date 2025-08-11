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
    - xArm joint commands (7 DOF)
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

# Import teleoperation modules
from modules_teleop.xarm_controller import XarmController
from camera.multi_realsense import MultiRealsense
from utils import get_root, mkdir
from multiprocess.managers import SharedMemoryManager
CHECK_POINT_PATH = "/data/xarm_orca_diffusion/checkpoints/last/pretrained_model"

# ROS2 imports for hand control
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    print("Warning: ROS2 not available, hand control disabled")
    ROS2_AVAILABLE = False

# LeRobot imports
try:
    from lerobot.policies.factory import make_policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    LEROBOT_AVAILABLE = True
except ImportError:
    print("Warning: LeRobot not available")
    LEROBOT_AVAILABLE = False


class HandController(Node):
    """ROS2 Node for controlling hand joints"""
    
    def __init__(self):
        super().__init__('hand_controller')
        
        # Publishers for hand control
        self.act_hand_pub = self.create_publisher(
            Float64MultiArray, 
            '/orca_hand/act_joint_states', 
            10
        )
        
        # Subscribers for hand observation
        self.obs_hand_sub = self.create_subscription(
            Float64MultiArray,
            '/orca_hand/obs_joint_states',
            self.obs_hand_callback,
            10
        )
        
        self.current_hand_state = None
        self.hand_state_lock = threading.Lock()
        
    def obs_hand_callback(self, msg):
        """Callback for receiving hand observations"""
        with self.hand_state_lock:
            self.current_hand_state = np.array(msg.data)
    
    def get_hand_state(self) -> Optional[np.ndarray]:
        """Get current hand joint state"""
        with self.hand_state_lock:
            if self.current_hand_state is not None:
                return self.current_hand_state.copy()
            return None
    
    def send_hand_command(self, hand_joints: np.ndarray):
        """Send hand joint commands"""
        msg = Float64MultiArray()
        msg.data = hand_joints.tolist()
        self.act_hand_pub.publish(msg)



class DiffusionPolicyRollout:
    """Main class for diffusion policy rollout"""
    
    def __init__(
        self,
        policy_path: str,
        exp_name: str = "diffusion_rollout",
        robot_ip: str = "192.168.1.196",
        camera_serial_numbers: Optional[list] = None,
        image_width: int = 424,
        image_height: int = 240,
        action_horizon: int = 16,
        history_length: int = 2,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.policy_path = Path(policy_path)
        self.exp_name = exp_name
        self.robot_ip = robot_ip
        self.camera_serial_numbers = camera_serial_numbers
        self.image_width = image_width
        self.image_height = image_height
        self.image_resolution = (image_width, image_height)
        self.action_horizon = action_horizon
        self.history_length = history_length
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        self.root = get_root(__file__)
        
        # Initialize components
        self.policy = None
        self.xarm_controller = None
        self.hand_controller = None
        self.camera_system = None
        self.shm_manager = None
        
        # Data buffers
        self.observation_buffer = deque(maxlen=history_length)
        self.action_buffer = deque(maxlen=action_horizon)
        self.step_count = 0
        
        # Control flags
        self.running = False
        self.policy_thread = None
        
        # Visualization components
        self.visualization_thread = None
        self.display_queue = deque(maxlen=10)
        self.coordinate_history = deque(maxlen=100)
        self.predicted_coordinates = None
        
    def load_policy(self):
        """Load the trained diffusion policy using LeRobot factory"""
        print(f"Loading policy from: {self.policy_path}")
        
        if not LEROBOT_AVAILABLE:
            raise RuntimeError("LeRobot is not available. Please install lerobot package.")
        
        # Check if policy path exists
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy path does not exist: {self.policy_path}")
        
        try:
            # Load policy config first (like in train.py)
            policy_config = PreTrainedConfig.from_pretrained(str(self.policy_path))
            policy_config.pretrained_path = str(self.policy_path)
            
            # Create dataset metadata the same way as train.py (lines 129-131)
            ds_meta = LeRobotDatasetMetadata(
                "tao_dataset", 
                root="/data/local_datasets"
            )
            
            # Use dataset.meta like train.py line 158 (this includes computed stats)
            self.policy = make_policy(
                cfg=policy_config,
                ds_meta=ds_meta,
            )
            self.policy.to(self.device)
            self.policy.eval()
            print("Policy loaded successfully using LeRobot make_policy with full dataset metadata including stats")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load policy: {e}. Please check policy path and files.")
    
    def setup_robot_control(self):
        """Initialize robot control systems"""
        print("Setting up robot control...")
        
        # Initialize xArm controller
        self.xarm_controller = XarmController(
            start_time=time.time(),
            ip=self.robot_ip,
            gripper_enable=False,
            mode="3D",
            command_mode="joints",  # Use joint control for diffusion policy
            robot_id=-1,
            verbose=self.verbose
        )
        
        # Initialize hand controller if ROS2 is available
        if ROS2_AVAILABLE:
            rclpy.init()
            self.hand_controller = HandController()
            print("Hand controller initialized")
        else:
            print("ROS2 not available, hand control disabled")
    
    def setup_cameras(self):
        """Initialize camera system"""
        print("Setting up cameras...")
        
        # Initialize shared memory manager (critical for camera data)
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        # Use auto-detection if serial numbers not specified
        if self.camera_serial_numbers is None:
            self.camera_serial_numbers = None  # Will auto-detect all available cameras
        
        # Initialize camera system with shared memory manager (match teleop.py settings)
        self.camera_system = MultiRealsense(
            shm_manager=self.shm_manager,  # Critical for proper operation
            serial_numbers=self.camera_serial_numbers,  # Use specified cameras or auto-detect
            resolution=(424, 240),  # Use same resolution as working teleop.py
            capture_fps=30,  # Use same FPS as working teleop.py
            put_fps=30,  # Ensure consistent put rate
            enable_depth=False,  # Diffusion policy typically uses RGB only
            enable_color=True,
            get_max_k=30,  # Allow buffering
            verbose=self.verbose
        )
        
        # Set camera parameters (matching teleoperation settings)
        print("Setting camera exposure and white balance...")
        self.camera_system.set_exposure(exposure=100, gain=60)  # 100: bright, 60: dark
        self.camera_system.set_white_balance(3800)
        
    def preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Preprocess camera images for policy input"""
        processed = {}
        
        # Use actual camera keys and map to cam_0, cam_1 for consistency
        camera_keys = list(images.keys())[:2]  # Use first 2 cameras
        
        for idx, cam_key in enumerate(camera_keys):
            if cam_key in images:
                img = images[cam_key]['color']  # RGB image
                
                # Assert input image dimensions (H, W, C) format
                assert img.ndim == 3, f"Expected 3D image (H,W,C), got shape {img.shape}"
                assert img.shape[2] == 3, f"Expected 3 color channels, got {img.shape[2]}"
                
                # Resize to policy input size
                img = cv2.resize(img, self.image_resolution)
                
                # Convert BGR to RGB (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1] and convert to tensor (C, H, W)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                
                # Assert output tensor dimensions (C, H, W)
                assert img.shape == (3, self.image_height, self.image_width), f"Expected tensor shape (3, {self.image_height}, {self.image_width}), got {img.shape}"
                
                processed[f'observation.images.cam_{idx}'] = img
        
        # Assert we have exactly 2 cameras for policy input
        assert len(processed) == 2, f"Expected 2 cameras for policy input, got {len(processed)}"
        
        return processed
    
    def get_robot_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current robot state with separate xarm and hand components"""
        try:
            # Get xArm joint state (7 DOF) - use queue directly like RobotTeleopEnv
            xarm_joints = None
            if self.xarm_controller.is_controller_alive and not self.xarm_controller.cur_qpos_q.empty():
                xarm_joints = self.xarm_controller.cur_qpos_q.get()
            
            if xarm_joints is None:
                if self.step_count % 100 == 0:  # Only print occasionally
                    print("Debug: xarm joint queue is empty or controller not alive")
                return None
            
            # Assert xArm joint dimensions
            assert isinstance(xarm_joints, np.ndarray), f"Expected numpy array for xarm joints, got {type(xarm_joints)}"
            assert xarm_joints.shape == (7,), f"Expected 7 DOF for xArm, got shape {xarm_joints.shape}"
            
            # Get hand joint state (17 DOF for observation)
            if self.hand_controller is not None:
                hand_joints = self.hand_controller.get_hand_state()
                if hand_joints is None:
                    # Use zeros if hand state not available (17 DOF for hand obs)
                    hand_joints = np.zeros(17)
            else:
                hand_joints = np.zeros(17)
            
            # Assert hand joint dimensions
            assert isinstance(hand_joints, np.ndarray), f"Expected numpy array for hand joints, got {type(hand_joints)}"
            assert hand_joints.shape == (17,), f"Expected 17 DOF for hand, got shape {hand_joints.shape}"
            
            # Return separate components to match LeRobot format
            robot_state = {
                'observation.xarm_joint_pos': xarm_joints,
                'observation.hand_joint_pos': hand_joints
            }
            return robot_state
            
        except Exception as e:
            if self.step_count % 100 == 0:  # Only print occasionally
                print(f"Error getting robot state: {e}")
            return None
    
    def create_observation_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Create observation batch from current buffer following LeRobot format"""
        if len(self.observation_buffer) < self.history_length:
            return None
        
        # Assert observation buffer has correct length
        assert len(self.observation_buffer) == self.history_length, f"Expected buffer length {self.history_length}, got {len(self.observation_buffer)}"
        
        batch = {}
        
        # Stack observations across history
        obs_list = list(self.observation_buffer)
        
        # Process images - stack along sequence dimension, then add batch dimension
        for cam_name in ['observation.images.cam_0', 'observation.images.cam_1']:
            if cam_name in obs_list[0]:
                # Assert all observations have the same image tensor shape
                first_shape = obs_list[0][cam_name].shape
                for i, obs in enumerate(obs_list):
                    assert obs[cam_name].shape == first_shape, f"Image shape mismatch at step {i}: expected {first_shape}, got {obs[cam_name].shape}"
                
                # Stack sequence: (history_length, C, H, W)
                images = torch.stack([obs[cam_name] for obs in obs_list])
                # Add batch dimension: (1, history_length, C, H, W)
                batch[cam_name] = images.unsqueeze(0).to(self.device)
                
                # Assert final batch dimensions
                expected_shape = (1, self.history_length, 3, self.image_height, self.image_width)
                assert batch[cam_name].shape == expected_shape, f"Expected batch shape {expected_shape}, got {batch[cam_name].shape}"
        
        # Process combined robot state (matches policy input format)
        if 'observation.state' in obs_list[0]:
            # Assert all observations have the same state tensor shape
            first_shape = obs_list[0]['observation.state'].shape
            for i, obs in enumerate(obs_list):
                assert obs['observation.state'].shape == first_shape, f"State shape mismatch at step {i}: expected {first_shape}, got {obs['observation.state'].shape}"
            
            # Stack sequence: (history_length, state_dim) 
            states = torch.stack([obs['observation.state'] for obs in obs_list])
            # Add batch dimension: (1, history_length, state_dim)
            batch['observation.state'] = states.unsqueeze(0).to(self.device)
            
            # Assert final batch dimensions (24 = 7 xarm + 17 hand)
            expected_shape = (1, self.history_length, 24)
            assert batch['observation.state'].shape == expected_shape, f"Expected batch shape {expected_shape}, got {batch['observation.state'].shape}"
        
        # Assert batch contains all required keys
        required_keys = ['observation.images.cam_0', 'observation.images.cam_1', 'observation.state']
        for key in required_keys:
            assert key in batch, f"Missing required key '{key}' in observation batch"
        
        return batch
    
    def execute_actions(self, actions: Dict[str, np.ndarray]):
        """Execute predicted actions on robot (separate hand and gello actions)"""
        try:
            # Handle different action formats based on policy output
            if isinstance(actions, dict):
                # If actions are already separated by policy
                hand_actions = actions.get('action.hand_joint_pos', np.zeros(16))
                gello_actions = actions.get('action.gello_joint_pos', np.zeros(8))
            else:
                # If actions are concatenated (16 hand + 8 gello = 24 total)
                assert isinstance(actions, np.ndarray), f"Expected numpy array for actions, got {type(actions)}"
                assert actions.shape == (24,), f"Expected 24-dim action vector (16 hand + 8 gello), got shape {actions.shape}"
                
                hand_actions = actions[:16]  # First 16 DOF for hand
                gello_actions = actions[16:24]  # Next 8 DOF for gello -> xarm mapping
            
            # Assert action dimensions
            assert isinstance(hand_actions, np.ndarray), f"Expected numpy array for hand actions, got {type(hand_actions)}"
            assert hand_actions.shape == (16,), f"Expected 16 DOF for hand actions, got shape {hand_actions.shape}"
            
            assert isinstance(gello_actions, np.ndarray), f"Expected numpy array for gello actions, got {type(gello_actions)}"
            assert gello_actions.shape == (8,), f"Expected 8 DOF for gello actions, got shape {gello_actions.shape}"
            
            # Store predicted coordinates for visualization
            self.predicted_coordinates = gello_actions[:7]  # Joint space predictions
            
            # Send hand commands (16 DOF)
            if self.hand_controller is not None:
                self.hand_controller.send_hand_command(hand_actions)
            
            # Convert gello actions to xarm joint commands (8 gello -> 7 xarm mapping)
            if self.xarm_controller is not None:
                # Use first 7 gello joints for xarm (ignore gripper for now)
                xarm_actions = gello_actions[:7]
                assert xarm_actions.shape == (7,), f"Expected 7 DOF for xarm actions, got shape {xarm_actions.shape}"
                self.xarm_controller.move_joints(xarm_actions, wait=False)
            
        except Exception as e:
            print(f"Error executing actions: {e}")
    
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
        
        print("Visualization thread started...")
        
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
                start_x = (window_width - total_cam_width) // 2  # Center horizontally
                
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
                        
                        # Label camera - centered under image
                        label = small_font.render(f"Camera {idx}", True, BLACK)
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
            status_text = f"Step: {self.step_count} | Obs Buffer: {len(self.observation_buffer)} | Action Buffer: {len(self.action_buffer)}"
            status_surface = small_font.render(status_text, True, BLACK)
            screen.blit(status_surface, (20, window_height - 30))
            
            pygame.display.flip()
            
            frame_count += 1
            clock.tick(30)  # 30 FPS
        
        pygame.quit()
        print("Visualization thread stopped")
    
    def policy_inference_thread(self):
        """Background thread for policy inference following LeRobot pattern"""
        print("Starting policy inference thread...")
        
        while self.running:
            try:
                # Create observation batch
                batch = self.create_observation_batch()
                if batch is None:
                    time.sleep(0.01)
                    continue
                
                # Run policy inference using LeRobot's select_action interface
                with torch.no_grad():
                    # Reset policy if this is the first step
                    if self.step_count == 0 and hasattr(self.policy, 'reset'):
                        self.policy.reset()
                    
                    # Use select_action method (standard LeRobot interface)
                    action = self.policy.select_action(batch)
                    
                    # Convert to numpy
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    # Assert action tensor properties
                    assert isinstance(action, np.ndarray), f"Expected numpy array after conversion, got {type(action)}"
                    
                    # Handle batch dimension - take first element if batch size > 1
                    if action.ndim == 2:  # (batch_size, action_dim)
                        assert action.shape[0] == 1, f"Expected batch size 1, got {action.shape[0]}"
                        action = action[0]  # Take first batch element
                    
                    # Assert final action dimensions (24 = 16 hand + 8 gello)
                    assert action.ndim == 1, f"Expected 1D action vector, got shape {action.shape}"
                    assert action.shape == (24,), f"Expected 24-dim action vector, got shape {action.shape}"
                    
                    # Add to action buffer
                    self.action_buffer.append(action)
                
                time.sleep(0.02)  # 50Hz inference rate
                
            except Exception as e:
                print(f"Error in policy inference: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                break
    
    def run_rollout(self, duration: float = 60.0):
        """Run the diffusion policy rollout"""
        # Create data directories
        data_dir = self.root / "log" / "data" / self.exp_name
        mkdir(data_dir, overwrite=False, resume=False)
        print(f"Starting diffusion policy rollout for {duration} seconds...")
        
        # Setup all systems
        self.setup_robot_control()
        self.setup_cameras()
        self.load_policy()
        
        # Start robot systems (following RobotTeleopEnv pattern)
        print("Starting robot controller...")
        self.xarm_controller.start()
        
        # Give robot controller time to initialize (like RobotTeleopEnv does)
        print("Waiting for robot controller to initialize...")
        time.sleep(3)  # Simple wait like in RobotTeleopEnv
        
        # Check if robot state is available
        robot_state_available = self.xarm_controller.get_current_joint() is not None
        if robot_state_available:
            print("Robot state is ready")
        else:
            print("Warning: Robot state not immediately available, will continue trying during rollout")
        
        print("Starting camera system...")
        try:
            self.camera_system.start()
            # Critical: restart_put after starting (matches teleoperation pattern)
            self.camera_system.restart_put(time.time() + 1)
            time.sleep(3)  # Let cameras warm up
            
            # Check if cameras are ready
            if not self.camera_system.is_ready:
                print("Warning: Camera system not ready, continuing with limited functionality")
        except Exception as e:
            print(f"Warning: Camera initialization failed: {e}")
            print("Continuing without camera data...")
        
        # Start policy inference thread
        self.running = True
        self.policy_thread = threading.Thread(target=self.policy_inference_thread)
        self.policy_thread.start()
        
        # Start visualization thread (if enabled)
        if not getattr(self, 'no_visualization', False):
            print("Starting visualization thread...")
            self.visualization_thread = threading.Thread(target=self.visualization_thread_func)
            self.visualization_thread.start()
        else:
            print("Visualization disabled")
        

        
        print("Starting main rollout loop...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                loop_start = time.time()
                
                # Get camera observations
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
                except Exception as e:
                    if self.step_count % 100 == 0:  # Only print occasionally to avoid spam
                        print(f"Warning: Failed to get camera data: {e}")
                    images = {}  # Continue without camera data
                
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
                
                # Only add to observation buffer if we have both robot state and images
                if robot_state is not None and images:
                    # Store current robot joints for visualization
                    current_joints = robot_state['observation.xarm_joint_pos']
                    self.coordinate_history.append(current_joints)
                    
                    # Preprocess observations
                    processed_images = self.preprocess_images(images)
                    
                    # Create observation dictionary matching policy input format
                    observation = {**processed_images}
                    
                    # Combine robot state into single state vector (7 xarm + 17 hand = 24 total)
                    xarm_joints = robot_state['observation.xarm_joint_pos']  # 7 DOF
                    hand_joints = robot_state['observation.hand_joint_pos']  # 17 DOF
                    combined_state = np.concatenate([xarm_joints, hand_joints])  # 24 total
                    
                    observation['observation.state'] = torch.from_numpy(combined_state).float()
                    
                    # Add to observation buffer
                    self.observation_buffer.append(observation)
                    
                    # Execute action if available
                    if len(self.action_buffer) > 0:
                        action = self.action_buffer.popleft()
                        self.execute_actions(action)
                elif not images and self.step_count % 100 == 0:
                    print("Warning: No camera data available, skipping observation collection")
                
                self.step_count += 1
                
                # Print status every 100 steps
                if self.step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    has_images = len(images) > 0
                    has_robot_state = robot_state is not None
                    print(f"Step {self.step_count}, Elapsed: {elapsed:.1f}s, "
                          f"Obs buffer: {len(self.observation_buffer)}, "
                          f"Action buffer: {len(self.action_buffer)}, "
                          f"Images: {has_images}, Robot state: {has_robot_state}")
                
                # Maintain loop rate
                loop_time = time.time() - loop_start
                target_dt = 1.0 / 30.0  # 30 Hz
                if loop_time < target_dt:
                    time.sleep(target_dt - loop_time)
        
        except KeyboardInterrupt:
            print("Rollout interrupted by user")
        
        finally:
            self.cleanup()
    
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
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Rollout duration in seconds")
    parser.add_argument("--camera_serials", type=str, nargs="+", default=["239222300740", "239222303153"],
                       help="Camera serial numbers (uses auto-detection if not specified)")
    parser.add_argument("--image_width", type=int, default=424,
                       help="Image width for policy input (default: 424)")
    parser.add_argument("--image_height", type=int, default=240,
                       help="Image height for policy input (default: 240)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for policy inference (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--no_visualization", action="store_true",
                       help="Disable real-time visualization window")
    
    args = parser.parse_args()
    
    # Create rollout system
    rollout = DiffusionPolicyRollout(
        policy_path=args.policy_path,
        exp_name=args.exp_name,
        robot_ip=args.robot_ip,
        camera_serial_numbers=args.camera_serials,
        image_width=args.image_width,
        image_height=args.image_height,
        device=args.device,
        verbose=args.verbose
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