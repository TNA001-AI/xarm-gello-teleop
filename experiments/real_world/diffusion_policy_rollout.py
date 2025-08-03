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
    - Hand joint states (from ROS2)

Outputs:
    - xArm joint commands (7 DOF)
    - Hand joint commands (via ROS2)
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
from collections import deque

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import teleoperation modules
from modules_teleop.xarm_controller import XarmController
from camera.multi_realsense import MultiRealsense
from utils import get_root, mkdir

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
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.policies.utils import get_device_from_parameters
    LEROBOT_AVAILABLE = True
except ImportError:
    print("Warning: LeRobot not available, using mock policy")
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


class MockPolicy:
    """Mock policy for testing when LeRobot is not available"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        self.history_length = 2
        self.hand_action_dim = 16  # Hand actions
        self.gello_action_dim = 8  # Gello actions 
        self.action_queue = deque(maxlen=16)
        
    def reset(self):
        """Reset policy state"""
        self.action_queue.clear()
        
    def select_action(self, batch):
        """Select action following LeRobot policy interface"""
        batch_size = 1  # Usually 1 for rollout
        
        # If queue is empty, generate new action sequence
        if len(self.action_queue) == 0:
            # Generate 16-step action sequence (hand + gello combined)
            total_action_dim = self.hand_action_dim + self.gello_action_dim  # 24 total
            action_sequence = torch.randn(batch_size, 16, total_action_dim)
            for i in range(16):
                self.action_queue.append(action_sequence[:, i])
        
        # Return next action from queue
        if len(self.action_queue) > 0:
            return self.action_queue.popleft()
        else:
            return torch.randn(batch_size, self.hand_action_dim + self.gello_action_dim)
    
    def eval(self):
        """Set to eval mode (no-op for mock)"""
        pass


class DiffusionPolicyRollout:
    """Main class for diffusion policy rollout"""
    
    def __init__(
        self,
        policy_path: str,
        exp_name: str = "diffusion_rollout",
        robot_ip: str = "192.168.1.196",
        camera_serial_numbers: Optional[list] = None,
        image_width: int = 848,
        image_height: int = 480,
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
        
        # Data buffers
        self.observation_buffer = deque(maxlen=history_length)
        self.action_buffer = deque(maxlen=action_horizon)
        self.step_count = 0
        
        # Control flags
        self.running = False
        self.policy_thread = None
        
    def load_policy(self):
        """Load the trained diffusion policy using LeRobot factory"""
        print(f"Loading policy from: {self.policy_path}")
        
        if not LEROBOT_AVAILABLE:
            print("Using mock policy for testing")
            config = type('Config', (), {})()
            self.policy = MockPolicy(config)
            return
        
        try:
            # Use LeRobot's PreTrainedPolicy.from_pretrained for loading
            self.policy = PreTrainedPolicy.from_pretrained(str(self.policy_path.parent))
            self.policy.to(self.device)
            self.policy.eval()
            print("Policy loaded successfully using LeRobot PreTrainedPolicy")
            
        except Exception as e:
            print(f"Error loading policy with LeRobot: {e}")
            print("Using mock policy for testing")
            config = type('Config', (), {})()
            self.policy = MockPolicy(config)
    
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
        
        # Use two cameras by default if serial numbers not specified
        if self.camera_serial_numbers is None:
            self.camera_serial_numbers = ['cam_left', 'cam_right']  # Will auto-detect actual serials
        
        # Initialize camera system
        self.camera_system = MultiRealsense(
            serial_numbers=self.camera_serial_numbers[:2],  # Use only first 2 cameras
            resolution=(1280, 720),  # Capture at higher res, will resize for policy
            capture_fps=30,
            enable_depth=False,  # Diffusion policy typically uses RGB only
            enable_color=True,
            verbose=self.verbose
        )
        
    def preprocess_images(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Preprocess camera images for policy input"""
        processed = {}
        
        camera_names = ['cam_left', 'cam_right']
        camera_keys = list(images.keys())[:2]  # Use first 2 cameras
        
        for cam_key, cam_name in zip(camera_keys, camera_names):
            if cam_key in images:
                img = images[cam_key]['color']  # RGB image
                
                # Resize to policy input size
                img = cv2.resize(img, self.image_resolution)
                
                # Convert BGR to RGB (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1] and convert to tensor (C, H, W)
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                
                processed[f'observation.images.{cam_name}'] = img
        
        return processed
    
    def get_robot_state(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current robot state with separate xarm and hand components"""
        try:
            # Get xArm joint state (7 DOF)
            xarm_joints = self.xarm_controller.get_current_joint()
            if xarm_joints is None:
                return None
            
            # Get hand joint state (17 DOF for observation)
            if self.hand_controller is not None:
                hand_joints = self.hand_controller.get_hand_state()
                if hand_joints is None:
                    # Use zeros if hand state not available (17 DOF for hand obs)
                    hand_joints = np.zeros(17)
            else:
                hand_joints = np.zeros(17)
            
            # Return separate components to match LeRobot format
            robot_state = {
                'observation.xarm_joint_pos': xarm_joints,
                'observation.hand_joint_pos': hand_joints
            }
            return robot_state
            
        except Exception as e:
            print(f"Error getting robot state: {e}")
            return None
    
    def create_observation_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Create observation batch from current buffer following LeRobot format"""
        if len(self.observation_buffer) < self.history_length:
            return None
        
        batch = {}
        
        # Stack observations across history
        obs_list = list(self.observation_buffer)
        
        # Process images - stack along sequence dimension, then add batch dimension
        for cam_name in ['observation.images.cam_left', 'observation.images.cam_right']:
            if cam_name in obs_list[0]:
                # Stack sequence: (history_length, C, H, W)
                images = torch.stack([obs[cam_name] for obs in obs_list])
                # Add batch dimension: (1, history_length, C, H, W)
                batch[cam_name] = images.unsqueeze(0).to(self.device)
        
        # Process robot state components separately (match LeRobot format)
        for state_key in ['observation.xarm_joint_pos', 'observation.hand_joint_pos']:
            if state_key in obs_list[0]:
                # Stack sequence: (history_length, joint_dim) 
                states = torch.stack([obs[state_key] for obs in obs_list])
                # Add batch dimension: (1, history_length, joint_dim)
                batch[state_key] = states.unsqueeze(0).to(self.device)
        
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
                hand_actions = actions[:16]  # First 16 DOF for hand
                gello_actions = actions[16:24]  # Next 8 DOF for gello -> xarm mapping
            
            # Send hand commands (16 DOF)
            if self.hand_controller is not None:
                self.hand_controller.send_hand_command(hand_actions)
            
            # Convert gello actions to xarm joint commands (8 gello -> 7 xarm mapping)
            if self.xarm_controller is not None:
                # Use first 7 gello joints for xarm (ignore gripper for now)
                xarm_actions = gello_actions[:7]
                self.xarm_controller.move_joints(xarm_actions, wait=False)
            
        except Exception as e:
            print(f"Error executing actions: {e}")
    
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
                    
                    # Handle batch dimension - take first element if batch size > 1
                    if action.ndim == 2:  # (batch_size, action_dim)
                        action = action[0]  # Take first batch element
                    
                    # Add to action buffer
                    self.action_buffer.append(action)
                
                time.sleep(0.02)  # 50Hz inference rate
                
            except Exception as e:
                print(f"Error in policy inference: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                break
    
    def run_rollout(self, duration: float = 60.0):
        """Run the diffusion policy rollout"""
        print(f"Starting diffusion policy rollout for {duration} seconds...")
        
        # Setup all systems
        self.setup_robot_control()
        self.setup_cameras()
        self.load_policy()
        
        # Start robot systems
        print("Starting robot controller...")
        self.xarm_controller.start()
        
        print("Starting camera system...")
        self.camera_system.start()
        time.sleep(2)  # Let cameras warm up
        
        # Start policy inference thread
        self.running = True
        self.policy_thread = threading.Thread(target=self.policy_inference_thread)
        self.policy_thread.start()
        
        # Create data directories
        data_dir = self.root / "log" / "data" / self.exp_name
        mkdir(data_dir, overwrite=False, resume=False)
        
        print("Starting main rollout loop...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                loop_start = time.time()
                
                # Get camera observations
                images = {}
                if hasattr(self.camera_system, 'get_images'):
                    images = self.camera_system.get_images()
                elif hasattr(self.camera_system, 'cameras'):
                    for serial, camera in self.camera_system.cameras.items():
                        if hasattr(camera, 'get_image'):
                            images[serial] = camera.get_image()
                
                # Get robot state
                robot_state = self.get_robot_state()
                
                if images and robot_state is not None:
                    # Preprocess observations
                    processed_images = self.preprocess_images(images)
                    
                    # Create observation dictionary with separate state components
                    observation = {**processed_images}
                    
                    # Add robot state components separately
                    for state_key, state_value in robot_state.items():
                        observation[state_key] = torch.from_numpy(state_value).float()
                    
                    # Add to observation buffer
                    self.observation_buffer.append(observation)
                    
                    # Execute action if available
                    if len(self.action_buffer) > 0:
                        action = self.action_buffer.popleft()
                        self.execute_actions(action)
                
                self.step_count += 1
                
                # Print status every 100 steps
                if self.step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Step {self.step_count}, Elapsed: {elapsed:.1f}s, "
                          f"Obs buffer: {len(self.observation_buffer)}, "
                          f"Action buffer: {len(self.action_buffer)}")
                
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
        
        # Stop inference thread
        self.running = False
        if self.policy_thread is not None:
            self.policy_thread.join(timeout=5)
        
        # Stop robot systems
        if self.xarm_controller is not None:
            self.xarm_controller.stop()
        
        if self.camera_system is not None:
            self.camera_system.stop()
        
        # Shutdown ROS2
        if ROS2_AVAILABLE and self.hand_controller is not None:
            self.hand_controller.destroy_node()
            rclpy.shutdown()
        
        print("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Diffusion Policy Rollout for Hand-Arm Robot")
    parser.add_argument("--policy_path", type=str, required=True,
                       help="Path to trained diffusion policy (.safetensors file)")
    parser.add_argument("--exp_name", type=str, default="diffusion_rollout",
                       help="Experiment name for logging")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.196",
                       help="xArm robot IP address")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Rollout duration in seconds")
    parser.add_argument("--camera_serials", type=str, nargs="+", default=None,
                       help="Camera serial numbers (uses auto-detection if not specified)")
    parser.add_argument("--image_width", type=int, default=424,
                       help="Image width for policy input (default: 424)")
    parser.add_argument("--image_height", type=int, default=240,
                       help="Image height for policy input (default: 240)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for policy inference (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
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