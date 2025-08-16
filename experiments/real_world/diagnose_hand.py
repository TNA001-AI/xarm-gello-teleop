#!/usr/bin/env python3
"""
Diagnostic script to check hand ROS2 communication and identify issues
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import numpy as np
from collections import deque

class HandDiagnostic(Node):
    def __init__(self):
        super().__init__('hand_diagnostic')
        
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Subscribe to ALL joint state topics
        self.subscribers = {}
        self.data_history = {}
        self.message_counts = {}
        
        topics_to_monitor = [
            "/orca_hand/obs_joint_states",
            "/orca_hand/act_joint_states", 
            "/gello/act_joint_states",
            "/joint_states",
            "/gello/obs_joint_states",  # Check if this exists
        ]
        
        for topic in topics_to_monitor:
            self.data_history[topic] = deque(maxlen=10)
            self.message_counts[topic] = 0
            try:
                sub = self.create_subscription(
                    JointState, topic, 
                    lambda msg, t=topic: self.callback(msg, t), 
                    qos
                )
                self.subscribers[topic] = sub
                print(f"✓ Subscribed to: {topic}")
            except Exception as e:
                print(f"✗ Failed to subscribe to {topic}: {e}")
        
        # Timer for periodic status reports
        self.timer = self.create_timer(2.0, self.print_status)
        self.start_time = time.time()
        
        print("\nMonitoring hand topics for 30 seconds...")
        print("=" * 60)
        
    def callback(self, msg, topic):
        self.message_counts[topic] += 1
        
        # Store message data
        data = {
            'timestamp': time.time(),
            'joint_names': list(msg.name),
            'positions': [float(p) for p in msg.position] if msg.position else [],
            'velocities': [float(v) for v in msg.velocity] if msg.velocity else [],
            'efforts': [float(e) for e in msg.effort] if msg.effort else []
        }
        
        # Check if data is changing
        if len(self.data_history[topic]) > 0:
            last_data = self.data_history[topic][-1]
            if last_data['positions'] == data['positions']:
                data['is_stuck'] = True
            else:
                data['is_stuck'] = False
                data['changes'] = sum(1 for a, b in zip(last_data['positions'], data['positions']) 
                                    if abs(a - b) > 0.001)
        else:
            data['is_stuck'] = None
            
        self.data_history[topic].append(data)
        
    def print_status(self):
        elapsed = time.time() - self.start_time
        print(f"\n[T+{elapsed:.1f}s] Status Report:")
        print("-" * 60)
        
        for topic in self.subscribers.keys():
            count = self.message_counts[topic]
            
            if count == 0:
                print(f"✗ {topic}: NO DATA")
                continue
                
            history = self.data_history[topic]
            latest = history[-1] if history else None
            
            if not latest:
                continue
                
            # Calculate message rate
            if len(history) >= 2:
                dt = history[-1]['timestamp'] - history[0]['timestamp']
                rate = len(history) / dt if dt > 0 else 0
            else:
                rate = 0
                
            print(f"✓ {topic}:")
            print(f"  Messages: {count} ({rate:.1f} Hz)")
            print(f"  Joints: {len(latest['joint_names'])}")
            
            if latest['positions']:
                pos_range = f"[{min(latest['positions']):.3f}, {max(latest['positions']):.3f}]"
                print(f"  Position range: {pos_range}")
                
                # Check if values are stuck
                stuck_count = sum(1 for d in history if d.get('is_stuck', False))
                if stuck_count > len(history) * 0.8:  # More than 80% stuck
                    print(f"  ⚠️  WARNING: Values appear STUCK! ({stuck_count}/{len(history)} messages unchanged)")
                elif latest.get('is_stuck'):
                    print(f"  ⚠️  Last value unchanged")
                else:
                    changes = latest.get('changes', 0)
                    print(f"  ✓ Values changing ({changes} joints moved)")
                    
                # Show first few joint names and values
                for i in range(min(3, len(latest['joint_names']))):
                    name = latest['joint_names'][i]
                    val = latest['positions'][i] if i < len(latest['positions']) else 'N/A'
                    print(f"    {name}: {val:.3f}" if isinstance(val, float) else f"    {name}: {val}")
                    
        if elapsed > 30:
            print("\n" + "=" * 60)
            print("Diagnostic complete. Summary:")
            for topic, count in self.message_counts.items():
                status = "✓ ACTIVE" if count > 0 else "✗ INACTIVE"
                print(f"  {topic}: {status} ({count} messages)")
            raise KeyboardInterrupt

def main():
    print("Hand Diagnostic Tool")
    print("=" * 60)
    print("This will monitor all hand-related ROS2 topics and check for:")
    print("1. Topic availability")
    print("2. Message rates")
    print("3. Stuck/unchanging values")
    print("4. Data ranges and joint names")
    print("")
    
    rclpy.init()
    node = HandDiagnostic()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down diagnostic...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()