#!/usr/bin/env python3
"""
ROS2 Bag to Text Converter

A standalone utility to convert ROS2 bag data into text files.
Based on the ROS2 bag reading functionality from teleop_postprocess_with_hand.py.

Usage:
    python ros2bag_to_text.py --bag_path /path/to/bag --topic /topic_name --output_dir /output/path
    
Example:
    python ros2bag_to_text.py --bag_path ros2_bag/rosbag2_2025_01_23_12_34_56 --topic /orca_hand/joint_angles --output_dir hand_data/
"""

from pathlib import Path
import argparse
import os
import numpy as np

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
    print("Warning: ROS2 dependencies not available.")
    ROS2_AVAILABLE = False


def read_ros2_bag(bag_path: str, topic_name: str = None):
    """
    Read messages from a ROS2 bag file for a specific topic or all topics
    Returns list of (topic, timestamp, message) tuples
    """
    if not ROS2_AVAILABLE:
        print("ROS2 not available, cannot read bag files")
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
    
    print(f"Available topics in bag:")
    for topic_info in topic_types:
        print(f"  {topic_info.name} ({topic_info.type})")
    
    messages = []
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        if topic_name is None or topic == topic_name:
            try:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                messages.append((topic, timestamp / 1e9, msg))  # Convert nanoseconds to seconds
            except Exception as e:
                print(f"Warning: Failed to deserialize message for topic {topic}: {e}")
                continue
    
    return messages


def extract_message_data(msg):
    """
    Extract numerical data from different message types
    Returns numpy array or None if extraction fails
    """
    import numpy as np
    # Handle JointState messages (common for robot joints)
    if hasattr(msg, 'position') and msg.position:
        return np.array(msg.position)
    
    # Handle Float64MultiArray messages
    if hasattr(msg, 'data') and msg.data:
        return np.array(msg.data)
    
    # Handle geometry_msgs/Twist
    if hasattr(msg, 'linear') and hasattr(msg, 'angular'):
        linear = [msg.linear.x, msg.linear.y, msg.linear.z]
        angular = [msg.angular.x, msg.angular.y, msg.angular.z]
        return np.array(linear + angular)
    
    # Handle geometry_msgs/Pose
    if hasattr(msg, 'position') and hasattr(msg, 'orientation'):
        position = [msg.position.x, msg.position.y, msg.position.z]
        orientation = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        return np.array(position + orientation)
    
    # Handle geometry_msgs/Point
    if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'z'):
        return np.array([msg.x, msg.y, msg.z])
    
    # Try to extract any array-like data
    for attr in ['data', 'values', 'points']:
        if hasattr(msg, attr):
            data = getattr(msg, attr)
            if data and hasattr(data, '__iter__'):
                try:
                    return np.array(data)
                except:
                    continue
    
    return None


def save_messages_to_text(messages, output_dir: Path, topic_name: str):
    """
    Save ROS2 messages to text files using timestamp as filename (same format as joints)
    """
    import numpy as np
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata file
    metadata_file = output_dir / "metadata.txt"
    
    successful_extractions = 0
    
    # Write metadata
    with open(metadata_file, 'w') as f:
        f.write(f"Topic: {topic_name}\n")
        f.write(f"Total messages: {len(messages)}\n")
        if messages:
            f.write(f"Message type: {type(messages[0][2]).__name__}\n")
            f.write(f"Time range: {messages[0][1]:.6f} - {messages[-1][1]:.6f} seconds\n")
        
        # Write sample message attributes
        if messages:
            sample_msg = messages[0][2]
            f.write(f"Message attributes: {[attr for attr in dir(sample_msg) if not attr.startswith('_')]}\n")
    
    # Process each message and save with timestamp filename (same format as joints)
    for i, (topic, timestamp, msg) in enumerate(messages):
        # Extract numerical data
        data = extract_message_data(msg)
        
        if data is not None:
            # Save individual file with timestamp as filename (same format as joints)
            # Use 3 decimal precision for timestamp filename to match joint format
            timestamp_file = output_dir / f"{timestamp:.3f}.txt"
            np.savetxt(timestamp_file, data, fmt='%.6f')
            
            successful_extractions += 1
        else:
            print(f"Warning: Could not extract data from message at timestamp {timestamp}")
            print(f"Message type: {type(msg)}")
            print(f"Message attributes: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
    
    print(f"Successfully processed {successful_extractions}/{len(messages)} messages")
    print(f"Output saved to: {output_dir}")
    print(f"Files named with timestamps (e.g., {messages[0][1]:.3f}.txt)")
    
    return successful_extractions


def convert_bag_to_text(bag_path: str, topic_name: str = None, output_dir: str = "bag_output"):
    """
    Main conversion function
    """
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    
    if not bag_path.exists():
        print(f"Error: Bag path {bag_path} does not exist")
        return False
    
    print(f"Reading ROS2 bag from: {bag_path}")
    messages = read_ros2_bag(str(bag_path), topic_name)
    
    if not messages:
        print("No messages found in bag file")
        return False
    
    # Group messages by topic
    topics = {}
    for topic, timestamp, msg in messages:
        if topic not in topics:
            topics[topic] = []
        topics[topic].append((topic, timestamp, msg))
    
    # Save each topic separately
    for topic, topic_messages in topics.items():
        topic_output_dir = output_dir
        print(f"\nProcessing topic: {topic} ({len(topic_messages)} messages)")
        save_messages_to_text(topic_messages, topic_output_dir, topic)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bag data to text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert specific topic to text
   python experiments/real_world/ros2bag_to_text.py --bag_path /home/yolandazhu/xarm-gello-teleop/experiments/log/data/test/r0/ros2_bag/hand_joint_angles_1755105918
  
  # Convert all topics in bag
  python ros2bag_to_text.py --bag_path ros2_bag/rosbag2_2025_01_23_12_34_56
  
  # Specify output directory
  python ros2bag_to_text.py --bag_path ros2_bag/rosbag2_2025_01_23_12_34_56 --output_dir my_output/
        """
    )
    
    parser.add_argument('--bag_path', type=str, required=True,
                       help='Path to the ROS2 bag directory')
    parser.add_argument('--topic', type=str, default=None,
                       help='Specific topic to extract (if not specified, extracts all topics)')
    parser.add_argument('--output_dir', type=str, default='bag_output',
                       help='Output directory for text files (default: bag_output)')
    parser.add_argument('--list_topics', action='store_true',
                       help='List available topics in the bag and exit')
    
    args = parser.parse_args()
    
    if not ROS2_AVAILABLE:
        print("Error: ROS2 dependencies not available")
        return 1
    
    # List topics only
    if args.list_topics:
        print(f"Listing topics in: {args.bag_path}")
        messages = read_ros2_bag(args.bag_path)
        return 0
    
    # Convert bag to text
    success = convert_bag_to_text(args.bag_path, args.topic, args.output_dir)
    
    if success:
        print("\nConversion completed successfully!")
        return 0
    else:
        print("\nConversion failed!")
        return 1


if __name__ == '__main__':
    exit(main())