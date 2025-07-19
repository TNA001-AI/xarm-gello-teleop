from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import get_root, mkdir
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))

from modules_teleop.robot_env_teleop import RobotTeleopEnv


if __name__ == '__main__':
    # cv2 encounter error when using multi-threading, use tk instead
    # cv2.setNumThreads(cv2.getNumberOfCPUs())
    # cv2.namedWindow("real env monitor", cv2.WINDOW_NORMAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    env = RobotTeleopEnv(
        mode='3D',
        exp_name=args.name,
        resolution=(848, 480),
        capture_fps=30,
        record_fps=30,
        perception_process_func=None,
        use_robot=True,
        use_gello=True,
        bimanual=args.bimanual,
        gripper_enable=False,
        data_dir="data",
        debug=True,
    )

    env.start()
    env.join()
