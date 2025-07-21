from typing import Union
from pathlib import Path
import argparse
import os
import subprocess
import numpy as np
import glob
import cv2
import torch
import shutil
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import supervision as sv
import open3d as o3d
import time
import kornia
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import get_root, mkdir
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))
sys.path.append(str(root / "../third-party/sam2"))


def match_timestamps(name: str, recording_dirs: dict, num_cams: int = 4):
    # post process the recordin,,,.,.pg

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

            # load the recording
            recording_dir = root / "log" / "data" / recording_name
            calibration_dir = recording_dir / "calibration"
            subprocess.run(f'cp -r {calibration_dir} {episode_save_dir}', shell=True)
            robot_dir = recording_dir / "robot"
            robot_timesteps = sorted([float(d[:-4]) for d in os.listdir(robot_dir)])
            action_dir = recording_dir / str(action_name)
            with open(action_dir / 'timestamps.txt', 'r') as f:
                action_timesteps = f.readlines()
            action_timesteps = [[float(tt) for tt in t.split()[-num_cams:]] for t in action_timesteps]

            # match timesteps
            # iterate over action timesteps
            # for each action timestep, find the closest robot timestep
            for t, action_timestep in enumerate(action_timesteps):
                master_timestep = action_timestep[0]
                timesteps = [master_timestep]
                for cam in range(1, num_cams):
                    min_dist = 999
                    min_dist_tt = -100
                    for tt in range(max(t-1, 0), min(t+1, len(action_timesteps))):
                        t_diff = abs(action_timesteps[tt][cam] - master_timestep)
                        if t_diff < min_dist:
                            min_dist = t_diff
                            min_dist_tt = tt
                    assert min_dist_tt != -100
                    timesteps.append(action_timesteps[min_dist_tt][cam])
                
                # find corresponding robot data
                min_dist = 999
                min_dist_tt = 100
                for tt in range(len(robot_timesteps)):
                    t_diff = abs(master_timestep - robot_timesteps[tt])
                    if t_diff < min_dist:
                        min_dist = t_diff
                        min_dist_tt = tt
                assert min_dist_tt != -100
                assert min_dist_tt > 0
                if min_dist_tt >= len(robot_timesteps) - 1:
                    print('[Warning] robot recording ends before camera recording')
                    break
                
                # saves things
                # save the matched timesteps
                with open(episode_save_dir / "timestamps.txt", 'a') as f:
                    f.write(' '.join([str(tt) for tt in timesteps]) + '\n')
                
                # save the matched timesteps
                for cam in range(num_cams):
                    source_dir = action_dir / f"camera_{cam}" / "rgb" / f"{t:06d}.jpg"
                    target_dir = episode_save_dir_cam_list[cam] / "rgb" / f"{t:06d}.jpg"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)

                    source_dir = action_dir / f"camera_{cam}" / "depth" / f"{t:06d}.png"
                    target_dir = episode_save_dir_cam_list[cam] / "depth" / f"{t:06d}.png"
                    subprocess.run(f'cp {source_dir} {target_dir}', shell=True)
                
                # interpolate robot motion using the closest robot timesteps
                if master_timestep > robot_timesteps[min_dist_tt]:
                    tt1 = min_dist_tt
                    tt2 = min_dist_tt + 1
                else:
                    tt1 = min_dist_tt - 1
                    tt2 = min_dist_tt
                weight = (master_timestep - robot_timesteps[tt1]) / (robot_timesteps[tt2] - robot_timesteps[tt1] + 1e-6)
                robot_data1 = np.loadtxt(robot_dir / f"{robot_timesteps[tt1]:.3f}.txt")
                robot_data2 = np.loadtxt(robot_dir / f"{robot_timesteps[tt2]:.3f}.txt")

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

                assert robot_data1.shape[0] in [1, 5, 9]  # bi-manual (2 * (1 pos + 3 rot) + 1 gripper) or single arm (1 pos + 3 rot + 1 gripper or 1 pos)
                if robot_data1.shape[0] > 1:  # 5 or 9
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

                # print('adding end effector offset [0, 0, 0.175]')
                # eef_t = np.array([0, 0, 0.175])
                # robot_data = robot_data + eef_t
                np.savetxt(episode_save_dir_robot / f"{t:06d}.txt", robot_data)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=1)
    parser.add_argument('--num_cams', type=int, default=2)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    args = parser.parse_args()


    # for ii in [
    #     # 0, 1, 2, 3,  # rope
    #     # 4,  # plush
    #     # 5,  # bag,
    #     # 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  # cloth
    #     # 17,  # rigid
    #     # 18,  # granular
    #     # 19,  # bread
    #     # 20,  # paper
    #     # 21, 22, 23,  # box
    #     # 24, 25, 26, 27, 28, 29,  # multi-cloth
    #     # 30,  # bag,
    #     # 31, 32, # multi-cloth 2
    #     # 33, 34, 35, 36, 37,  # multi-rope
    #     38, 39, 40, 41, 42, 43, 44, 
    #     # 45,  # rope-occ
    # ]:
    if True:
        ii = int(args.data_id)

        eef_T = [0, 0, -0.01]
        cameras = None

        if ii == 99:
            args.name = 'teleop_test_rl_processed'
            args.bimanual = True
            args.num_cams = 2
            dirs = {
                # 'teleop_test_rl/traj_1-1': ['1738796329'],
                # 'teleop_test_rl/traj_1-2': ['1738796997'],
                # 'teleop_test_rl/traj_1-3': ['1738797083'],
                # 'teleop_test_rl/traj_2-1': ['1738797229'],
                # 'teleop_test_rl/traj_2-2': ['1738797295'],
                # 'teleop_test_rl/traj_3-1': ['1738797646'],
                # 'teleop_test_rl/traj_3-3': ['1738797702'],
                # 'teleop_test_rl/traj_4-1': ['1738797380'],
                # 'teleop_test_rl/traj_4-2': ['1738797492'],
                # 'teleop_test_rl/traj_4-3': ['1738797559'],
                # 'teleop_test_rl/test_traj': ['1738809978'],
                'teleop_test_rl/rel_traj_mode1': ['1739216010'],
                'teleop_test_rl/rel_traj_mode2': ['1739216080'],
                'teleop_test_rl/rel_traj_mode3': ['1739216498'],
            }

        if ii == 0:
            args.name = 'test_processed'
            args.bimanual = False
            args.num_cams = 4
            dirs = {
                'test/recording_1': ['1740075589', '1740075610', '1740075626', '1740075645', '1740075660', '1740075681'],
                'test/recording_3': ['1740075844', '1740075853', '1740075863', '1740075873', '1740075883', '1740075898', '1740075910', '1740075923', '1740075972']
            }
        
        if ii == 1:
            args.name = 'test_run_processed'
            args.bimanual = False
            args.num_cams = 2
            dirs = {
                'tao/recording_1': ['1753122120'],
            }

        # try:
        if True:
            match_timestamps(args.name, dirs, num_cams=args.num_cams)

        # except Exception as e:
        #     with open(f'{args.name}_failed.txt', 'a') as f:
        #         f.write(f"{ii}: {e}\n")
        #     print(f"Error in {args.name}: {e}")
        #     continue

