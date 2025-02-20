from typing import Optional
from pathlib import Path
import sys
import os

import cv2
import json
import time
import pickle
import numpy as np
import torch
import math
import copy
from multiprocess.managers import SharedMemoryManager

from .xarm_wrapper import XARM7
from camera.multi_realsense import MultiRealsense, SingleRealsense
from utils.pcd_utils import depth2fgpcd, rpy_to_rotation_matrix

from nclaw.utils import get_root
root = get_root(__file__)


def get_bounding_box():
    return np.array([[0.0, 0.6], [-0.35, 0.45 + 0.75], [-0.65, 0.05]])  # the world frame robot workspace


class BimanualRobotEnv:
    def __init__(self, 
            WH=[640, 480],
            capture_fps=15,
            obs_fps=15,
            n_obs_steps=2,
            enable_color=True,
            enable_depth=True,
            process_depth=False,
            verbose=False,
            gripper_enable=False,
            speed=50,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.WH = WH
        self.capture_fps = capture_fps
        self.obs_fps = obs_fps
        self.n_obs_steps = n_obs_steps

        self.calibrate_result_dir = root / 'log' / 'latest_calibration'
        os.makedirs(self.calibrate_result_dir, exist_ok=True)
        self.vis_dir = f'{self.calibrate_result_dir}/vis'
        os.makedirs(self.vis_dir, exist_ok=True)

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.n_fixed_cameras = len(self.serial_numbers)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        # assert self.n_fixed_cameras == 4, "Currently only support 4 fixed cameras"
        self.serial_numbers_dict = {
            '235422302222': 'right',
            '239222303404': 'left',
            '239222303153': 'left-mid',
            '239222300740': 'right-mid',
        }

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.realsense =  MultiRealsense(
                serial_numbers=self.serial_numbers,
                shm_manager=self.shm_manager,
                resolution=(self.WH[0], self.WH[1]),
                capture_fps=self.capture_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                verbose=verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance(3800)
        self.last_realsense_data = None
        self.enable_color = enable_color
        self.enable_depth = enable_depth

        self.calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.calibration_board = cv2.aruco.CharucoBoard(
            size=(6, 5),
            squareLength=0.04,
            markerLength=0.03,
            dictionary=self.calibration_dictionary,
        )
        self.calibration_parameters = cv2.aruco.DetectorParameters()
        calibration_parameters =  cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.calibration_board,
            calibration_parameters,
        )
        self.R_cam2world = None
        self.t_cam2world = None
        self.R_leftbase2world = None
        self.t_leftbase2world = None
        self.R_rightbase2world = None
        self.t_rightbase2world = None

        self.robot_left = XARM7(interface='192.168.1.196', gripper_enable=gripper_enable, speed=speed)
        self.robot_right = XARM7(interface='192.168.1.224', gripper_enable=gripper_enable, speed=speed)
        self.gripper_enable = gripper_enable

        self.bbox = get_bounding_box()
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame
        self.world_y = 0.01  # the world y coordinate of the eef during action
        self.state = None

        self.R_base2board = np.array([
            [1.0, 0, 0],
            [0, -1.0, 0],
            [0, 0, -1.0]
        ])
        self.t_base2board = np.array(
            [-0.095, 0.085, -0.01]
        )
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot_left.is_alive and self.robot_right.is_alive
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        left_robot_obs = dict()
        left_robot_obs['joint_angles'] = self.robot_left.get_current_joint()
        left_robot_obs['pose'] = self.robot_left.get_current_pose()
        if self.gripper_enable:
            left_robot_obs['gripper_position'] = self.robot_left.get_gripper_state()
        
        right_robot_obs = dict()
        right_robot_obs['joint_angles'] = self.robot_right.get_current_joint()
        right_robot_obs['pose'] = self.robot_right.get_current_pose()
        if self.gripper_enable:
            right_robot_obs['gripper_position'] = self.robot_right.get_gripper_state()

        # align camera obs timestamps
        dt = 1 / self.obs_fps
        timestamp_list = [x['timestamp'][-1] for x in self.last_realsense_data.values()]
        last_timestamp = np.max(timestamp_list)
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        # the last timestamp is the latest one

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            if get_color:
                assert self.enable_color
                camera_obs[f'color_{camera_idx}'] = value['color'][this_idxs]  # BGR
            if get_depth and isinstance(camera_idx, int):
                assert self.enable_depth
                camera_obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(left_robot_obs)
        obs_data.update(right_robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return (
            [self.R_cam2world[i].copy() for i in self.serial_numbers[:4]],
            [self.t_cam2world[i].copy() for i in self.serial_numbers[:4]],
        )

    def get_bbox(self):
        return self.bbox.copy()

    def reset_robot(self, wait=True):
        self.robot_left.reset(wait=wait)
        self.robot_right.reset(wait=wait)
    
    def get_robot_pose(self, raw=False):
        raw_pose_left = self.robot_left.get_current_pose()
        raw_pose_right = self.robot_right.get_current_pose()
        if raw:
            return raw_pose_left, raw_pose_right
        else:
            R_leftgripper2base = rpy_to_rotation_matrix(
                raw_pose_left[3], raw_pose_left[4], raw_pose_left[5]
            )
            t_leftgripper2base = np.array(raw_pose_left[:3]) / 1000
            R_rightgripper2base = rpy_to_rotation_matrix(
                raw_pose_right[3], raw_pose_right[4], raw_pose_right[5]
            )
            t_rightgripper2base = np.array(raw_pose_right[:3]) / 1000
        return R_leftgripper2base, t_leftgripper2base, R_rightgripper2base, t_rightgripper2base

    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        rvecs = {}
        tvecs = {}
        rvecs_list = []
        tvecs_list = []

        # Calculate the markers
        obs = self.get_obs(get_depth=visualize)
        intrs = self.get_intrinsics()
        dist_coef = np.zeros(5)

        for i in range(self.n_fixed_cameras):
            device = self.serial_numbers[i]
            device_name = self.serial_numbers_dict[device]

            intr = intrs[i]
            calibration_img = obs[f'color_{i}'][-1].copy()
            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_img_{device}.jpg', calibration_img)
            
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            if device_name in ['left-mid', 'right-mid']:  # 'left', 'right'
                calibration_img_left = copy.deepcopy(calibration_img)
                calibration_img_right = copy.deepcopy(calibration_img)

                calibration_img_left[:, int(calibration_img.shape[1] / 2):] = 0  # left
                calibration_img_right[:, :int(calibration_img.shape[1] / 2)] = 0  # right
                charuco_corners_left, charuco_ids_left, marker_corners_left, marker_ids_left = self.charuco_detector.detectBoard(calibration_img_left)
                charuco_corners_right, charuco_ids_right, marker_corners_right, marker_ids_right = self.charuco_detector.detectBoard(calibration_img_right)
                if charuco_corners_left is None:
                    charuco_corners_left = []
                if charuco_corners_right is None:
                    charuco_corners_right = []
                print(f"Detected {len(charuco_corners_left)} charuco corners on left half, {len(charuco_corners_right)} charuco corners on right half in {device_name}")

                if device_name in ['left-mid']:  # , 'left'
                    charuco_corners = charuco_corners_right
                    charuco_ids = charuco_ids_right
                    marker_corners = marker_corners_right
                    marker_ids = marker_ids_right
                
                if device_name in ['right-mid']:  # , 'right'
                    charuco_corners = charuco_corners_left
                    charuco_ids = charuco_ids_left
                    marker_corners = marker_corners_left
                    marker_ids = marker_ids_left
                
                # NOTE only use if using right-mid to calibrate between two calibration boards
                # if device_name == 'right-mid':
                #     retval_left, rvec_left, tvec_left = cv2.aruco.estimatePoseCharucoBoard(
                #         charuco_corners_left, 
                #         charuco_ids_left, 
                #         self.calibration_board, 
                #         cameraMatrix=intr, 
                #         distCoeffs=dist_coef,
                #         rvec=None,
                #         tvec=None,
                #     )
                #     retval_right, rvec_right, tvec_right = cv2.aruco.estimatePoseCharucoBoard(
                #         charuco_corners_right, 
                #         charuco_ids_right, 
                #         self.calibration_board, 
                #         cameraMatrix=intr, 
                #         distCoeffs=dist_coef,
                #         rvec=None,
                #         tvec=None,
                #     )
                #     R_leftboard2cam = cv2.Rodrigues(rvec_right)[0]  # left board looks like right
                #     t_leftboard2cam = tvec_right[:, 0]
                #     R_rightboard2cam = cv2.Rodrigues(rvec_left)[0]
                #     t_rightboard2cam = tvec_left[:, 0]

                #     T_leftboard2cam = np.eye(4)
                #     T_leftboard2cam[:3, :3] = R_leftboard2cam
                #     T_leftboard2cam[:3, 3] = t_leftboard2cam
                #     T_rightboard2cam = np.eye(4)
                #     T_rightboard2cam[:3, :3] = R_rightboard2cam
                #     T_rightboard2cam[:3, 3] = t_rightboard2cam
                    
                #     T_leftboard2rightboard = np.linalg.inv(T_rightboard2cam) @ T_leftboard2cam
                #     R_leftboard2rightboard = T_leftboard2rightboard[:3, :3]
                #     t_leftboard2rightboard = T_leftboard2rightboard[:3, 3]

                #     print(f"R_leftboard2rightboard: {R_leftboard2rightboard}")
                #     print(f"t_leftboard2rightboard: {t_leftboard2rightboard}")
                #     import ipdb; ipdb.set_trace()

            else:
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(calibration_img)
                if charuco_corners is None:
                    charuco_corners = []
                print(f"Detected {len(charuco_corners)} charuco corners in {device_name}")

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), marker_corners, marker_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{device}.jpg', calibration_img_vis)

                calibration_depth = obs[f'depth_{i}'][-1].copy()
                calibration_depth = np.minimum(calibration_depth, 2.0)
                calibration_depth_vis = calibration_depth / calibration_depth.max() * 255
                calibration_depth_vis = calibration_depth_vis[:, :, np.newaxis].repeat(3, axis=2)
                calibration_depth_vis = cv2.applyColorMap(calibration_depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(f'{self.vis_dir}/calibration_depth_{device}.jpg', calibration_depth_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, 
                charuco_ids, 
                self.calibration_board, 
                cameraMatrix=intr, 
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )
            
            if device_name in ['right-mid', 'right']:  # 'left'
                t_left2right = np.array([-0.01098454, -0.75598785, 0.00174724])
                R_left2right = np.array([[0.99964882, 0.02595905, 0.00532487],
                                         [-0.0259967, 0.99963663, 0.00712666],
                                         [-0.00513793, -0.00726259, 0.99996043]])
                T_left2right = np.eye(4)
                T_left2right[:3, :3] = R_left2right
                T_left2right[:3, 3] = t_left2right

                R_right2cam = cv2.Rodrigues(rvec)[0]
                t_right2cam = tvec[:, 0]
                T_right2cam = np.eye(4)
                T_right2cam[:3, :3] = R_right2cam
                T_right2cam[:3, 3] = t_right2cam

                T_left2cam = T_right2cam @ T_left2right
                R_left2cam = T_left2cam[:3, :3]
                t_left2cam = T_left2cam[:3, 3]

                rvec = cv2.Rodrigues(R_left2cam)[0]
                tvec = t_left2cam[:, np.newaxis]

            if not retval:
                print("pose estimation failed")
                import ipdb; ipdb.set_trace()

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{device}.jpg", calibration_img_vis)

            rvecs[device] = rvec
            tvecs[device] = tvec
            rvecs_list.append(rvec)
            tvecs_list.append(tvec)
        
        if save:
            # save rvecs, tvecs
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'wb') as f:
                pickle.dump(rvecs, f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'wb') as f:
                pickle.dump(tvecs, f)
            
            # save rvecs, tvecs, intrinsics as numpy array
            rvecs_list = np.array(rvecs_list)
            tvecs_list = np.array(tvecs_list)
            intrs = np.array(intrs)
            np.save(f'{self.calibrate_result_dir}/rvecs.npy', rvecs_list)
            np.save(f'{self.calibrate_result_dir}/tvecs.npy', tvecs_list)
            np.save(f'{self.calibrate_result_dir}/intrinsics.npy', intrs)

        if return_results:
            return rvecs, tvecs

    def calibrate(self, re_calibrate=False, visualize=True):
        if re_calibrate:
            rvecs, tvecs = self.fixed_camera_calibrate(visualize=visualize)
            R_base2board = np.array([
                [1.0, 0, 0],
                [0, -1.0, 0],
                [0, 0, -1.0]
            ])
            t_base2board = np.array(
                [-0.095, 0.085, -0.01]
            )
            self.R_leftbase2world = R_base2board
            self.t_leftbase2world = t_base2board

            R_rightbase2rightboard = R_base2board
            t_rightbase2rightboard = t_base2board
            T_rightbase2rightboard = np.eye(4)
            T_rightbase2rightboard[:3, :3] = R_rightbase2rightboard
            T_rightbase2rightboard[:3, 3] = t_rightbase2rightboard
            t_left2right = np.array([-0.01098454, -0.75598785, 0.00174724])
            R_left2right = np.array([[0.99964882, 0.02595905, 0.00532487],
                                     [-0.0259967, 0.99963663, 0.00712666],
                                     [-0.00513793, -0.00726259, 0.99996043]])
            T_left2right = np.eye(4)
            T_left2right[:3, :3] = R_left2right
            T_left2right[:3, 3] = t_left2right

            T_rightbase2leftboard = np.linalg.inv(T_left2right) @ T_rightbase2rightboard
            R_rightbase2leftboard = T_rightbase2leftboard[:3, :3]
            t_rightbase2leftboard = T_rightbase2leftboard[:3, 3]
            self.R_rightbase2world = R_rightbase2leftboard
            self.t_rightbase2world = t_rightbase2leftboard
            with open(f'{self.calibrate_result_dir}/base.pkl', 'wb') as f:
                pickle.dump({
                    'R_leftbase2world': R_base2board, 
                    't_leftbase2world': t_base2board,
                    'R_rightbase2world': R_rightbase2leftboard,
                    't_rightbase2world': t_rightbase2leftboard
                }, f)

            print('calibration finished')
        else:
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
                rvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
                tvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
                base = pickle.load(f)
            R_leftbase2world = base['R_leftbase2world']
            t_leftbase2world = base['t_leftbase2world']
            R_rightbase2world = base['R_rightbase2world']
            t_rightbase2world = base['t_rightbase2world']
            self.R_leftbase2world = R_leftbase2world
            self.t_leftbase2world = t_leftbase2world
            self.R_rightbase2world = R_rightbase2world
            self.t_rightbase2world = t_rightbase2world

        self.R_cam2world = {}
        self.t_cam2world = {}

        for i in range(self.n_fixed_cameras):
            device = self.serial_numbers[i]
            R_world2cam = cv2.Rodrigues(rvecs[device])[0]
            t_world2cam = tvecs[device][:, 0]
            self.R_cam2world[device] = R_world2cam.T
            self.t_cam2world[device] = -R_world2cam.T @ t_world2cam
        
        if visualize:
            self.verify_eef_points()
    
    def verify_eef_points(self):
        if self.last_realsense_data is None:
            _ = self.get_obs()
        eef_pos_left, eef_pos_right = self.get_eef_points()
        extr = self.get_extrinsics()
        intr = self.get_intrinsics()
        for i in range(self.n_fixed_cameras):
            color_img = self.last_realsense_data[i]['color'][-1].copy()
            for eef_pos in [eef_pos_left, eef_pos_right]:
                device = self.serial_numbers[i]
                R_cam2world = extr[0][i]
                t_cam2world = extr[1][i]
                R_world2cam = R_cam2world.T
                t_world2cam = -R_cam2world.T @ t_cam2world
                eef_pos_in_cam = R_world2cam @ eef_pos.T + t_world2cam[:, np.newaxis]
                eef_pos_in_cam = eef_pos_in_cam.T
                fx, fy, cx, cy = intr[i][0, 0], intr[i][1, 1], intr[i][0, 2], intr[i][1, 2]
                eef_pos_in_cam = eef_pos_in_cam / eef_pos_in_cam[:, 2][:, np.newaxis]
                eef_pos_in_cam = eef_pos_in_cam[:, :2]
                eef_pos_in_cam[:, 0] = eef_pos_in_cam[:, 0] * fx + cx
                eef_pos_in_cam[:, 1] = eef_pos_in_cam[:, 1] * fy + cy
                eef_pos_in_cam = eef_pos_in_cam.astype(int)
                for pos in eef_pos_in_cam:
                    cv2.circle(color_img, tuple(pos), 5, (255, 0, 0), -1)
                eef_pos_axis = np.array([[[0, 0, 0], [0, 0, 0.1]], [[0, 0, 0], [0, 0.1, 0]], [[0, 0, 0], [0.1, 0, 0]]])
                axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                for j, axis in enumerate(eef_pos_axis):
                    axis_eef = axis + eef_pos[0]
                    axis_in_cam = R_world2cam @ axis_eef.T + t_world2cam[:, np.newaxis]
                    axis_in_cam = axis_in_cam.T
                    axis_in_cam = axis_in_cam / axis_in_cam[:, 2][:, np.newaxis]
                    axis_in_cam = axis_in_cam[:, :2]
                    axis_in_cam[:, 0] = axis_in_cam[:, 0] * fx + cx
                    axis_in_cam[:, 1] = axis_in_cam[:, 1] * fy + cy
                    axis_in_cam = axis_in_cam.astype(int)
                    cv2.line(color_img, tuple(axis_in_cam[0]), tuple(axis_in_cam[1]), axis_colors[j], 2)
            cv2.imwrite(f'{self.vis_dir}/eef_in_cam_{device}.jpg', color_img)

        vis_3d = True
        if vis_3d:
            import open3d as o3d
            points_list = []
            colors_list = []
            for i in range(self.n_fixed_cameras):
                device = self.serial_numbers[i]
                color_img = self.last_realsense_data[i]['color'][-1].copy()
                depth_img = self.last_realsense_data[i]['depth'][-1].copy() / 1000.0
                intr = self.get_intrinsics()[i]
                extr = self.get_extrinsics()
                mask = np.logical_and(depth_img > 0, depth_img < 2.0).reshape(-1)
                mask = mask[:, None].repeat(3, axis=1)
                points = depth2fgpcd(depth_img, intr).reshape(-1, 3)
                colors = color_img.reshape(-1, 3)[:, ::-1]
                points = points[mask].reshape(-1, 3)
                colors = colors[mask].reshape(-1, 3)
                R_cam2world = extr[0][i]
                t_cam2world = extr[1][i]
                points = R_cam2world @ points.T + t_cam2world[:, np.newaxis]
                points_list.append(points.T)
                colors_list.append(colors)

            points = np.concatenate(points_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)
            
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=self.bbox[:, 0], max_bound=self.bbox[:, 1]))
            o3d.visualization.draw_geometries([pcd])
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            outliers = None
            new_outlier = None
            rm_iter = 0
            while new_outlier is None or len(new_outlier.points) > 0:
                _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0 + rm_iter * 0.5)
                new_pcd = pcd.select_by_index(inlier_idx)
                new_outlier = pcd.select_by_index(inlier_idx, invert=True)
                if outliers is None:
                    outliers = new_outlier
                else:
                    outliers += new_outlier
                pcd = new_pcd
                rm_iter += 1

            pcd_eef_left = o3d.geometry.PointCloud()
            pcd_eef_left.points = o3d.utility.Vector3dVector(eef_pos_left)
            pcd_eef_left.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
            pcd_eef_right = o3d.geometry.PointCloud()
            pcd_eef_right.points = o3d.utility.Vector3dVector(eef_pos_right)
            pcd_eef_right.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
            o3d.visualization.draw_geometries([pcd, pcd_eef_left, pcd_eef_right])

    def get_eef_points(self):
        assert self.R_leftbase2world is not None
        assert self.t_leftbase2world is not None
        assert self.R_rightbase2world is not None
        assert self.t_rightbase2world is not None
        R_leftgripper2base, t_leftgripper2base, R_rightgripper2base, t_rightgripper2base = self.get_robot_pose()
        R_leftgripper2world = self.R_leftbase2world @ R_leftgripper2base
        t_leftgripper2world = self.R_leftbase2world @ t_leftgripper2base + self.t_leftbase2world
        R_rightgripper2world = self.R_rightbase2world @ R_rightgripper2base
        t_rightgripper2world = self.R_rightbase2world @ t_rightgripper2base + self.t_rightbase2world
        left_stick_point_in_world = R_leftgripper2world @ self.eef_point.T + t_leftgripper2world[:, np.newaxis]
        left_stick_point_in_world = left_stick_point_in_world.T
        right_stick_point_in_world = R_rightgripper2world @ self.eef_point.T + t_rightgripper2world[:, np.newaxis]
        right_stick_point_in_world = right_stick_point_in_world.T
        return left_stick_point_in_world, right_stick_point_in_world

