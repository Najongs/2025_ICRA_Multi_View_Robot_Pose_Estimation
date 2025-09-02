# dataset.py

import os
import glob
import json
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
from PIL import Image

# --- Ground Truth 생성 Helper 함수 ---

def create_gt_heatmap(keypoint_2d, heatmap_size, sigma):
    H, W = heatmap_size
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    exponent = dist_sq / (2 * sigma**2)
    heatmap = np.exp(-exponent)
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

def get_dh_matrix(a, d, alpha, theta):
    alpha_rad = math.radians(alpha)
    theta_rad = math.radians(theta)
    return np.array([
        [np.cos(theta_rad), -np.sin(theta_rad) * np.cos(alpha_rad),  np.sin(theta_rad) * np.sin(alpha_rad), a * np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad) * np.cos(alpha_rad), -np.cos(theta_rad) * np.sin(alpha_rad), a * np.sin(theta_rad)],
        [0, np.sin(alpha_rad), np.cos(alpha_rad), d],
        [0, 0, 0, 1]
    ])

def angle_to_joint_coordinate(joint_angles, selected_view):
    fr5_dh_parameters = [
        {'alpha': 90,  'a': 0,     'd': 0.152, 'theta': 0},
        {'alpha': 0,   'a': -0.425,'d': 0,     'theta': 0},
        {'alpha': 0,   'a': -0.395,'d': 0,     'theta': 0},
        {'alpha': 90,  'a': 0,     'd': 0.102, 'theta': 0},
        {'alpha': -90, 'a': 0,     'd': 0.102, 'theta': 0},
        {'alpha': 0,   'a': 0,     'd': 0.100, 'theta': 0}
    ]
    joint_coords_3d = [np.array([0, 0, 0])]
    view_rotations = {
        'top': R.from_euler('zyx', [-85, 0, 180], degrees=True),
        'left': R.from_euler('zyx', [180, 0, 90], degrees=True),
        'right': R.from_euler('zyx', [0, 0, 90], degrees=True)
    }
    T_base_correction = np.eye(4)
    if selected_view in view_rotations:
        T_base_correction[:3, :3] = view_rotations[selected_view].as_matrix()
    T_cumulative = T_base_correction
    base_point = np.array([[0], [0], [0], [1]])
    for i in range(6):
        params, theta = fr5_dh_parameters[i], joint_angles[i] + fr5_dh_parameters[i]['theta']
        T_i = get_dh_matrix(params['a'], params['d'], params['alpha'], theta)
        T_cumulative = T_cumulative @ T_i
        joint_coords_3d.append((T_cumulative @ base_point)[:3, 0])
    return np.array(joint_coords_3d, dtype=np.float32)

def joint_coordinate_to_pixel_plane(joint_coords, aruco_result, camera_matrix, dist_coeffs):
    rvec = np.array([math.radians(aruco_result[f'rvec_{axis}']) for axis in 'xyz'], dtype=np.float32)
    tvec = np.array([aruco_result[f'tvec_{axis}'] for axis in 'xyz'], dtype=np.float32).reshape(3, 1)
    pixel_coords, _ = cv2.projectPoints(joint_coords, rvec, tvec, camera_matrix, dist_coeffs)
    return pixel_coords.reshape(-1, 2)

# --- 데이터셋 클래스 ---

class RobotPoseDataset(Dataset):
    def __init__(self, pairs, transform=None, heatmap_size=(128, 128), sigma=2.0):
        self.pairs = pairs
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        print("Loading and preprocessing metadata...")
        aruco_path = '../dataset/Fr5/Fr5_aruco_pose_summary.json'
        with open(aruco_path, 'r') as f:
            self.aruco_lookup = {f"{item['view']}_{item['cam']}": item for item in json.load(f)}
        self.calib_lookup = {}
        calib_dir = "../dataset/Fr5/Fr5_calib_cam_from_conf"
        for calib_path in glob.glob(os.path.join(calib_dir, "*.json")):
            filename = os.path.basename(calib_path).replace("_calib.json", "")
            with open(calib_path, 'r') as f:
                self.calib_lookup[filename] = json.load(f)
        print("Metadata loaded.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        image_path, joint_path = pair['image_path'], pair['joint_path']
        filename = os.path.basename(image_path)
        parts = filename.split('_')
        serial_str, selected_cam = parts[1], parts[2] + "cam"
        serial_to_view = {"38007749": "left", "34850673": "right", "30779426": "top"}
        selected_view = serial_to_view[serial_str]
        calib = self.calib_lookup[f"{selected_view}_{serial_str}_{selected_cam}"]
        camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(calib["distortion_coeffs"], dtype=np.float32)
        image = Image.open(image_path).convert('RGB')
        undistorted_image_np = cv2.undistort(np.array(image), camera_matrix, dist_coeffs)
        image_tensor = self.transform(Image.fromarray(undistorted_image_np))
        aruco_result = self.aruco_lookup[f"{selected_view}_{selected_cam}"]
        with open(joint_path, 'r') as f:
            joint_angle_data = json.load(f)
        gt_angles = torch.tensor(joint_angle_data, dtype=torch.float32)
        joint_coords = angle_to_joint_coordinate(joint_angle_data, selected_view)
        pixel_coords = joint_coordinate_to_pixel_plane(joint_coords, aruco_result, camera_matrix, dist_coeffs)
        original_h, original_w, _ = undistorted_image_np.shape
        scaled_keypoints = pixel_coords * np.array([self.heatmap_size[1] / original_w, self.heatmap_size[0] / original_h])
        gt_heatmaps_np = np.zeros((len(pixel_coords), self.heatmap_size[0], self.heatmap_size[1]), dtype=np.float32)
        for i, (px, py) in enumerate(pixel_coords):
            if 0 <= px < original_w and 0 <= py < original_h:
                gt_heatmaps_np[i] = create_gt_heatmap(scaled_keypoints[i], self.heatmap_size, self.sigma)
        gt_heatmaps = torch.from_numpy(gt_heatmaps_np)
        return image_tensor, gt_heatmaps, gt_angles