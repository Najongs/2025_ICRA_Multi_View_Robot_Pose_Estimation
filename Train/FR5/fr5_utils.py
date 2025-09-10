# utils.py
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ===== 공용 상수 =====
MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
NUM_ANGLES = 6
NUM_JOINTS = 7
FEATURE_DIM = 768
HEATMAP_SIZE = (128, 128)
MAX_VIEWS_PER_GROUP = 8

# ===== GT heatmap =====
def create_gt_heatmap(keypoint_2d, HEATMAP_SIZE, sigma):
    H, W = HEATMAP_SIZE
    x, y = float(keypoint_2d[0]), float(keypoint_2d[1])
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2.0 * (sigma**2))).astype(np.float32)
    # 너무 작은 수치 zero로 컷(수치 안정)
    eps = np.finfo(np.float32).eps
    heatmap[heatmap < eps * heatmap.max()] = 0.0
    return heatmap  # (H, W), float32

# ===== FK (Modified DH) =====
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
    # FR5 DH parameters (degrees and meters)
    fr5_dh_parameters = [
        {'alpha': 90,  'a': 0,     'd': 0.152, 'theta': 0},
        {'alpha': 0,   'a': -0.425,'d': 0,     'theta': 0},
        {'alpha': 0,   'a': -0.395,'d': 0,     'theta': 0},
        {'alpha': 90,  'a': 0,     'd': 0.102, 'theta': 0},
        {'alpha': -90, 'a': 0,     'd': 0.102, 'theta': 0},
        {'alpha': 0,   'a': 0,     'd': 0.100, 'theta': 0}
    ]
    joint_coords_3d = [np.array([0, 0, 0])] # J0 (베이스)

    # 카메라 뷰에 따른 베이스 좌표계 보정 회전 정의
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
        params = fr5_dh_parameters[i]
        theta = joint_angles[i] + params['theta']
        T_i = get_dh_matrix(params['a'], params['d'], params['alpha'], theta)
        T_cumulative = T_cumulative @ T_i
        joint_pos = T_cumulative @ base_point
        joint_coords_3d.append(joint_pos[:3, 0])
    return np.array(joint_coords_3d, dtype=np.float32)

# ===== 3D → 2D 투영 =====
def project_3d_to_2d(joint_coords_3d, aruco_result, K_new, dist=None):
    Rvec = np.array([
        math.radians(aruco_result['rvec_x']),
        math.radians(aruco_result['rvec_y']),
        math.radians(aruco_result['rvec_z'])
    ], dtype=np.float32)
    Tvec = np.array([
        aruco_result['tvec_x'],
        aruco_result['tvec_y'],
        aruco_result['tvec_z']
    ], dtype=np.float32).reshape(3, 1)

    pixel_coords, _ = cv2.projectPoints(
        joint_coords_3d, Rvec, Tvec, K_new, dist
    )
    return pixel_coords.reshape(-1, 2)

# ===== 그룹핑 =====
def perform_grouping(df, tolerance, max_views):
    groups = []
    if not df.empty:
        cur = []
        for _, row in df.iterrows():
            if not cur:
                cur.append(row); continue
            start = cur[0]['joint_timestamp']
            if (row['joint_timestamp'] - start > tolerance) or (len(cur) >= max_views):
                joint_angles = [cur[0][f'joint_{j}'] for j in range(1, NUM_ANGLES + 1)]
                image_paths = [{'image_path': v['image_path']} for v in cur]
                groups.append({'views': image_paths, 'joint_angles': joint_angles})
                cur = [row]
            else:
                cur.append(row)
        if cur:
            joint_angles = [cur[0][f'joint_{j}'] for j in range(1, NUM_ANGLES + 1)]
            image_paths = [{'image_path': v['image_path']} for v in cur]
            groups.append({'views': image_paths, 'joint_angles': joint_angles})
    return groups

# ===== (선택) 지표 유틸 =====
def heatmap_argmax(hmap):  # (J,H,W) -> (J,2)
    J, H, W = hmap.shape
    out = []
    for j in range(J):
        idx = np.argmax(hmap[j])
        y, x = divmod(idx, W)
        out.append((x, y))
    return np.array(out, dtype=np.float32)
