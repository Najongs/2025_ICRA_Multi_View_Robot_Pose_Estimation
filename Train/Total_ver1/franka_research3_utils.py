# utils.py
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ===== 공용 상수 =====
MODEL_NAME = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
NUM_ANGLES = 7
NUM_JOINTS = 8
FEATURE_DIM = 768
HEATMAP_SIZE = (128, 128)
MAX_VIEWS_PER_GROUP = 8

# ===== GT heatmap =====
def create_gt_heatmap(keypoint_2d, heatmap_size, sigma):
    H, W = heatmap_size
    x, y = keypoint_2d
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    dist_sq = (xx - x)**2 + (yy - y)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    heatmap[heatmap < np.finfo(float).eps * heatmap.max()] = 0
    return heatmap

# ===== FK (Modified DH) =====
def _get_modified_dh_matrix(a, d, alpha, theta):
    alpha_rad, theta_rad = math.radians(alpha), math.radians(theta)
    cos_th, sin_th = np.cos(theta_rad), np.sin(theta_rad)
    cos_al, sin_al = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([
        [cos_th, -sin_th, 0, a],
        [sin_th * cos_al, cos_th * cos_al, -sin_al, -d * sin_al],
        [sin_th * sin_al, cos_th * sin_al,  cos_al,  d * cos_al],
        [0, 0, 0, 1]
    ])

def angle_to_joint_coordinate(joint_angles, selected_view):
    fr3_dh_parameters = [
        {'a': 0,       'd': 0.333, 'alpha': 0,   'theta_offset': 0},
        {'a': 0,       'd': 0,     'alpha': -90, 'theta_offset': 0},
        {'a': 0,       'd': 0.316, 'alpha': 90,  'theta_offset': 0},
        {'a': 0.0825,  'd': 0,     'alpha': 90,  'theta_offset': 0},
        {'a': -0.0825, 'd': 0.384, 'alpha': -90, 'theta_offset': 0},
        {'a': 0,       'd': 0,     'alpha': 90,  'theta_offset': 0},
        {'a': 0.088,   'd': 0,     'alpha': 90,  'theta_offset': 0},
        {'a': 0,       'd': 0.107, 'alpha': 0,   'theta_offset': 0}
    ]
    view_rot = {
        'view1': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view2': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view3': R.from_euler('zyx', [90, 180, 0], degrees=True),
        'view4': R.from_euler('zyx', [90, 180, 0], degrees=True),
    }
    T = np.eye(4)
    if selected_view in view_rot:
        T[:3, :3] = view_rot[selected_view].as_matrix()

    origin = np.array([0, 0, 0, 1])
    pts = [np.array([0, 0, 0])]
    for i, ang_rad in enumerate(joint_angles):
        p = fr3_dh_parameters[i]
        theta_deg = math.degrees(ang_rad) + p['theta_offset']
        T = T @ _get_modified_dh_matrix(p['a'], p['d'], p['alpha'], theta_deg)
        pts.append((T @ origin)[:3])
    return np.array(pts, dtype=np.float32)

# ===== 3D → 2D 투영 =====
def project_3d_to_2d(coords_3d, aruco_result, camera_matrix, dist_coeffs=None):
    rvec = np.array([aruco_result['rvec_x'], aruco_result['rvec_y'], aruco_result['rvec_z']])
    tvec = np.array([aruco_result['tvec_x'], aruco_result['tvec_y'], aruco_result['tvec_z']])
    pix, _ = cv2.projectPoints(coords_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return pix.reshape(-1, 2)

# ===== 그룹핑 =====
def perform_grouping(df, tolerance, max_views):
    groups = []
    if not df.empty:
        cur = []
        for _, row in df.iterrows():
            if not cur:
                cur.append(row); continue
            start = cur[0]['robot_timestamp']
            if (row['robot_timestamp'] - start > tolerance) or (len(cur) >= max_views):
                joint_angles = [cur[0][f'position_fr3_joint{j}'] for j in range(1, NUM_ANGLES + 1)]
                image_paths = [{'image_path': v['image_path']} for v in cur]
                groups.append({'views': image_paths, 'joint_angles': joint_angles})
                cur = [row]
            else:
                cur.append(row)
        if cur:
            joint_angles = [cur[0][f'position_fr3_joint{j}'] for j in range(1, NUM_ANGLES + 1)]
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
