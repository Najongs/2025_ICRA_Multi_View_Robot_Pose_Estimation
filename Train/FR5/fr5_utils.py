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
def get_dh_matrix(a, d, alpha_deg, theta_deg):
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0 ,       sa,       ca,      d],
        [0 ,        0,        0,      1],
    ], dtype=np.float64)

def angle_to_joint_coordinate(joint_angles_rad, selected_view):
    """
    joint_angles_rad: list/np.ndarray of 6 joint angles in radians
    returns: (7,3) J0~J6 in camera base-corrected 3D (meters)
    """
    # FR5 DH: (alpha, a, d) in deg/m, theta is variable (deg)
    fr5_dh_parameters = [
        {'alpha':  90, 'a':  0.000, 'd': 0.152, 'theta': 0.0},
        {'alpha':   0, 'a': -0.425, 'd': 0.000, 'theta': 0.0},
        {'alpha':   0, 'a': -0.395, 'd': 0.000, 'theta': 0.0},
        {'alpha':  90, 'a':  0.000, 'd': 0.102, 'theta': 0.0},
        {'alpha': -90, 'a':  0.000, 'd': 0.102, 'theta': 0.0},
        {'alpha':   0, 'a':  0.000, 'd': 0.100, 'theta': 0.0},
    ]
    # base correction for each view (deg)
    view_euler_zyx_deg = {
        'top':   (-85, 0, 180),
        'left':  (180, 0,  90),
        'right': (  0, 0,  90),
    }
    T_base = np.eye(4, dtype=np.float64)
    if selected_view in view_euler_zyx_deg:
        z, y, x = view_euler_zyx_deg[selected_view]
        T_base[:3, :3] = R.from_euler('zyx', [z, y, x], degrees=True).as_matrix()

    T = T_base.copy()
    pts = [np.array([0., 0., 0.], dtype=np.float64)]  # J0
    base_point = np.array([0., 0., 0., 1.], dtype=np.float64)

    # Convert rad -> deg once
    joint_angles_deg = np.degrees(np.asarray(joint_angles_rad, dtype=np.float64))

    for i, prm in enumerate(fr5_dh_parameters):
        theta_deg = joint_angles_deg[i] + prm['theta']
        T = T @ get_dh_matrix(prm['a'], prm['d'], prm['alpha'], theta_deg)
        p = T @ base_point
        pts.append(p[:3])
    return np.asarray(pts, dtype=np.float32)  # (7,3)

# ===== 3D → 2D 투영 =====
def project_3d_to_2d(joint_coords_3d, aruco_result, K_new, dist=None):
    """
    - aruco_result가 저장된 방식에 따라 rvec 해석:
      1) rvec_x/y/z 가 이미 Rodrigues 벡터(라디안*축)인 경우 그대로 사용
      2) rvec_*가 오일러(deg)라면 'rvec_mode': 'euler_deg' 키를 json에 넣어 구분
    """
    # 기본: Rodrigues 벡터(라디안 단위의 회전벡터)로 가정
    rvec_mode = aruco_result.get('rvec_mode', 'rodrigues')
    if rvec_mode == 'euler_deg':
        # zyx(euler deg) -> R -> Rodrigues
        rz, ry, rx = aruco_result['rvec_z'], aruco_result['rvec_y'], aruco_result['rvec_x']
        R_cam = R.from_euler('zyx', [rz, ry, rx], degrees=True).as_matrix().astype(np.float64)
        rvec, _ = cv2.Rodrigues(R_cam)
    else:
        # 이미 로드리게스 벡터 저장 가정
        rvec = np.array([
            [aruco_result['rvec_x']],
            [aruco_result['rvec_y']],
            [aruco_result['rvec_z']],
        ], dtype=np.float64)

    tvec = np.array([
        [aruco_result['tvec_x']],
        [aruco_result['tvec_y']],
        [aruco_result['tvec_z']],
    ], dtype=np.float64)

    # OpenCV는 float64 선호
    pts3d = np.asarray(joint_coords_3d, dtype=np.float64).reshape(-1, 1, 3)
    imgpts, _ = cv2.projectPoints(pts3d, rvec, tvec, K_new.astype(np.float64), dist)
    return imgpts.reshape(-1, 2).astype(np.float32)

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
