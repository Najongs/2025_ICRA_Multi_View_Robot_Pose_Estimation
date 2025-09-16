# utils.py
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

MAX_VIEWS_PER_GROUP = 8

# =========================================================
# 공용 상수
# =========================================================
@dataclass(frozen=True)
class RobotSpec:
    num_angles: int
    num_joints: int
    heatmap_size: Tuple[int, int]
    max_views_per_group: int

ROBOT_SPECS: Dict[str, RobotSpec] = {
    "fr3":     RobotSpec(num_angles=7, num_joints=8, heatmap_size=(128,128), max_views_per_group=8),
    "fr5":     RobotSpec(num_angles=6, num_joints=7, heatmap_size=(128,128), max_views_per_group=8),
    "meca500": RobotSpec(num_angles=6, num_joints=7, heatmap_size=(128,128), max_views_per_group=8),
}

def get_spec(robot: str) -> RobotSpec:
    rob = robot.lower()
    if rob not in ROBOT_SPECS:
        raise ValueError(f"Unknown robot: {robot}")
    return ROBOT_SPECS[rob]

# (선택) 레거시 코드 호환: 기존 전역 상수를 “로봇별 값”으로 갱신
def set_globals_for(robot: str) -> None:
    spec = get_spec(robot)
    globals()["NUM_ANGLES"] = spec.num_angles
    globals()["NUM_JOINTS"] = spec.num_joints
    globals()["HEATMAP_SIZE"] = spec.heatmap_size
    globals()["MAX_VIEWS_PER_GROUP"] = spec.max_views_per_group

# =========================================================
# ---- Heatmap Utilities
# =========================================================
def create_gt_heatmap(keypoint_2d: Sequence[float],
                      heatmap_size: Tuple[int, int],
                      sigma: float) -> np.ndarray:
    """
    단일 키포인트용 2D 가우시안 히트맵.
    keypoint_2d: (x, y) in pixel on heatmap grid (0..W-1, 0..H-1)
    return: (H, W) float32
    """
    H, W = heatmap_size
    x, y = float(keypoint_2d[0]), float(keypoint_2d[1])
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    heatmap = np.exp(-dist_sq / (2.0 * (sigma ** 2))).astype(np.float32)
    eps = np.finfo(np.float32).eps
    heatmap[heatmap < eps * heatmap.max()] = 0.0
    return heatmap


def create_multi_gt_heatmaps(keypoints_2d: np.ndarray,
                             heatmap_size: Tuple[int, int],
                             sigma: float,
                             visible: Optional[np.ndarray] = None) -> np.ndarray:
    """
    다중 키포인트(J개) 히트맵 벡터화 생성.
    keypoints_2d: (J,2)  (x,y) on heatmap grid
    visible: (J,) 1/0 mask. None이면 모두 1
    return: (J, H, W) float32
    """
    H, W = heatmap_size
    J = keypoints_2d.shape[0]
    if visible is None:
        visible = np.ones((J,), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    xx = xx[None, ...]
    yy = yy[None, ...]

    xs = keypoints_2d[:, 0].astype(np.float32)[:, None, None]
    ys = keypoints_2d[:, 1].astype(np.float32)[:, None, None]
    dist_sq = (xx - xs) ** 2 + (yy - ys) ** 2  # (J,H,W)
    heat = np.exp(-dist_sq / (2.0 * (sigma ** 2))).astype(np.float32)
    heat *= visible[:, None, None]
    eps = np.finfo(np.float32).eps
    heat[heat < eps * heat.max(axis=(1, 2), keepdims=True)] = 0.0
    return heat


def heatmap_argmax(hmap: np.ndarray) -> np.ndarray:
    """
    hmap: (J,H,W) -> (J,2)  (x,y)
    """
    J, H, W = hmap.shape
    idx = hmap.reshape(J, -1).argmax(axis=1)
    y = (idx // W).astype(np.float32)
    x = (idx %  W).astype(np.float32)
    return np.stack([x, y], axis=-1)


# =========================================================
# ---- Camera & Projection Utilities
# =========================================================
def build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Intrinsics K(3x3) 구성"""
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def resize_intrinsics(K: np.ndarray,
                      scale_x: float,
                      scale_y: float) -> np.ndarray:
    """
    이미지 리사이즈 시 K 보정.
    scale_x = newW / oldW, scale_y = newH / oldH
    """
    K = K.astype(np.float32).copy()
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K


def rodrigues_from_euler_deg(euler_deg: Sequence[float],
                             order: str = "zyx") -> np.ndarray:
    """오일러(도) -> Rodrigues rvec (3,)"""
    Rm = R.from_euler(order, euler_deg, degrees=True).as_matrix()
    rvec, _ = cv2.Rodrigues(Rm.astype(np.float32))
    return rvec.reshape(3)

def _euler_deg_to_rodrigues(euler_deg: Sequence[float], order: str = "zyx") -> np.ndarray:
    Rm = R.from_euler(order, euler_deg, degrees=True).as_matrix()
    rvec, _ = cv2.Rodrigues(Rm.astype(np.float32))
    return rvec.reshape(3).astype(np.float32)

def compose_Rt_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """rvec(3,), tvec(3,) -> Rt(3x4)"""
    Rm, _ = cv2.Rodrigues(rvec.astype(np.float32))
    Rt = np.concatenate([Rm, tvec.reshape(3, 1).astype(np.float32)], axis=1)
    return Rt

def project_3d_to_2d(points_3d: np.ndarray,
                     K: np.ndarray,
                     pose: Dict[str, Union[float, np.ndarray]],
                     dist: Optional[np.ndarray] = None,
                     mode: str = "auto",
                     euler_order: str = "zyx") -> np.ndarray:
    """
    범용 3D -> 2D 투영.
    pose 지원 형태:
      - Rodrigues: pose["rvec"](3,), pose["tvec"](3,)
      - Euler(deg): pose["rvec_x_deg","rvec_y_deg","rvec_z_deg"], pose["tvec_x","tvec_y","tvec_z"]
    mode:
      - "auto": 키를 보고 자동 판별
      - "rodrigues": 강제 Rodrigues
      - "euler_deg": 강제 Euler(deg)
    """
    if mode == "auto":
        if "rvec" in pose:
            mode = "rodrigues"
        elif ("rvec_x_deg" in pose) or ("rvec_y_deg" in pose) or ("rvec_z_deg" in pose):
            mode = "euler_deg"
        elif ("rvec_x" in pose) and ("rvec_y" in pose) and ("rvec_z" in pose) and pose.get("rvec_is_euler", False):
            mode = "euler_deg"
        else:
            mode = "rodrigues"  # 합리적 기본값(일반적인 ArUco)

    if mode == "rodrigues":
        rvec = np.array(pose["rvec"], dtype=np.float32).reshape(3)
        tvec = np.array(pose["tvec"], dtype=np.float32).reshape(3)
    elif mode == "euler_deg":
        # Euler(deg) → Rodrigues 변환
        rx = pose.get("rvec_x_deg", pose.get("rvec_x", 0.0))
        ry = pose.get("rvec_y_deg", pose.get("rvec_y", 0.0))
        rz = pose.get("rvec_z_deg", pose.get("rvec_z", 0.0))
        rvec = _euler_deg_to_rodrigues([rx, ry, rz], order=euler_order)
        tvec = np.array([
            pose.get("tvec_x", pose.get("mean_x", 0.0)),
            pose.get("tvec_y", pose.get("mean_y", 0.0)),
            pose.get("tvec_z", pose.get("mean_z", 0.0)),
        ], dtype=np.float32).reshape(3)
    else:
        raise ValueError(f"Unknown pose mode: {mode}")

    pts, _ = cv2.projectPoints(points_3d.astype(np.float32),
                               rvec.astype(np.float32),
                               tvec.astype(np.float32),
                               K.astype(np.float32),
                               dist)
    return pts.reshape(-1, 2).astype(np.float32)

def project_3d_to_2d_meca500_legacy(points_3d: np.ndarray,
                                    aruco_result: Dict[str, float],
                                    camera_matrix: np.ndarray,
                                    dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    MECA500 레거시 방식:
      - rvec_x/y/z 를 '축별 각도(deg)' 로 가정하고 rad로 변환 후
        Rodrigues 벡터처럼 cv2.projectPoints에 그대로 전달.
      - 현재 보유한 MECA500 GT 재현용.
    """
    # 안전한 키 접근 (rvec_*_deg 우선, 없으면 rvec_*)
    rvec_deg = np.array([
        aruco_result.get('rvec_x_deg', aruco_result.get('rvec_x', 0.0)),
        aruco_result.get('rvec_y_deg', aruco_result.get('rvec_y', 0.0)),
        aruco_result.get('rvec_z_deg', aruco_result.get('rvec_z', 0.0)),
    ], dtype=np.float32)
    rvec = np.deg2rad(rvec_deg).reshape(3, 1).astype(np.float32)

    tvec = np.array([
        aruco_result.get('tvec_x', aruco_result.get('mean_x', 0.0)),
        aruco_result.get('tvec_y', aruco_result.get('mean_y', 0.0)),
        aruco_result.get('tvec_z', aruco_result.get('mean_z', 0.0)),
    ], dtype=np.float32).reshape(3, 1)

    # OpenCV 입력 형식/dtype 정규화
    pts3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 1, 3)
    K = np.asarray(camera_matrix, dtype=np.float32)
    D = None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float32)

    pix, _ = cv2.projectPoints(pts3d, rvec, tvec, K, D)
    return pix.reshape(-1, 2).astype(np.float32)

def project_3d_to_2d_fr5_legacy(points_3d: np.ndarray,
                                aruco_result: Dict[str, float],
                                K_new: np.ndarray,
                                dist: Optional[np.ndarray] = None) -> np.ndarray:
    """
    FR5 레거시 방식:
      - rvec_x/y/z 를 '축별 각도(deg)' 로 가정하고 각 성분을 rad로 변환하여
        Rodrigues 벡터처럼 cv2.projectPoints에 그대로 전달.
      - 현재 보유한 FR5 GT 재현용.
    """
    Rvec = np.array([
        math.radians(aruco_result['rvec_x']),
        math.radians(aruco_result['rvec_y']),
        math.radians(aruco_result['rvec_z']),
    ], dtype=np.float32).reshape(3, 1)

    Tvec = np.array([
        aruco_result['tvec_x'],
        aruco_result['tvec_y'],
        aruco_result['tvec_z'],
    ], dtype=np.float32).reshape(3, 1)

    pix, _ = cv2.projectPoints(points_3d.astype(np.float32),
                               Rvec, Tvec,
                               K_new.astype(np.float32),
                               dist)
    return pix.reshape(-1, 2).astype(np.float32)

def project_3d_to_2d_by_robot(points_3d: np.ndarray,
                              robot: str,
                              aruco_result: Dict[str, float],
                              K: np.ndarray,
                              dist: Optional[np.ndarray] = None,
                              *,
                              force_legacy: Optional[bool] = None,
                              prefer_euler_keys: bool = False,
                              euler_order: str = "zyx") -> np.ndarray:
    """
    로봇별 3D→2D 투영 라우팅.
    - FR5 기본: 레거시 방식(project_3d_to_2d_fr5_legacy)
    - MECA500 기본: 레거시 방식(project_3d_to_2d_meca500_legacy)
    - 그 외(FR3 등): 안전 라우팅(project_3d_to_2d_aruco)
    파라미터:
      force_legacy: True면 레거시 강제, False면 안전 라우팅 강제, None이면 로봇별 기본.
      prefer_euler_keys: 안전 라우팅 사용 시 *_deg 키가 있으면 Euler로 처리할지 여부.
    """
    rob = (robot or "").lower()

    # 레거시 기본 선택: FR5, MECA500
    if force_legacy is None:
        use_legacy = (rob in ("fr5", "meca500", "meca"))
    else:
        use_legacy = bool(force_legacy)

    if use_legacy:
        if rob in ("meca500", "meca"):
            return project_3d_to_2d_meca500_legacy(points_3d, aruco_result, K, dist)
        elif rob == "fr5":
            return project_3d_to_2d_fr5_legacy(points_3d, aruco_result, K, dist)
        # 레거시 강제인데 다른 로봇이면 안전 라우팅으로 폴백
        return project_3d_to_2d_aruco(points_3d, aruco_result, K, dist,
                                      prefer_euler_keys=prefer_euler_keys,
                                      euler_order=euler_order)
    else:
        # 안전 라우팅
        return project_3d_to_2d_aruco(points_3d, aruco_result, K, dist,
                                      prefer_euler_keys=prefer_euler_keys,
                                      euler_order=euler_order)

def project_3d_to_2d_aruco(points_3d: np.ndarray,
                           aruco_result: Dict[str, float],
                           K: np.ndarray,
                           dist: Optional[np.ndarray] = None,
                           prefer_euler_keys: bool = False,
                           euler_order: str = "zyx") -> np.ndarray:
    """
    ArUco 결과 딕셔너리(rvec,tvec 혹은 rvec_*_deg, tvec_*)를 안전 처리.
    prefer_euler_keys=True면 *_deg가 있으면 Euler로 처리 후 Rodrigues 변환.
    """
    pose: Dict[str, Union[float, np.ndarray]] = {}
    if (not prefer_euler_keys) and ("rvec_x" in aruco_result and "rvec_y" in aruco_result and "rvec_z" in aruco_result):
        # (드물지만) Rodrigues가 세 성분으로 저장된 경우
        pose["rvec"] = np.array([aruco_result["rvec_x"], aruco_result["rvec_y"], aruco_result["rvec_z"]], dtype=np.float32)
        pose["tvec"] = np.array([aruco_result["tvec_x"], aruco_result["tvec_y"], aruco_result["tvec_z"]], dtype=np.float32)
        return project_3d_to_2d(points_3d, K, pose, dist, mode="rodrigues")

    # 기본: Euler(deg) 키 우선
    if ("rvec_x_deg" in aruco_result) or ("rvec_y_deg" in aruco_result) or ("rvec_z_deg" in aruco_result):
        pose.update({
            "rvec_x_deg": aruco_result.get("rvec_x_deg", aruco_result.get("rvec_x", 0.0)),
            "rvec_y_deg": aruco_result.get("rvec_y_deg", aruco_result.get("rvec_y", 0.0)),
            "rvec_z_deg": aruco_result.get("rvec_z_deg", aruco_result.get("rvec_z", 0.0)),
            "tvec_x": aruco_result.get("tvec_x", aruco_result.get("mean_x", 0.0)),
            "tvec_y": aruco_result.get("tvec_y", aruco_result.get("mean_y", 0.0)),
            "tvec_z": aruco_result.get("tvec_z", aruco_result.get("mean_z", 0.0)),
        })
        return project_3d_to_2d(points_3d, K, pose, dist, mode="euler_deg", euler_order=euler_order)

    # Rodrigues로 들어온 경우
    pose["rvec"] = np.array([aruco_result["rvec_x"], aruco_result["rvec_y"], aruco_result["rvec_z"]], dtype=np.float32)
    pose["tvec"] = np.array([aruco_result["tvec_x"], aruco_result["tvec_y"], aruco_result["tvec_z"]], dtype=np.float32)
    return project_3d_to_2d(points_3d, K, pose, dist, mode="rodrigues")


def pack_K_Rt_from_poses(poses: List[Dict[str, Union[float, np.ndarray]]],
                         Ks: List[np.ndarray],
                         pose_mode: str = "rodrigues") -> Tuple[np.ndarray, np.ndarray]:
    """
    여러 뷰의 (K, rvec/tvec 또는 euler_deg) -> (V,3,3), (V,3,4)
    """
    assert len(poses) == len(Ks), "poses and Ks length mismatch"
    V = len(poses)
    K_out = np.zeros((V, 3, 3), dtype=np.float32)
    Rt_out = np.zeros((V, 3, 4), dtype=np.float32)
    for i, (pose_i, K_i) in enumerate(zip(poses, Ks)):
        if pose_mode == "euler_deg":
            rx = pose_i.get("rvec_x_deg", pose_i["rvec_x"])
            ry = pose_i.get("rvec_y_deg", pose_i["rvec_y"])
            rz = pose_i.get("rvec_z_deg", pose_i["rvec_z"])
            rvec = rodrigues_from_euler_deg([rx, ry, rz])  # order 기본 zyx
            tvec = np.array([pose_i["tvec_x"], pose_i["tvec_y"], pose_i["tvec_z"]], dtype=np.float32)
        else:
            rvec = np.array(pose_i["rvec"], dtype=np.float32).reshape(3)
            tvec = np.array(pose_i["tvec"], dtype=np.float32).reshape(3)
        Rt_out[i] = compose_Rt_from_rvec_tvec(rvec, tvec)
        K_out[i] = K_i.astype(np.float32)
    return K_out, Rt_out


# =========================================================
# ---- Modified DH & Forward Kinematics
# =========================================================
@dataclass
class DHParam:
    a: float
    d: float
    alpha: float  # deg
    theta_offset: float = 0.0  # deg


def _get_modified_dh_matrix(a: float, d: float, alpha_deg: float, theta_deg: float) -> np.ndarray:
    """
    Modified DH transform (4x4).
    주의: 라이브러리/문헌마다 표기가 다를 수 있으니, 로봇별 파라미터와 함께 일관되게 사용.
    """
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct,            -st,            0.0,  a],
        [st * ca,  ct * ca,       -sa,  -d * sa],
        [st * sa,  ct * sa,        ca,   d * ca],
        [0.0,           0.0,       0.0,  1.0],
    ], dtype=np.float32)

def dh_matrix_standard(a: float, d: float, alpha_deg: float, theta_deg: float) -> np.ndarray:
    """
    Standard DH transform (4x4)
    """
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa,  a * ct],
        [st,  ct * ca, -ct * sa,  a * st],
        [0.,       sa,      ca,        d],
        [0.,      0.,      0.,       1.],
    ], dtype=np.float32)

# ---- FR3 (Franka Research 3) spec: 7 DOF -> 8 points (base + 7)
FR3_DH_PARAMETERS: List[DHParam] = [
    DHParam(a=0.0,      d=0.333, alpha=0,    theta_offset=0),
    DHParam(a=0.0,      d=0.0,   alpha=-90,  theta_offset=0),
    DHParam(a=0.0,      d=0.316, alpha=90,   theta_offset=0),
    DHParam(a=0.0825,   d=0.0,   alpha=90,   theta_offset=0),
    DHParam(a=-0.0825,  d=0.384, alpha=-90,  theta_offset=0),
    DHParam(a=0.0,      d=0.0,   alpha=90,   theta_offset=0),
    DHParam(a=0.088,    d=0.0,   alpha=90,   theta_offset=0),
    DHParam(a=0.0,      d=0.107, alpha=0,    theta_offset=0),  # EE/Tool
]

FR3_VIEW_ROTATIONS = {
    "view1": R.from_euler("zyx", [90, 180, 0], degrees=True),
    "view2": R.from_euler("zyx", [90, 180, 0], degrees=True),
    "view3": R.from_euler("zyx", [90, 180, 0], degrees=True),
    "view4": R.from_euler("zyx", [90, 180, 0], degrees=True),
}

# ---- FR5 spec (참고/겸용): 6 DOF -> 7 points
FR5_DH_PARAMETERS: List[DHParam] = [
    DHParam(alpha=90,   a=0.0,    d=0.152, theta_offset=0.0),
    DHParam(alpha=0,    a=-0.425, d=0.0,   theta_offset=0.0),
    DHParam(alpha=0,    a=-0.395, d=0.0,   theta_offset=0.0),
    DHParam(alpha=90,   a=0.0,    d=0.102, theta_offset=0.0),
    DHParam(alpha=-90,  a=0.0,    d=0.102, theta_offset=0.0),
    DHParam(alpha=0,    a=0.0,    d=0.100, theta_offset=0.0),
]
FR5_VIEW_ROTATIONS = {
    "top":   R.from_euler("zyx", [-85, 0, 180], degrees=True),
    "left":  R.from_euler("zyx", [180, 0, 90],  degrees=True),
    "right": R.from_euler("zyx", [0,   0, 90],  degrees=True),
}

MECA500_DH_PARAMETERS: List[DHParam] = [
    # alpha(deg), a(m),     d(m),    theta_offset(deg)
    DHParam(alpha=-90, a=0.000,  d=0.135, theta_offset=0),     # J1
    DHParam(alpha=0,   a=0.135,  d=0.000, theta_offset=-90),   # J2 (offset -90°)
    DHParam(alpha=-90, a=0.038,  d=0.000, theta_offset=0),     # J3
    DHParam(alpha=90,  a=0.000,  d=0.120, theta_offset=0),     # J4
    DHParam(alpha=-90, a=0.000,  d=0.000, theta_offset=0),     # J5
    DHParam(alpha=0,   a=0.000,  d=0.070, theta_offset=0),     # J6
]

# 베이스 좌표계 보정: X축 180° 후 Z축 90° (사용자 제공 코드를 그대로 반영)
# SciPy Rotation 곱셈은 오른쪽부터 적용되므로, rot_x_180 다음 rot_z_90이 적용됩니다.
_rot_x_180 = R.from_euler('x', 180, degrees=True)
_rot_z_90  = R.from_euler('z', 90,  degrees=True)
MECA500_BASE_CORRECTION = (_rot_z_90 * _rot_x_180).as_matrix().astype(np.float32)

# (옵션) 뷰별 추가 보정이 필요하면 여기 딕셔너리에 추가하세요.
MECA500_VIEW_ROTATIONS: Dict[str, R] = {
    "default": R.from_matrix(MECA500_BASE_CORRECTION),
    # 예) "view1": R.from_euler("zyx", [...], degrees=True),
}

def angle_to_joint_coordinate_MECA500(joint_angles: Sequence[float],
                                      selected_view: str = "default",
                                      input_unit: str = "deg") -> np.ndarray:
    """
    MECA500 FK (Standard DH):
      - 입력 관절각: 기본 deg (로그 스케일과 맞춤). rad이면 input_unit="rad".
      - 출력: 베이스 포함 7개 점 (0..6), shape=(7,3)
      - 베이스 좌표계 보정: X 180° → Z 90° (MECA500_BASE_CORRECTION)
      - selected_view에 추가 회전이 정의돼 있으면 그 회전으로 덮어씌우지 않고 '추가로' 곱하지 않습니다.
        (이미 사용자 코드에서 고정 보정으로 쓰이므로 기본은 base correction을 그대로 사용)
    """
    if input_unit == "rad":
        joint_deg = [math.degrees(a) for a in joint_angles]
    elif input_unit == "deg":
        joint_deg = list(joint_angles)
    else:
        raise ValueError("input_unit must be 'rad' or 'deg'")

    # 시작 변환 = 베이스 보정 회전
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = MECA500_BASE_CORRECTION.copy()

    # 필요 시, 뷰 이름으로 '추가' 베이스 보정을 하고 싶다면 아래 주석 해제 후 합성 규칙을 정하세요.
    # if selected_view in MECA500_VIEW_ROTATIONS:
    #     T[:3, :3] = (MECA500_VIEW_ROTATIONS[selected_view].as_matrix().astype(np.float32)) @ T[:3, :3]

    base_point = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    pts = [base_point[:3, 0].copy()]  # base

    # Standard DH로 누적
    for i, p in enumerate(MECA500_DH_PARAMETERS):
        theta = (joint_deg[i] if i < len(joint_deg) else 0.0) + p.theta_offset
        T = T @ dh_matrix_standard(p.a, p.d, p.alpha, theta)
        pts.append((T @ base_point)[:3, 0])

    return np.array(pts, dtype=np.float32)  # (7,3)


def angle_to_joint_coordinate_FR3(joint_angles: Sequence[float],
                                  selected_view: str = "view1",
                                  input_unit: str = "rad") -> np.ndarray:
    """
    FR3용 FK: 관절각 -> 3D 관절 좌표(베이스 포함 8점)
    joint_angles: len>=7 (마지막 EE는 고정 변환)
    input_unit: "rad" or "deg"
    """
    if input_unit == "rad":
        joint_deg = [math.degrees(a) for a in joint_angles]
    elif input_unit == "deg":
        joint_deg = list(joint_angles)
    else:
        raise ValueError("input_unit must be 'rad' or 'deg'")

    T = np.eye(4, dtype=np.float32)
    if selected_view in FR3_VIEW_ROTATIONS:
        T[:3, :3] = FR3_VIEW_ROTATIONS[selected_view].as_matrix().astype(np.float32)

    base_point = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    pts = [base_point[:3, 0].copy()]  # base

    for i, p in enumerate(FR3_DH_PARAMETERS):
        theta = (joint_deg[i] if i < len(joint_deg) else 0.0) + p.theta_offset
        T = T @ _get_modified_dh_matrix(p.a, p.d, p.alpha, theta)
        pts.append((T @ base_point)[:3, 0])

    return np.array(pts, dtype=np.float32)  # (8,3)


def angle_to_joint_coordinate_FR5(joint_angles: Sequence[float],
                                  selected_view: str = "top",
                                  input_unit: str = "deg") -> np.ndarray:
    """
    FR5용 FK: 관절각 -> 3D 관절 좌표(베이스 포함 7점)
    input_unit: "deg" (기본) or "rad"
    """
    if input_unit == "rad":
        joint_deg = [math.degrees(a) for a in joint_angles]
    elif input_unit == "deg":
        joint_deg = list(joint_angles)
    else:
        raise ValueError("input_unit must be 'rad' or 'deg'")

    T_base_correction = np.eye(4)
    if selected_view in FR5_VIEW_ROTATIONS:
        T_base_correction[:3, :3] = FR5_VIEW_ROTATIONS[selected_view].as_matrix().astype(np.float32)

    T_cumulative = T_base_correction
    base_point = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    pts = [base_point[:3, 0].copy()]  # base

    for i, p in enumerate(FR5_DH_PARAMETERS):
        
        theta = (joint_deg[i] if i < len(joint_deg) else 0.0) + p.theta_offset
        T_cumulative = T_cumulative @ _get_modified_dh_matrix(p.a, p.d, p.alpha, theta)
        pts.append((T_cumulative @ base_point)[:3, 0])
    return np.array(pts, dtype=np.float32)  # (7,3)


# 통합 FK 라우팅에 'meca500' 추가
def angle_to_joint_coordinate(joint_angles: Sequence[float],
                              robot: str = "fr3",
                              selected_view: Optional[str] = None,
                              input_unit: str = "rad") -> np.ndarray:
    """
    통합 FK 엔트리포인트.
    robot: 'fr3' | 'fr5' | 'meca500'
    """
    rob = robot.lower()
    if rob == "fr3":
        return angle_to_joint_coordinate_FR3(
            joint_angles, selected_view or "view1", input_unit=input_unit
        )
    elif rob == "fr5":
        iu = input_unit if input_unit in ("deg", "rad") else "deg"
        return angle_to_joint_coordinate_FR5(
            joint_angles, selected_view or "top", input_unit=iu
        )
    elif rob in ("meca500", "meca"):
        # MECA500은 기본 deg 로깅을 많이 쓰므로 기본을 "deg"로 둡니다.
        iu = input_unit if input_unit in ("deg", "rad") else "deg"
        return angle_to_joint_coordinate_MECA500(
            joint_angles, selected_view or "default", input_unit=iu
        )
    else:
        raise ValueError("robot must be 'fr3' or 'fr5' or 'meca500'")

# =========================================================
# ---- Grouping Utilities (스펙 기반으로 정리)
# =========================================================
def perform_grouping_fr3(df,
                         tolerance: float,
                         max_views: Optional[int] = None,
                         num_angles: Optional[int] = None) -> List[Dict]:
    """
    FR3 로깅 포맷 전용 그룹핑.
    필요한 컬럼:
      - 'robot_timestamp', 'image_path',
      - 'position_fr3_joint1'..'position_fr3_joint7'
    스펙 미지정 시 get_spec('fr3')에서 num_angles/max_views를 자동 적용.
    """
    spec = get_spec("fr3")
    if max_views is None:
        max_views = spec.max_views_per_group
    if num_angles is None:
        num_angles = spec.num_angles

    groups: List[Dict] = []
    if df is None or len(df) == 0:
        return groups

    if "robot_timestamp" in df.columns:
        df = df.sort_values("robot_timestamp", ascending=True)

    cur = []
    for _, row in df.iterrows():
        if not cur:
            cur.append(row); continue
        start = cur[0]["robot_timestamp"]
        time_ok = (row["robot_timestamp"] - start) <= tolerance
        size_ok = len(cur) < max_views
        if time_ok and size_ok:
            cur.append(row)
        else:
            joint_angles = [cur[0][f"position_fr3_joint{j}"] for j in range(1, num_angles + 1)]
            image_paths = [{"image_path": v["image_path"]} for v in cur]
            groups.append({"views": image_paths, "joint_angles": joint_angles})
            cur = [row]
    if cur:
        joint_angles = [cur[0][f"position_fr3_joint{j}"] for j in range(1, num_angles + 1)]
        image_paths = [{"image_path": v["image_path"]} for v in cur]
        groups.append({"views": image_paths, "joint_angles": joint_angles})
    return groups


def perform_grouping_meca500(df,
                             tolerance: float,
                             max_views: Optional[int] = None,
                             angle_prefix: str = "meca_joint",
                             angle_count: Optional[int] = None,
                             timestamp_col: str = "robot_timestamp") -> List[Dict]:
    """
    MECA500 전용/범용 그룹핑.
    스펙 미지정 시 get_spec('meca500')에서 angle_count/max_views를 자동 적용.
    """
    spec = get_spec("meca500")
    if max_views is None:
        max_views = spec.max_views_per_group
    if angle_count is None:
        angle_count = spec.num_angles

    groups: List[Dict] = []
    if df is None or len(df) == 0:
        return groups

    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col, ascending=True)

    joint_cols = [f"{angle_prefix}{j}" for j in range(1, angle_count + 1)]
    cur = []
    for _, row in df.iterrows():
        if not cur:
            cur.append(row); continue
        start = cur[0][timestamp_col]
        time_ok = (row[timestamp_col] - start) <= tolerance
        size_ok = len(cur) < max_views
        if time_ok and size_ok:
            cur.append(row)
        else:
            joint_angles = [cur[0][c] for c in joint_cols]
            image_paths = [{"image_path": v["image_path"]} for v in cur]
            groups.append({"views": image_paths, "joint_angles": joint_angles})
            cur = [row]
    if cur:
        joint_angles = [cur[0][c] for c in joint_cols]
        image_paths = [{"image_path": v["image_path"]} for v in cur]
        groups.append({"views": image_paths, "joint_angles": joint_angles})
    return groups


def perform_grouping_fr5(df,
                         tolerance: float,
                         max_views: Optional[int] = None,
                         angle_count: Optional[int] = None) -> List[Dict]:
    """
    FR5 로그 포맷 전용 그룹핑.
    필요한 컬럼:
      - 'joint_timestamp', 'image_path',
      - f'joint_1'..f'joint_{angle_count}'
    스펙 미지정 시 get_spec('fr5')에서 angle_count/max_views를 자동 적용.
    """
    spec = get_spec("fr5")
    if max_views is None:
        max_views = spec.max_views_per_group
    if angle_count is None:
        angle_count = spec.num_angles

    groups: List[Dict] = []
    if df is None or len(df) == 0:
        return groups

    if "joint_timestamp" in df.columns:
        df = df.sort_values("joint_timestamp", ascending=True)

    cur = []
    for _, row in df.iterrows():
        if not cur:
            cur.append(row); continue
        start = cur[0]["joint_timestamp"]
        time_ok = (row["joint_timestamp"] - start) <= tolerance
        size_ok = len(cur) < max_views
        if time_ok and size_ok:
            cur.append(row)
        else:
            joint_angles = [cur[0][f"joint_{j}"] for j in range(1, angle_count + 1)]
            image_paths = [{"image_path": v["image_path"]} for v in cur]
            groups.append({"views": image_paths, "joint_angles": joint_angles})
            cur = [row]
    if cur:
        joint_angles = [cur[0][f"joint_{j}"] for j in range(1, angle_count + 1)]
        image_paths = [{"image_path": v["image_path"]} for v in cur]
        groups.append({"views": image_paths, "joint_angles": joint_angles})
    return groups


def perform_grouping(df,
                     tolerance: float,
                     max_views: Optional[int] = None,
                     robot: str = "fr3") -> List[Dict]:
    """
    로봇 타입에 맞춰 그룹핑 라우팅.
    max_views가 None이면 해당 로봇 스펙에서 자동 사용.
    """
    rob = robot.lower()
    spec = get_spec(rob)
    if max_views is None:
        max_views = spec.max_views_per_group

    if rob == "fr3":
        return perform_grouping_fr3(df, tolerance, max_views=max_views)
    elif rob == "fr5":
        return perform_grouping_fr5(df, tolerance, max_views=max_views)
    elif rob in ("meca500", "meca"):
        return perform_grouping_meca500(df, tolerance, max_views=max_views)
    else:
        # 기본은 FR3 규칙
        return perform_grouping_fr3(df, tolerance, max_views=max_views)