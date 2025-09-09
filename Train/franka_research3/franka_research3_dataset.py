# dataset.py
import os, glob, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from franka_research3_utils import (
    NUM_JOINTS, NUM_ANGLES, HEATMAP_SIZE,
    create_gt_heatmap, angle_to_joint_coordinate, project_3d_to_2d
)

# === 경로 유틸 ===
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../.."))
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "franka_research3")

def _join_ds(*parts):
    """데이터셋 루트에 상대경로를 붙여 절대경로로 바꿔준다."""
    return os.path.abspath(os.path.join(DATASET_ROOT, *parts))

def _safe_read_json(abs_path):
    with open(abs_path, "r") as f:
        return json.load(f)

def _resolve_path(p: str) -> str:
    """
    CSV의 image_path가 다음 어떤 형태여도 올바른 절대경로로 변환한다.
      - 절대경로: 그대로 반환
      - 'dataset/franka_research3/...'
      - './dataset/franka_research3/...'
      - '../dataset/franka_research3/...'
      - 'franka_research3/...'
      - 그 외 상대경로
    규칙: 경로 안에 'franka_research3'가 있으면 그 다음부터만 잘라서 DATASET_ROOT 뒤에 붙인다.
    """
    if not p:
        return p
    # 이미 절대경로면 그대로
    if os.path.isabs(p):
        return os.path.abspath(p)

    # 정규화
    p_norm = os.path.normpath(p)

    # 경로 세그먼트 분리
    parts = p_norm.split(os.sep)

    # 'franka_research3'가 포함된 경우 → 그 이후만 붙이기
    if 'franka_research3' in parts:
        idx = parts.index('franka_research3')
        tail_parts = parts[idx + 1:]  # franka_research3 뒤쪽만
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail_parts))

    # 'dataset' 접두가 붙은 경우 ('dataset/...') → 한 번 더 정규화해서 검사
    if parts and parts[0] == 'dataset':
        # dataset 다음에 바로 franka_research3가 오지 않는 특이 케이스도 커버
        # (위 if에서 이미 걸렀을 확률이 높지만 안전망으로 둠)
        tail_parts = parts[1:]
        if tail_parts and tail_parts[0] == 'franka_research3':
            return os.path.abspath(os.path.join(DATASET_ROOT, *tail_parts[1:]))
        # 일반 fallback
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail_parts))

    # 그 외 상대경로는 DATASET_ROOT 기준으로 붙이기
    return os.path.abspath(os.path.join(DATASET_ROOT, p_norm))


class RobotPoseDataset(Dataset):
    def __init__(self, groups, transform=None, HEATMAP_SIZE=(128, 128), sigma=5.0):
        self.groups, self.transform, self.heatmap_size, self.sigma = groups, transform, HEATMAP_SIZE, sigma
        print("Loading and preprocessing metadata...")
        self.aruco_lookup, self.calib_lookup = {}, {}

        # pose1/pose2 아루코 요약 파일 절대경로로 로드
        for pose in ['pose1', 'pose2']:
            aruco_json = _join_ds(f"{pose}_aruco_pose_summary.json")
            data = _safe_read_json(aruco_json)
            for item in data:
                self.aruco_lookup[f"{pose}_{item['view']}_{item['cam']}"] = item

        # 카메라 캘리브 json들 절대경로로 로드
        calib_glob = _join_ds("franka_research3_calib_cam_from_conf", "*.json")
        for path in glob.glob(calib_glob):
            key = os.path.basename(path).replace("_calib.json", "")
            self.calib_lookup[key] = _safe_read_json(path)

        self.serial_to_view = {
            '41182735': "view1", '49429257': "view2",
            '44377151': "view3", '49045152': "view4"
        }
        print("✅ Metadata loaded.")


    def __len__(self): return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        try:
            joint_angle_data_rad = group['joint_angles']
            joint_angle_data_deg = np.degrees(joint_angle_data_rad)
            gt_angles = torch.tensor(joint_angle_data_deg, dtype=torch.float32)

            image_dict, gt_heatmaps_dict = {}, {}
            for view_data in group['views']:
                raw_path = view_data['image_path']
                path = _resolve_path(raw_path)  # 절대/상대 모두 안전하게 처리

                parts = os.path.basename(path).split('_')
                serial, cam_type = parts[1], parts[2]
                view = self.serial_to_view[serial]
                view_key = f"{serial}_{cam_type}"

                calib = self.calib_lookup[f"{view}_{serial}_{cam_type}cam"]
                cam_mat, dist_coeff = np.array(calib["camera_matrix"]), np.array(calib["distortion_coeffs"])
                pose_tag = 'pose1' if 'pose1' in path or 'pose1' in raw_path else 'pose2'
                aruco = self.aruco_lookup[f"{pose_tag}_{view}_{cam_type}cam"]

                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    raise FileNotFoundError(f"Image not found: {path}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]

                # undistort + new intrinsics
                K_new, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeff, (w, h), alpha=0)
                undistorted_np = cv2.undistort(img_rgb, cam_mat, dist_coeff, None, K_new)

                # 3D -> 2D (K_new, dist=None)
                joint_coords_3d = angle_to_joint_coordinate(joint_angle_data_rad, view)
                coords_2d = project_3d_to_2d(joint_coords_3d, aruco, K_new, None)

                h, w, _ = undistorted_np.shape
                scaled_kpts = coords_2d * [self.heatmap_size[1]/w, self.heatmap_size[0]/h]

                heatmaps_np = np.zeros((NUM_JOINTS, *self.heatmap_size), dtype=np.float32)
                for i in range(NUM_JOINTS):
                    heatmaps_np[i] = create_gt_heatmap(scaled_kpts[i], self.heatmap_size, self.sigma)

                image_dict[view_key] = self.transform(Image.fromarray(undistorted_np))
                gt_heatmaps_dict[view_key] = torch.from_numpy(heatmaps_np)

            return image_dict, gt_heatmaps_dict, gt_angles
        except Exception:
            return None, None, None
