# dataset.py (상단 import만 당신 프로젝트에 맞게)
import os, glob, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from fr5_utils import (
    create_gt_heatmap, angle_to_joint_coordinate, project_3d_to_2d
)

# ---------------------------
# 경로 유틸
# ---------------------------
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../.."))
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "Fr5")

def _join_ds(*parts):
    return os.path.abspath(os.path.join(DATASET_ROOT, *parts))

def _safe_read_json(abs_path):
    with open(abs_path, "r") as f:
        return json.load(f)

def _resolve_path(p: str) -> str:
    if not p: return p
    if os.path.isabs(p):
        return os.path.abspath(p)
    p_norm = os.path.normpath(p)
    parts = p_norm.split(os.sep)
    if 'Fr5' in parts:
        idx = parts.index('Fr5')
        tail = parts[idx+1:]
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail))
    if parts and parts[0] == 'dataset':
        tail = parts[1:]
        if tail and tail[0] == 'Fr5':
            return os.path.abspath(os.path.join(DATASET_ROOT, *tail[1:]))
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail))
    return os.path.abspath(os.path.join(DATASET_ROOT, p_norm))

# ---------------------------
# Dataset
# ---------------------------
class RobotPoseDataset(Dataset):
    def __init__(self, groups, transform=None, HEATMAP_SIZE=(128,128), sigma=5.0):
        self.groups = groups
        self.transform = transform
        self.heatmap_size = HEATMAP_SIZE
        self.sigma = float(sigma)

        print("Loading and preprocessing metadata...")
        # ArUco
        aruco_json = _join_ds("Fr5_aruco_pose_summary.json")
        data = _safe_read_json(aruco_json)
        # 키: f"{pose_tag}_{view}_{cam}" 혹은 f"{view}_{cam}" 둘 다 지원
        self.aruco_lookup = {}
        for item in data:
            key1 = f"{item.get('pose','')}_{item['view']}_{item['cam']}".lstrip('_')
            key2 = f"{item['view']}_{item['cam']}"
            self.aruco_lookup[key1] = item
            self.aruco_lookup[key2] = item

        # Calib
        self.calib_lookup = {}
        for path in glob.glob(_join_ds("Fr5_calib_cam_from_conf", "*.json")):
            key = os.path.basename(path).replace("_calib.json", "")
            self.calib_lookup[key] = _safe_read_json(path)

        # Serial → view 매핑
        self.serial_to_view = {"38007749": "left", "34850673": "right", "30779426": "top"}
        print("✅ Metadata loaded.")

    def __len__(self): 
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        try:
            joint_angles_rad = np.asarray(group['joint_angles'], dtype=np.float64)
            # 라벨은 (학습 목적에 맞게) deg로 들고가든 rad로 들고가든 상관없음
            gt_angles = torch.tensor(np.degrees(joint_angles_rad), dtype=torch.float32)

            image_dict, heatmaps_dict = {}, {}

            for view_data in group['views']:
                raw_path = view_data['image_path']
                img_path = _resolve_path(raw_path)
                filename = os.path.basename(img_path)  # zed_38007749_left_*.jpg
                parts = filename.split('_')
                serial, cam_type = parts[1], parts[2]      # '38007749', 'left'
                view = self.serial_to_view[serial]         # 'left'
                cam_key = f"{cam_type}cam"                 # 'leftcam'

                # Calib key: f"{view}_{serial}_{cam}cam"
                calib_key = f"{view}_{serial}_{cam_key}"
                calib = self.calib_lookup[calib_key]
                K = np.array(calib["camera_matrix"], dtype=np.float64)
                dist = np.array(calib["distortion_coeffs"], dtype=np.float64).reshape(-1,1)

                # 어떤 pose 세트인지 추정
                pose_tag = 'pose1' if ('pose1' in img_path or 'pose1' in raw_path) else 'pose2'
                # ArUco key 우선순위
                aruco = ( self.aruco_lookup.get(f"{pose_tag}_{view}_{cam_key}")
                          or self.aruco_lookup[f"{view}_{cam_key}"] )

                # Read + undistort with K_new
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    raise FileNotFoundError(img_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
                undist = cv2.undistort(img_rgb, K, dist, None, K_new)

                # FK → 3D joints (meters), view-base correction inside
                joints_3d = angle_to_joint_coordinate(joint_angles_rad, view)

                # 3D→2D with K_new, dist=None
                kpts_2d = project_3d_to_2d(joints_3d, aruco, K_new, dist=None)  # (7,2), float32

                # Heatmap 스케일 정합 (원본 undist 크기 기준)
                Ht, Wt = self.heatmap_size
                sx, sy = (Wt / w), (Ht / h)
                kpts_scaled = np.empty_like(kpts_2d)
                kpts_scaled[:, 0] = kpts_2d[:, 0] * sx
                kpts_scaled[:, 1] = kpts_2d[:, 1] * sy

                # Heatmaps
                num_joints = joints_3d.shape[0]
                heatmaps = np.zeros((num_joints, Ht, Wt), dtype=np.float32)
                for j in range(num_joints):
                    heatmaps[j] = create_gt_heatmap(kpts_scaled[j], self.heatmap_size, self.sigma)

                # To tensor
                img_pil = Image.fromarray(undist)
                img_tensor = self.transform(img_pil) if self.transform else torch.from_numpy(undist).permute(2,0,1).float()/255.

                view_key = f"{serial}_{cam_type}"  # 예: "38007749_left"
                image_dict[view_key] = img_tensor
                heatmaps_dict[view_key] = torch.from_numpy(heatmaps)  # (J,H,W) float32

            return image_dict, heatmaps_dict, gt_angles
        except Exception as e:
            # 학습 루프에서 collate_fn으로 None 필터링 권장
            print(f"[Dataset] Error at idx={idx}: {e}")
            return None, None, None
