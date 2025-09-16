# fr5_dataset.py
import os, glob, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple, Optional

from fr5_utils import (
    create_gt_heatmap,      # (pt_xy, (H, W), sigma) -> (H, W) float32
    angle_to_joint_coordinate,  # (angles_deg, view) -> (J, 3) in robot coords
    project_3d_to_2d            # (points_3d, aruco_result, K, dist) -> (J, 2)
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
    """상대/절대/anchor 포함 경로를 모두 DATASET_ROOT 기준으로 정규화"""
    if not p:
        return p
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
# 공통 유틸
# ---------------------------
def _scale_points(points_xy: np.ndarray,
                  from_size: Tuple[int,int],
                  to_size: Tuple[int,int]) -> np.ndarray:
    """points(x,y)을 from_size(W,H) -> to_size(W,H)로 선형 스케일"""
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = points_xy[:, 1] * (Ht / float(Hf))
    return out

def _parse_filename_for_view(filename: str) -> Tuple[str, str]:
    """
    zed_38007749_left_*.jpg -> ('38007749', 'left')
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"[FR5] Unexpected filename pattern: {filename}")
    return parts[1], parts[2]

# ---------------------------
# Dataset
# ---------------------------
class FR5_RobotPoseDataset(Dataset):
    """
    groups[i] 예시:
    {
      "joint_angles": [deg...],             # ★ FR5는 deg 사용(기존 파일 유지)
      "views": [{"image_path": ".../zed_38007749_left_xxx.jpg"}, ...],
      # (옵션) "pose_tag": "pose1" | "pose2" ...
    }
    """
    def __init__(self,
                 groups: List[Dict],
                 transform=None,
                 heatmap_size: Tuple[int,int]=(128,128),
                 sigma: float=5.0,
                 input_size: int=224):
        self.groups = groups
        self.transform = transform
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = float(sigma)
        self.input_size = int(input_size)

        print("[FR5] Loading and preprocessing metadata...")
        # 1) ArUco 요약
        aruco_json = _join_ds("Fr5_aruco_pose_summary.json")
        data = _safe_read_json(aruco_json)
        # 키: f"{pose}_{view}_{cam}" & f"{view}_{cam}" 둘 다 지원
        self.aruco_lookup: Dict[str, Dict] = {}
        for item in data:
            view = item.get("view"); cam = item.get("cam")
            pose = item.get("pose") or item.get("pose_tag") or ""
            if not view or not cam:
                continue
            k1 = f"{pose}_{view}_{cam}".lstrip('_')
            k2 = f"{view}_{cam}"
            self.aruco_lookup[k1] = item
            self.aruco_lookup[k2] = item

        # 2) Calibration
        self.calib_lookup: Dict[str, Dict] = {}
        for path in glob.glob(_join_ds("Fr5_calib_cam_from_conf", "*.json")):
            key = os.path.basename(path).replace("_calib.json", "")
            self.calib_lookup[key] = _safe_read_json(path)

        # 3) Serial → view
        self.serial_to_view = {"38007749": "left", "34850673": "right", "30779426": "top"}
        print("[FR5] ✅ Metadata loaded.")

    def __len__(self):
        return len(self.groups)

    def _pick_aruco(self, view: str, cam_key: str, pose_tag: Optional[str]) -> Dict:
        """pose_tag가 있으면 pose 우선 → 없으면 default(view_cam)"""
        if pose_tag:
            k1 = f"{pose_tag}_{view}_{cam_key}"
            if k1 in self.aruco_lookup:
                return self.aruco_lookup[k1]
        k2 = f"{view}_{cam_key}"
        if k2 in self.aruco_lookup:
            return self.aruco_lookup[k2]
        raise KeyError(f"[FR5] No ArUco for keys '{pose_tag}_{view}_{cam_key}' or '{view}_{cam_key}'")

    def __getitem__(self, idx):
        group = self.groups[idx]
        try:
            joint_angles_deg = np.asarray(group['joint_angles'], dtype=np.float64)  # ★ FR5=deg 유지
            gt_angles = torch.tensor(joint_angles_deg, dtype=torch.float32)

            image_dict: Dict[str, torch.Tensor] = {}
            heatmaps_dict: Dict[str, torch.Tensor] = {}

            pose_tag = group.get("pose_tag")

            for view_data in group['views']:
                raw_path = view_data['image_path']
                img_path = _resolve_path(raw_path)
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"[FR5] Image not found: {img_path}")

                serial, cam_type = _parse_filename_for_view(img_path)
                view = self.serial_to_view.get(serial, cam_type)  # 매핑 없으면 cam_type fallback
                cam_key = f"{cam_type}cam"                        # 'leftcam' 등

                # --- Calibration
                calib_key = f"{view}_{serial}_{cam_key}"
                if calib_key not in self.calib_lookup:
                    raise KeyError(f"[FR5] Missing calib: {calib_key}")
                calib = self.calib_lookup[calib_key]
                K = np.array(calib["camera_matrix"], dtype=np.float64)
                dist = np.array(calib["distortion_coeffs"], dtype=np.float64).reshape(-1,1)

                # --- ArUco
                aruco = self._pick_aruco(view, cam_key, pose_tag)

                # --- 이미지 로드 & undistort(K_new)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    raise FileNotFoundError(img_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
                undist = cv2.undistort(img_rgb, K, dist, None, K_new)

                # --- FK → 3D → 2D (주의: undist 좌표계에서는 dist=None)
                joints_3d = angle_to_joint_coordinate(joint_angles_deg, view)   # (J,3)
                kpts_2d   = project_3d_to_2d(joints_3d, aruco, K_new, dist=None).astype(np.float32)

                # --- 224×224로 비율무시 워핑 (모델 입력과 정확히 동일 변환)
                IN = self.input_size
                resized = cv2.resize(undist, (IN, IN), interpolation=cv2.INTER_LINEAR)
                kpts_on_IN = _scale_points(kpts_2d, from_size=(w, h), to_size=(IN, IN))

                # --- Heatmap 좌표로 스케일 (IN -> (Wt,Ht))
                Ht, Wt = self.heatmap_size
                kpts_hm = _scale_points(kpts_on_IN, from_size=(IN, IN), to_size=(Wt, Ht))

                # --- Heatmap 생성
                num_joints = joints_3d.shape[0]
                heatmaps = np.zeros((num_joints, Ht, Wt), dtype=np.float32)
                for j in range(num_joints):
                    heatmaps[j] = create_gt_heatmap(kpts_hm[j], (Ht, Wt), self.sigma)

                # --- 이미지 텐서 변환
                img_pil = Image.fromarray(resized)
                if self.transform:
                    img_tensor = self.transform(img_pil)
                else:
                    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0

                view_key = f"{serial}_{cam_type}"
                image_dict[view_key]    = img_tensor
                heatmaps_dict[view_key] = torch.from_numpy(heatmaps)

            return image_dict, heatmaps_dict, gt_angles

        except Exception as e:
            # 학습 루프에서 collate_fn으로 None 필터링 권장
            print(f"[FR5][idx={idx}] Error: {e}")
            return None, None, None

# ---------------------------
# DataLoader용: 실패 샘플 스킵
# ---------------------------
def collate_skip_none(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None
    images_list, heatmaps_list, angles_list = zip(*batch)
    return images_list, heatmaps_list, torch.stack(angles_list, dim=0)
