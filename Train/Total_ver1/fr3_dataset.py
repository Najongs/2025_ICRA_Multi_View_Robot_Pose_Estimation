# dataset.py (franka_research3)
import os, glob, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple, Optional

from utils import get_spec
from franka_research3_utils import (
    create_gt_heatmap, angle_to_joint_coordinate, project_3d_to_2d
)

# === 경로 유틸 ===
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../.."))
DATASET_ROOT = os.path.join(_PROJECT_ROOT, "dataset", "franka_research3")

def _join_ds(*parts):
    return os.path.abspath(os.path.join(DATASET_ROOT, *parts))

def _safe_read_json(abs_path):
    with open(abs_path, "r") as f:
        return json.load(f)

def _resolve_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return os.path.abspath(p)
    p_norm = os.path.normpath(p)
    parts = p_norm.split(os.sep)
    if 'franka_research3' in parts:
        idx = parts.index('franka_research3')
        tail = parts[idx + 1:]
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail))
    if parts and parts[0] == 'dataset':
        tail = parts[1:]
        if tail and tail[0] == 'franka_research3':
            return os.path.abspath(os.path.join(DATASET_ROOT, *tail[1:]))
        return os.path.abspath(os.path.join(DATASET_ROOT, *tail))
    return os.path.abspath(os.path.join(DATASET_ROOT, p_norm))

# === 공통 유틸 ===
def _scale_points(points_xy: np.ndarray,
                  from_size: Tuple[int,int],
                  to_size: Tuple[int,int]) -> np.ndarray:
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = points_xy[:, 1] * (Ht / float(Hf))
    return out

def _parse_filename_for_view(filename: str) -> Tuple[str, str]:
    # e.g., zed_41182735_left_*.jpg -> ('41182735', 'left')
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) < 3:
        raise ValueError(f"[FR3] Unexpected filename pattern: {filename}")
    return parts[1], parts[2]  # serial, cam_type

class FR3_RobotPoseDataset(Dataset):
    """
    groups[i]:
    {
      "joint_angles": [rad...],   # ★ FR3는 원본 rad 유지
      "views": [{"image_path": ".../zed_41182735_left_xxx.jpg"}, ...],
      # (opt) "pose_tag": "pose1" | "pose2"
    }
    """
    def __init__(self,
                 groups,
                 transform=None,
                 sigma=5.0,
                 input_size: int = 224):
        self.spec = get_spec("fr3")  # ✅ 단일 소스
        self.num_joints = get_spec("fr3").num_joints  # ★ 스펙 기반(=8)
        self.heatmap_size = self.spec.heatmap_size
        self.max_views = self.spec.max_views_per_group
        self.groups = groups
        self.transform = transform
        self.sigma = float(sigma)
        self.input_size = int(input_size)
        
        print("[FR3] Loading and preprocessing metadata...")
        self.aruco_lookup: Dict[str, Dict] = {}
        self.calib_lookup: Dict[str, Dict] = {}

        # pose1/pose2 아루코 요약 로드 (+ view_cam 기본키까지 등록)
        for pose in ['pose1', 'pose2']:
            aruco_json = _join_ds(f"{pose}_aruco_pose_summary.json")
            data = _safe_read_json(aruco_json)
            for item in data:
                view, cam = item.get('view'), item.get('cam')
                if not view or not cam:
                    continue
                self.aruco_lookup[f"{pose}_{view}_{cam}"] = item
                # 기본키(fallback)도 최신 데이터로 갱신
                self.aruco_lookup[f"{view}_{cam}"] = item

        # 카메라 캘리브 로드
        calib_glob = _join_ds("franka_research3_calib_cam_from_conf", "*.json")
        for path in glob.glob(calib_glob):
            key = os.path.basename(path).replace("_calib.json", "")
            self.calib_lookup[key] = _safe_read_json(path)

        # 시리얼→뷰
        self.serial_to_view = {
            '41182735': "view1", '49429257': "view2",
            '44377151': "view3", '49045152': "view4"
        }
        print("[FR3] ✅ Metadata loaded.")

    def __len__(self): 
        return len(self.groups)

    def _pick_aruco(self, view: str, cam_key: str, pose_tag: Optional[str]) -> Dict:
        # pose_tag 우선 → 기본(view_cam) 폴백
        if pose_tag:
            k1 = f"{pose_tag}_{view}_{cam_key}"
            if k1 in self.aruco_lookup:
                return self.aruco_lookup[k1]
        k2 = f"{view}_{cam_key}"
        if k2 in self.aruco_lookup:
            return self.aruco_lookup[k2]
        raise KeyError(f"[FR3] No ArUco for keys '{pose_tag}_{view}_{cam_key}' or '{view}_{cam_key}'")

    def __getitem__(self, idx):
        group = self.groups[idx]
        try:
            joint_angles_rad = np.asarray(group['joint_angles'], dtype=np.float64)
            gt_angles = torch.tensor(joint_angles_rad, dtype=torch.float32)

            image_dict, gt_heatmaps_dict = {}, {}
            pose_tag = group.get('pose_tag')

            # ★ 뷰 정렬 & 상한 적용
            #   - 파일명에서 serial, cam_type을 먼저 추출해 메타를 붙이고 정렬
            entries = []
            for view_data in group['views']:
                raw_path = view_data['image_path']
                path = _resolve_path(raw_path)
                serial, cam_type = _parse_filename_for_view(path)
                serial_rank = self._SERIAL_ORDER.index(serial) if serial in self._SERIAL_ORDER else 999
                cam_rank    = self._CAM_ORDER.get(cam_type, 9)
                entries.append((serial_rank, cam_rank, serial, cam_type, raw_path, path))

            # 일관 정렬 (serial 우선, 다음 cam_type)
            entries.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

            # 상한 적용: MAX_VIEWS_PER_GROUP 개수만 사용
            if len(entries) > self.max_views:
                # 필요 시 경고 한 줄 (학습 로깅에 도움)
                # print(f"[FR3] Truncating views {len(entries)} -> {self.max_views} at idx={idx}")
                entries = entries[:self.max_views]

            for _, _, serial, cam_type, raw_path, path in entries:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"[FR3] Image not found: {path}")

                view = self.serial_to_view.get(serial, f"view?")
                cam_key = f"{cam_type}cam"
                view_key = f"{serial}_{cam_type}"

                calib_key = f"{view}_{serial}_{cam_key}"
                if calib_key not in self.calib_lookup:
                    raise KeyError(f"[FR3] Missing calib: {calib_key}")
                calib = self.calib_lookup[calib_key]
                cam_mat = np.array(calib["camera_matrix"], dtype=np.float64)
                dist_coeff = np.array(calib["distortion_coeffs"], dtype=np.float64).reshape(-1,1)

                pose_from_path = 'pose1' if ('pose1' in path or 'pose1' in raw_path) else \
                                 ('pose2' if ('pose2' in path or 'pose2' in raw_path) else None)
                aruco = self._pick_aruco(view, cam_key, pose_tag or pose_from_path)

                img_bgr = cv2.imread(path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                K_new, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeff, (w, h), alpha=0)
                undist = cv2.undistort(img_rgb, cam_mat, dist_coeff, None, K_new)

                joints_3d = angle_to_joint_coordinate(joint_angles_rad[0:self.spec.num_angles], view)
                kpts_2d_full = project_3d_to_2d(joints_3d, aruco, K_new, dist=None).astype(np.float32)

                IN = self.input_size
                resized = cv2.resize(undist, (IN, IN), interpolation=cv2.INTER_LINEAR)
                kpts_on_IN = _scale_points(kpts_2d, from_size=(w, h), to_size=(IN, IN))

                # ---- 스펙 개수(=8)로 슬라이스
                J = int(self.num_joints)  # 보통 8
                kpts_2d = kpts_2d_full[:J, :]  # ★ 마지막(EE 등) 제외

                Ht, Wt = self.heatmap_size
                kpts_hm = _scale_points(kpts_on_IN, from_size=(IN, IN), to_size=(Wt, Ht))

                J = self.spec.num_joints
                heatmaps_np = np.zeros((J, Ht, Wt), dtype=np.float32)
                for j in range(J):
                    heatmaps_np[j] = create_gt_heatmap(kpts_hm[j], (Ht, Wt), self.sigma)

                img_pil = Image.fromarray(resized)
                img_tensor = self.transform(img_pil) if self.transform else \
                             torch.from_numpy(resized).permute(2,0,1).contiguous().float()/255.0

                image_dict[view_key] = img_tensor
                gt_heatmaps_dict[view_key] = torch.from_numpy(heatmaps_np)

            return image_dict, gt_heatmaps_dict, gt_angles

        except Exception as e:
            print(f"[FR3][idx={idx}] Error: {e}")
            return None, None, None
