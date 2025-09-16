# dataset.py
import os, glob, json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# 통합 utils 모듈 사용
from utils import (
    create_gt_heatmap,
    angle_to_joint_coordinate,    # robot='fr3' | 'fr5' | 'meca500'
    project_3d_to_2d_aruco,       # ArUco dict 안전 처리
    project_3d_to_2d_by_robot,
    perform_grouping,             # ★ 그룹핑 라우팅 (fr3/fr5/meca500)
)

# ---------------------------
# 공통 유틸
# ---------------------------
def _ensure_chw_float(img_np_uint8: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (3,H,W) float32 [0,1]"""
    return torch.from_numpy(img_np_uint8).permute(2, 0, 1).contiguous().float() / 255.0

def _scale_points(points_xy: np.ndarray,
                  from_size: Tuple[int,int],
                  to_size: Tuple[int,int]) -> np.ndarray:
    """points(x,y)을 from_size(W,H) -> to_size(W,H)로 스케일"""
    Wf, Hf = from_size
    Wt, Ht = to_size
    out = np.empty_like(points_xy, dtype=np.float32)
    out[:, 0] = points_xy[:, 0] * (Wt / float(Wf))
    out[:, 1] = points_xy[:, 1] * (Ht / float(Hf))
    return out

def _undistort_with_newK(img_rgb: np.ndarray, K: np.ndarray, dist: np.ndarray):
    """getOptimalNewCameraMatrix(alpha=0)로 newK를 얻고 undistort."""
    h, w = img_rgb.shape[:2]
    K_new, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undist = cv2.undistort(img_rgb, K, dist, None, K_new)
    return undist, K_new

def _parse_filename_for_view(filename: str):
    """
    예: zed_38007749_left_*.jpg -> serial='38007749', cam_type='left'
    모든 데이터셋에서 동일 패턴을 가정.
    """
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename pattern: {filename}")
    serial, cam_type = parts[1], parts[2]
    return serial, cam_type

# ---------------------------
# 데이터셋 스펙 정의
# ---------------------------
@dataclass
class DatasetSpec:
    name: str
    # dataset 루트(프로젝트 기준 상대경로)
    dataset_subdir: str
    # 아루코 요약 파일 목록(상대경로). 여러개면 전부 로드 후 lookup 통합
    aruco_summary_files: List[str]
    # 캘리브 json 저장 폴더(상대경로, glob "*.json")
    calib_dir: str
    # 파일 경로 내에 포함되는 기준 세그먼트(ex. 'Fr5', 'franka_research3', 'Meca_insertion')
    path_anchor: str
    # filename serial -> view 이름 매핑
    serial_to_view: Dict[str, str]
    # FK 기본 각도 단위
    default_fk_unit: str  # 'deg' or 'rad'

# 스펙 테이블
SPECS: Dict[str, DatasetSpec] = {
    "fr5": DatasetSpec(
        name="fr5",
        dataset_subdir=os.path.join("dataset", "Fr5"),
        aruco_summary_files=["Fr5_aruco_pose_summary.json"],
        calib_dir="Fr5_calib_cam_from_conf",
        path_anchor="Fr5",
        serial_to_view={"38007749": "left", "34850673": "right", "30779426": "top"},
        default_fk_unit="deg",
    ),
    "fr3": DatasetSpec(
        name="fr3",
        dataset_subdir=os.path.join("dataset", "franka_research3"),
        aruco_summary_files=["pose1_aruco_pose_summary.json", "pose2_aruco_pose_summary.json"],
        calib_dir="franka_research3_calib_cam_from_conf",
        path_anchor="franka_research3",
        serial_to_view={"41182735": "view1", "49429257": "view2", "44377151": "view3", "49045152": "view4"},
        default_fk_unit="rad",
    ),
    "meca500": DatasetSpec(
        name="meca500",
        dataset_subdir=os.path.join("dataset", "Meca_insertion"),
        aruco_summary_files=["Meca_insertion_aruco_pose_summary.json"],
        calib_dir="Meca_calib_cam_from_conf",
        path_anchor="Meca_insertion",
        # 동일 카메라 환경이라면 아래 시리얼-뷰 맵 사용
        serial_to_view={"41182735": "front", "49429257": "right", "44377151": "left", "49045152": "top"},
        default_fk_unit="deg",
    ),
}

# ---------------------------
# 경로 유틸 (데이터셋별 anchor 인식)
# ---------------------------
def _resolve_path(p: str, abs_dataset_root: str, anchor: str) -> str:
    """
    데이터 경로 문자열을 실제 절대 경로로 매핑.
    - 절대경로면 그대로
    - 경로 내에 anchor 세그먼트(예: 'Fr5', 'franka_research3', 'Meca_insertion')가 포함돼 있으면,
      anchor 이후 경로만 떼서 abs_dataset_root 뒤에 붙임.
    - 'dataset/<anchor>/...' 시작도 처리
    - 그 외엔 abs_dataset_root/<p_norm>
    """
    if not p:
        return p
    if os.path.isabs(p):
        return os.path.abspath(p)

    p_norm = os.path.normpath(p)
    parts = p_norm.split(os.sep)

    if anchor in parts:
        idx = parts.index(anchor)
        tail = parts[idx + 1 :]
        return os.path.abspath(os.path.join(abs_dataset_root, *tail))

    if parts and parts[0] == "dataset":
        tail = parts[1:]
        if tail and tail[0] == anchor:
            return os.path.abspath(os.path.join(abs_dataset_root, *tail[1:]))
        return os.path.abspath(os.path.join(abs_dataset_root, *tail))

    return os.path.abspath(os.path.join(abs_dataset_root, p_norm))

# ---------------------------
# 통합 Dataset
# ---------------------------
class UnifiedRobotPoseDataset(Dataset):
    """
    통합 데이터셋:
      - dataset_type: 'fr5' | 'fr3' | 'meca500'
      - items: 멀티뷰 그룹 또는 싱글뷰 페어 리스트
        * 멀티뷰: {'views': [{'image_path': str}, ...], 'joint_angles': [...], (optional) 'pose_tag': 'pose1'|'pose2'}
        * 싱글뷰: {'image_path': str, 'joint_angles': [...]}
    반환:
      image_dict:   {view_key: (3,IN,IN) float}
      heatmaps_dict:{view_key: (J,Ht,Wt) float}
      gt_angles:    (N_angles,) float (FK 단위 그대로)
    """
    def __init__(self,
                 dataset_type: str,
                 items: List[Dict],
                 transform=None,
                 heatmap_size: Tuple[int,int] = (128,128),
                 sigma: float = 5.0,
                 input_size: int = 224,
                 robot: Optional[str] = None,        # FK 대상 로봇 (None이면 dataset_type과 동일)
                 robot_fk_unit: Optional[str] = None # None이면 스펙 default 사용
                 ):
        assert dataset_type in SPECS, f"dataset_type must be one of {list(SPECS.keys())}"
        self.spec = SPECS[dataset_type]
        self.items = items
        self.transform = transform
        self.heatmap_size = tuple(heatmap_size)
        self.sigma = float(sigma)
        self.input_size = int(input_size)
        self.robot = (robot or dataset_type).lower()
        self.robot_fk_unit = (robot_fk_unit or self.spec.default_fk_unit)
        assert self.robot_fk_unit in ("deg", "rad")

        # 프로젝트 루트 기준 dataset 절대경로
        _cur_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.abspath(os.path.join(_cur_dir, "../.."))
        self.dataset_root = os.path.abspath(os.path.join(_project_root, self.spec.dataset_subdir))

        print(f"Loading metadata for dataset={self.spec.name} ...")
        self._build_lookups()
        print("✅ Metadata loaded.")

    def _build_lookups(self):
        # --- ArUco lookup
        self.aruco_lookup: Dict[str, Dict] = {}
        for rel in self.spec.aruco_summary_files:
            abs_path = os.path.join(self.dataset_root, rel)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"ArUco summary not found: {abs_path}")
            with open(abs_path, "r") as f:
                data = json.load(f)
            for item in data:
                view = item.get("view")
                cam = item.get("cam")
                pose_tag = item.get("pose", "") or item.get("pose_tag", "")
                if not view or not cam:
                    continue
                # 키 생성 규칙:
                # - pose_tag가 있으면 "pose_view_cam"
                # - 항상 "view_cam"도 등록 (우선도는 호출부에서 pose_tag 우선)
                k1 = f"{pose_tag}_{view}_{cam}".lstrip("_")
                k2 = f"{view}_{cam}"
                self.aruco_lookup[k1] = item
                self.aruco_lookup[k2] = item

        # --- Calibration lookup
        self.calib_lookup: Dict[str, Dict] = {}
        calib_glob = os.path.join(self.dataset_root, self.spec.calib_dir, "*.json")
        calib_files = glob.glob(calib_glob)
        if not calib_files:
            raise FileNotFoundError(f"No calibration files under: {os.path.dirname(calib_glob)}")
        for path in calib_files:
            key = os.path.basename(path).replace("_calib.json", "")
            with open(path, "r") as f:
                self.calib_lookup[key] = json.load(f)

        # --- Serial → view
        self.serial_to_view = dict(self.spec.serial_to_view)

    def __len__(self):
        return len(self.items)

    def _pick_aruco(self, view: str, cam_key: str, pose_tag: Optional[str]) -> Dict:
        """
        pose_tag가 있으면 'pose_view_cam'를 우선, 없으면 'view_cam'
        """
        if pose_tag:
            k1 = f"{pose_tag}_{view}_{cam_key}"
            if k1 in self.aruco_lookup:
                return self.aruco_lookup[k1]
        k2 = f"{view}_{cam_key}"
        if k2 in self.aruco_lookup:
            return self.aruco_lookup[k2]
        raise KeyError(f"No ArUco pose for keys '{pose_tag}_{view}_{cam_key}' or '{view}_{cam_key}'")

    def _resolve_one_path(self, p: str) -> str:
        return _resolve_path(p, self.dataset_root, self.spec.path_anchor)

    def _handle_one_view(self,
                         img_path: str,
                         joint_angles: np.ndarray,
                         pose_tag: Optional[str]) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        단일 뷰 처리:
        - 이미지를 undistort(K_new) 후 224×224 워핑
        - FK(robot) → 3D → 2D(K_new, dist=None)
        - heatmap 생성
        return: view_key, img_tensor, heatmaps_tensor
        """
        # 파일명에서 serial, cam_type
        serial, cam_type = _parse_filename_for_view(img_path)
        view = self.serial_to_view.get(serial, cam_type)  # 매핑 없으면 cam_type fallback
        cam_key = f"{cam_type}cam"                        # leftcam/rightcam/topcam/...

        # Calibration
        calib_key = f"{view}_{serial}_{cam_key}"
        if calib_key not in self.calib_lookup:
            raise KeyError(f"Missing calib for key={calib_key}")
        calib = self.calib_lookup[calib_key]
        K = np.array(calib["camera_matrix"], dtype=np.float64)
        dist = np.array(calib["distortion_coeffs"], dtype=np.float64).reshape(-1, 1)

        # ArUco
        aruco = self._pick_aruco(view, cam_key, pose_tag)

        # 이미지 로드 + undistort
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        undist, K_new = _undistort_with_newK(img_rgb, K, dist)
        h, w = undist.shape[:2]

        # FK → 3D → 2D
        joints_3d = angle_to_joint_coordinate(
            joint_angles, robot=self.robot, selected_view=view, input_unit=self.robot_fk_unit
        )
        kpts_2d = project_3d_to_2d_by_robot(
            points_3d=joints_3d,          # (J,3)
            robot=self.robot,             # 'fr5' | 'fr3' | 'meca500'
            aruco_result=aruco,           # 위에서 선택한 뷰/캠의 아루코 요약
            K=K_new,                      # undistort로 얻은 새 내참
            dist=None                     # ✅ undistort 했으니 왜곡은 None
        ).astype(np.float32)
        
        # 224×224 워핑
        IN = self.input_size
        resized = cv2.resize(undist, (IN, IN), interpolation=cv2.INTER_LINEAR)
        # 키포인트도 동일 워핑 (비율보존X)
        kpts_on_IN = _scale_points(kpts_2d, from_size=(w, h), to_size=(IN, IN))
        # Heatmap 좌표로 스케일 (IN -> (Wt,Ht))
        Ht, Wt = self.heatmap_size
        kpts_hm = _scale_points(kpts_on_IN, from_size=(IN, IN), to_size=(Wt, Ht))

        # Heatmap 생성
        num_joints = joints_3d.shape[0]
        heatmaps = np.zeros((num_joints, Ht, Wt), dtype=np.float32)
        for j in range(num_joints):
            heatmaps[j] = create_gt_heatmap(kpts_hm[j], (Ht, Wt), self.sigma)

        # 이미지 텐서
        img_pil = Image.fromarray(resized)
        img_tensor = self.transform(img_pil) if self.transform else _ensure_chw_float(resized)

        view_key = f"{serial}_{cam_type}"
        return view_key, img_tensor, torch.from_numpy(heatmaps)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            # --- joint angles (CSV의 단위를 그대로 보존; FK는 self.robot_fk_unit로 처리)
            joint_angles = np.asarray(item["joint_angles"], dtype=np.float64)
            gt_angles = torch.tensor(joint_angles, dtype=torch.float32)

            image_dict, heatmaps_dict = {}, {}

            # --- 멀티뷰 그룹 vs 싱글뷰 페어 자동 판별
            if "views" in item:
                # 멀티뷰
                pose_tag = item.get("pose_tag")  # fr3는 pose1/pose2가 필요할 수 있음
                for v in item["views"]:
                    raw_path = v["image_path"]
                    img_path = self._resolve_one_path(raw_path)
                    k, img_t, hm_t = self._handle_one_view(img_path, joint_angles, pose_tag)
                    image_dict[k] = img_t
                    heatmaps_dict[k] = hm_t
            else:
                # 싱글뷰
                raw_path = item["image_path"]
                img_path = self._resolve_one_path(raw_path)
                pose_tag = item.get("pose_tag")
                k, img_t, hm_t = self._handle_one_view(img_path, joint_angles, pose_tag)
                image_dict[k] = img_t
                heatmaps_dict[k] = hm_t

            return image_dict, heatmaps_dict, gt_angles

        except Exception as e:
            print(f"[UnifiedDataset:{self.spec.name}] Error at idx={idx}: {e}")
            return None, None, None

# ---------------------------
# DataLoader용 collate_fn
# ---------------------------
def collate_skip_none(batch):
    """
    __getitem__이 (None,None,None)을 반환한 샘플을 건너뛰는 collate_fn.
    """
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return None
    images_list, heatmaps_list, angles_list = zip(*batch)
    return images_list, heatmaps_list, torch.stack(angles_list, dim=0)

# === 아래 코드를 dataset.py 맨 아래에 추가하세요 ===
import pandas as pd

# 분산 학습일 수 있으므로 안전 import
try:
    import torch.distributed as dist
    _HAS_DIST = True
except Exception:
    _HAS_DIST = False

# ---------------------------
# 경로 조합 유틸 (CSV도 SPECS 기반으로 조합)
# ---------------------------
def _get_project_root():
    _cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(_cur_dir, "../.."))

def join_dataset_path(dataset_type: str, *parts):
    assert dataset_type in SPECS, f"dataset_type must be one of {list(SPECS.keys())}"
    project_root = _get_project_root()
    return os.path.abspath(os.path.join(project_root, SPECS[dataset_type].dataset_subdir, *parts))

# ---------------------------
# Grid-search for TIME_TOLERANCE
# ---------------------------
def grid_search_time_tolerance(df: pd.DataFrame,
                               dataset_type: str,
                               max_views: int,
                               candidates=None,
                               rank: int = 0):
    """
    df + dataset_type 에 맞춰 perform_grouping 호출하면서
    TIME_TOLERANCE를 그리드서치. (full-groups 최대 기준)
    return: best_tol, max_full_groups
    """
    if candidates is None:
        candidates = np.round(np.arange(0.05, 0.101, 0.01), 2)  # [0.05 .. 0.10]

    best_tol, max_full = 0.0, 0
    if rank == 0:
        print(f"\nStarting Grid Search for TIME_TOLERANCE in range: {list(candidates)}")
    for tol in candidates:
        # utils.perform_grouping은 robot 타입에 맞춰 라우팅
        groups = perform_grouping(df, tol, max_views, robot=dataset_type)
        view_counts = [len(g['views']) for g in groups]
        distribution = pd.Series(view_counts).value_counts().sort_index(ascending=False)
        if rank == 0:
            print("-" * 50)
            print(f"Testing Tolerance: {tol:.2f} seconds...")
            print(f"  -> Total groups created: {len(groups)}")
            print("  -> View count distribution:")
            print(distribution.to_string())

        # "full"의 정의는 max_views
        current_full = distribution.get(max_views, 0)
        if current_full > max_full:
            max_full = current_full
            best_tol = tol

    if rank == 0:
        print("-" * 50)
        print(f"\n🏆 Grid Search Recommendation: TIME_TOLERANCE = {best_tol} (produced {max_full} full groups)")
    return best_tol, max_full

# ---------------------------
# CSV → 그룹/페어 빌드 + (옵션) 그리드서치
# ---------------------------
def build_items_from_csv(dataset_type: str,
                         csv_filename: str,
                         max_views_per_group: int,
                         do_grid_search: bool = True,
                         final_tolerance: Optional[float] = None,
                         grid_candidates=None,
                         drop_single_view_groups: bool = True,
                         rank: int = 0):
    """
    dataset_type: 'fr3' | 'fr5' | 'meca500'
    csv_filename: 데이터셋 루트 기준 파일명 (예: 'fr3_matched_joint_angle.csv')
    return: items (멀티뷰 그룹 리스트 또는 싱글뷰 페어 리스트)
    """
    csv_path = join_dataset_path(dataset_type, csv_filename)
    if rank == 0:
        print(f"\nLoading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    # 데이터셋별 정렬 기준 컬럼
    sort_col = "robot_timestamp" if dataset_type in ("fr3", "meca500") else "joint_timestamp"
    if sort_col in df.columns:
        df.sort_values(sort_col, inplace=True, ignore_index=True)
        if rank == 0:
            print("✅ CSV file loaded and sorted successfully.")
    else:
        if rank == 0:
            print(f"⚠️ Sort column '{sort_col}' not found. Skipping sort.")

    # 분산 환경 브로드캐스트 (선택)
    if _HAS_DIST and dist.is_available() and dist.is_initialized():
        obj_list = [df if rank == 0 else None]
        dist.broadcast_object_list(obj_list, src=0)
        df = obj_list[0]

    # ---- 그룹핑 or 페어 빌드
    # MECA500 CSV가 싱글뷰(타임스탬프 없음)인 경우를 안전하게 감지:
    has_image_path = "image_path" in df.columns
    has_timestamp = ("robot_timestamp" in df.columns) or ("joint_timestamp" in df.columns)
    has_any_joint = any(c.startswith(("position_fr3_joint", "joint_")) for c in df.columns)

    if dataset_type == "meca500" and has_image_path and not has_timestamp:
        # 싱글 페어로 구성 (예: joint_1..joint_6 + image_path, timestamp 없음)
        if rank == 0:
            print("ℹ️ Detected single-view MECA500 CSV → building pairs (no grouping).")
        pairs = []
        # joint 컬럼 접두사/갯수 자동 추출
        joint_cols = sorted([c for c in df.columns if c.startswith("joint_")],
                            key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 1e9)
        for _, row in df.iterrows():
            pairs.append({
                "image_path": row["image_path"],
                "joint_angles": [row[c] for c in joint_cols],
            })
        if rank == 0:
            print(f"✅ Total {len(pairs)} pairs found.")
        return pairs

    # 그 외엔 그룹핑 (FR3/FR5 일반 케이스, 또는 timestamp가 있는 MECA500 케이스)
    if do_grid_search and final_tolerance is None:
        best_tol, _ = grid_search_time_tolerance(
            df, dataset_type, max_views=max_views_per_group,
            candidates=grid_candidates, rank=rank
        )
        final_tol = best_tol
    else:
        final_tol = final_tolerance if final_tolerance is not None else 0.07
        if rank == 0:
            print(f"\n(Grid search disabled) Using TIME_TOLERANCE = {final_tol}")

    if rank == 0:
        print(f"\nFinal TIME_TOLERANCE set to: {final_tol}")
    groups = perform_grouping(df, final_tol, max_views_per_group, robot=dataset_type)
    if rank == 0:
        print(f"Total {len(groups)} groups created before filtering.")

    if drop_single_view_groups:
        before = len(groups)
        groups = [g for g in groups if len(g["views"]) > 1]
        if rank == 0:
            print(f"ℹ️ Removed {before - len(groups)} groups with only 1 view.")
            total_images = sum(len(g["views"]) for g in groups)
            print(f"\n✅ Final Total Groups: {len(groups)}")
            print(f"✅ Final Total Images to be used: {total_images}")
            if groups:
                view_counts = [len(g["views"]) for g in groups]
                print("\n--- Final View count distribution ---")
                print(pd.Series(view_counts).value_counts().sort_index(ascending=False))

    return groups

# ---------------------------
# UnifiedDataset 바로 만들기
# ---------------------------
def build_unified_dataset_from_csv(dataset_type: str,
                                   csv_filename: str,
                                   transform=None,
                                   heatmap_size=(128,128),
                                   sigma=5.0,
                                   input_size=224,
                                   max_views_per_group=8,
                                   robot: Optional[str] = None,
                                   robot_fk_unit: Optional[str] = None,
                                   do_grid_search=True,
                                   final_tolerance: Optional[float] = None,
                                   grid_candidates=None,
                                   drop_single_view_groups=True,
                                   rank: int = 0):
    """
    한 줄로: CSV 로드 → (그룹/페어) 빌드 → UnifiedRobotPoseDataset 생성
    """
    items = build_items_from_csv(
        dataset_type=dataset_type,
        csv_filename=csv_filename,
        max_views_per_group=max_views_per_group,
        do_grid_search=do_grid_search,
        final_tolerance=final_tolerance,
        grid_candidates=grid_candidates,
        drop_single_view_groups=drop_single_view_groups,
        rank=rank,
    )
    ds = UnifiedRobotPoseDataset(
        dataset_type=dataset_type,
        items=items,
        transform=transform,
        heatmap_size=heatmap_size,
        sigma=sigma,
        input_size=input_size,
        robot=robot or dataset_type,
        robot_fk_unit=robot_fk_unit,  # None이면 각각 스펙 기본 단위 사용
    )
    return ds
