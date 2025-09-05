import os
import glob
import json
import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import pandas as pd
import matplotlib.pyplot as plt
import configparser

# =================================================================================
# 통합 설정 (변경 없음)
# =================================================================================
# --- 1단계 설정: Raw 데이터 경로 ---
RAW_DATA_DIRS = [
    "./Meca_insertion/Meca_ArUco/ArUco_cap1_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap2_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap3_250514"
]
# --- 2단계 설정: Pose 재계산용 ---
MARKER_REAL_SIZE_M = 0.05
# --- 3단계 설정: 시각화 및 최종 오프셋 ---
CALIB_DIR = "./Meca_insertion/Meca_calib_cam_from_conf"
STEREO_CONF_DIR = "./All_camera_conf"
IMAGE_DIR = "./Meca_insertion/Meca_ArUco/ArUco_cap1_250514"
FINAL_SUMMARY_OUTPUT_PATH = "./Meca_insertion/Meca_insertion_aruco_pose_summary.json"

camera_serials = {"front":41182735, "right":49429257, "left":44377151, "top":49045152}
views = ['front', 'left', 'right', 'top']
cams = ['leftcam', 'rightcam']

marker_offsets = {
    "front": { "1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065]) },
    "left": { "3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0.000, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065]) },
    "right": { "1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0.000, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "7": np.array([0.000, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065]) },
    "top": { "1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065]) },
}

# =================================================================================
# 헬퍼 함수
# =================================================================================
def load_stereo_config(serial):
    conf_path = os.path.join(STEREO_CONF_DIR, f"SN{serial}.conf")
    if not os.path.exists(conf_path):
        # ❌ 파일 없음 메시지 추가
        print(f"    ❌ [STEREO] Config file not found: {conf_path}")
        return None

    config = configparser.ConfigParser()
    config.read(conf_path, encoding='utf-8-sig')
    
    try:
        params = {
            'baseline': config.getfloat('STEREO', 'Baseline'),
            'ty': config.getfloat('STEREO', 'TY'),
            'tz': config.getfloat('STEREO', 'TZ'),
            'rx': config.getfloat('STEREO', 'RX_FHD1200'),
            'ry': config.getfloat('STEREO', 'CV_FHD1200'),
            'rz': config.getfloat('STEREO', 'RZ_FHD1200')
        }
        # ✅ 성공 메시지 추가
        print(f"    ✅ [STEREO] Successfully loaded stereo config for SN {serial}.")
        return params
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        # ❌ 오류 메시지 추가
        print(f"    ❌ [STEREO] Error reading config file {conf_path}: {e}")
        return None

# (다른 헬퍼 함수들은 변경 없음)
def parse_filename(filename):
    parts = filename.split('_')
    view, cam = parts[0], parts[2]
    return view, cam

def average_quaternion(quaternions):
    if len(quaternions) == 0: return np.array([0,0,0,1])
    return R.from_quat(quaternions).mean().as_quat()

def average_position(positions):
    if len(positions) == 0: return np.array([0,0,0])
    return np.mean(positions, axis=0)

def remove_outliers(positions, quaternions, pos_thresh=0.001, rot_thresh_deg=3):
    if len(positions) < 2: return positions, quaternions, np.ones(len(positions), dtype=bool)
    avg_pos, avg_quat = average_position(positions), average_quaternion(quaternions)
    pos_dists = np.linalg.norm(positions - avg_pos, axis=1)
    pos_mask = pos_dists < pos_thresh
    avg_rot = R.from_quat(avg_quat)
    angular_distances = np.array([np.rad2deg((avg_rot.inv() * R.from_quat(q)).magnitude()) for q in quaternions])
    rot_mask = angular_distances < rot_thresh_deg
    valid_mask = pos_mask & rot_mask
    return positions[valid_mask], quaternions[valid_mask], valid_mask

# 헬퍼 함수 섹션에 추가할 시각화 함수
def save_visualization_image(view, cam, image_path, K, dist, poses, marker_offsets, final_pose, output_dir):
    """
    계산된 자세 정보를 이미지에 시각화하여 지정된 경로에 파일로 저장합니다.

    Args:
        view (str): 'front', 'left' 등 현재 뷰 이름
        cam (str): 'leftcam' 또는 'rightcam'
        image_path (str): 시각화의 바탕이 될 원본 이미지 경로
        K (np.array): 카메라 매트릭스
        dist (np.array): 왜곡 계수
        poses (dict): Stage 2에서 계산된 개별 마커들의 자세 정보
        marker_offsets (dict): 마커 오프셋 설정값
        final_pose (dict): 최종 계산된 평균 객체 자세 {'rvec': ..., 'tvec': ...}
        output_dir (str): 이미지를 저장할 폴더 경로
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. 이미지 로드 및 왜곡 보정
    img = cv2.imread(image_path)
    undistorted_img = cv2.undistort(img, K, dist, None, K)

    # 2. 개별 마커들의 자세 시각화 (축 및 ID)
    for mid, offset in marker_offsets.get(view, {}).items():
        if mid in poses and mid != "8":
            p = poses[mid]
            tvec = np.array(list(p["position_m"].values()))
            quat = np.array(list(p["rotation_quat"].values()))
            rvec = R.from_quat(quat).as_rotvec()
            
            # 각 마커의 좌표축 그리기
            cv2.drawFrameAxes(undistorted_img, K, None, rvec, tvec.reshape(3, 1), 0.05)
            
            # 마커 ID 텍스트 표시
            marker_pos_2d, _ = cv2.projectPoints(tvec.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
            cv2.putText(undistorted_img, f"ID:{mid}", tuple(marker_pos_2d.ravel().astype(int)), font, 0.6, (255, 255, 0), 2)

    # 3. 최종 평균 자세 시각화 (축 및 중심점)
    mean_rvec = final_pose['rvec']
    mean_tvec = final_pose['tvec']
    
    # 최종 객체의 좌표축 그리기 (더 굵게)
    cv2.drawFrameAxes(undistorted_img, K, None, mean_rvec, mean_tvec, 0.1, thickness=4)
    
    # 최종 객체의 중심점 마커 그리기
    mean_proj, _ = cv2.projectPoints(mean_tvec.reshape(1, 3), np.zeros(3), np.zeros(3), K, None)
    xm, ym = mean_proj.ravel().astype(int)
    cv2.drawMarker(undistorted_img, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # 4. Matplotlib을 사용하여 이미지 파일로 저장
    plt.figure(figsize=(12, 9))
    # OpenCV의 BGR 색상 순서를 Matplotlib의 RGB 순서로 변환
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Final Mean Pose ({view.upper()}-{cam})")
    plt.axis('off')
    plt.tight_layout()
    
    # 파일 저장 및 리소스 정리
    output_path = os.path.join(output_dir, f"{view}_{cam}_pose_visualization.png")
    plt.savefig(output_path)
    plt.close() # 메모리 누수 방지를 위해 창을 닫아줍니다.
    print(f"✅ Visualization saved to: {output_path}")

# =================================================================================
# 메인 로직
# =================================================================================

# --- 1단계: 이상치 제거 및 평균 계산 ---
print("--- STAGE 1: Averaging Raw Data and Removing Outliers ---")
raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
total_files_found = 0
for base_dir in RAW_DATA_DIRS:
    for fname in os.listdir(base_dir):
        if not fname.endswith('.json'): continue
        total_files_found += 1 # ✅ 파일 카운트
        view, cam = parse_filename(fname)
        with open(os.path.join(base_dir, fname), 'r') as f:
            content = json.load(f)
            for marker_id, marker_data in content.items():
                if "corners_pixel" in marker_data:
                    raw_data[view][cam][marker_id].append(marker_data)
# ✅ 로드된 파일 수 정보 출력
print(f"🔍 [INFO] Found and processed {total_files_found} raw JSON files.")

corrected_data_stage1 = defaultdict(lambda: defaultdict(dict))
# (Stage 1의 나머지 부분은 변경 없음)
for view in raw_data:
    for cam in raw_data[view]:
        for marker_id, entries in raw_data[view][cam].items():
            if len(entries) < 2:
                if entries: corrected_data_stage1[view][cam][marker_id] = entries[0]
                continue
            
            positions = np.array([[m['position_m'][k] for k in 'xyz'] for m in entries])
            quaternions = np.array([[m['rotation_quat'][k] for k in 'xyzw'] for m in entries])
            corners = np.array([m['corners_pixel'] for m in entries], dtype=np.float32)

            positions_filtered, quats_filtered, mask = remove_outliers(positions, quaternions)
            
            if len(positions_filtered) == 0 or len(positions_filtered) < len(entries) / 2:
                continue

            avg_pos = average_position(positions_filtered)
            avg_quat = average_quaternion(quats_filtered)
            avg_corners = np.mean(corners[mask], axis=0)
            
            corrected_data_stage1[view][cam][marker_id] = {
                "position_m": dict(zip('xyz', avg_pos)),
                "rotation_quat": dict(zip('xyzw', avg_quat)),
                "corners_pixel": avg_corners.tolist()
            }
print("--- STAGE 1 Complete ---\n")


# --- 2단계: Top-Left 기준 Pose 재계산 ---
print("--- STAGE 2: Recalculating Pose from Top-Left Corner ---")
marker_3d_points = np.array([
    [0, 0, 0], [MARKER_REAL_SIZE_M, 0, 0], [MARKER_REAL_SIZE_M, MARKER_REAL_SIZE_M, 0], [0, MARKER_REAL_SIZE_M, 0]
], dtype=np.float32)
recalculated_data_stage2 = defaultdict(lambda: defaultdict(dict))

for view in corrected_data_stage1:
    for cam in corrected_data_stage1[view]:
        serial = camera_serials.get(view)
        if not serial: continue
        calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
        
        # ✅ 캘리브레이션 파일 존재 여부 확인 및 메시지 출력
        if not os.path.exists(calib_path):
            print(f"  ❌ [CALIB] Calibration file not found for {view}_{cam}. Skipping.")
            continue
        print(f"  ✅ [CALIB] Found calibration for {view}_{cam}.")
            
        with open(calib_path) as f: calib_data = json.load(f)
        camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float64)

        for marker_id, data in corrected_data_stage1[view][cam].items():
            corners_2d = np.array(data["corners_pixel"], dtype=np.float32)
            try:
                ret, rvec, tvec = cv2.solvePnP(marker_3d_points, corners_2d, camera_matrix, dist_coeffs)
            except ValueError:
                ret, rvec, tvec, _ = cv2.solvePnP(marker_3d_points, corners_2d, camera_matrix, dist_coeffs, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_IPPE)

            if not ret: continue
            
            rvec, tvec = cv2.solvePnPRefineLM(marker_3d_points, corners_2d, camera_matrix, dist_coeffs, rvec, tvec)
            new_position = {"x": tvec[0][0], "y": tvec[1][0], "z": tvec[2][0]}
            quat = R.from_rotvec(rvec.flatten()).as_quat()
            new_rotation = dict(zip('xyzw', quat))
            
            recalculated_data_stage2[view][cam][marker_id] = {
                "position_m": new_position, "rotation_quat": new_rotation, "corners_pixel": data["corners_pixel"]
            }
print("--- STAGE 2 Complete ---\n")


# --- 3단계: 최종 오프셋 적용, 시각화 및 요약 ---
print("--- STAGE 3: Applying Final Offsets, Summarizing, and Visualizing ---")
final_summary = []
RESULTS_IMAGE_DIR = "./Meca_insertion/Meca_calib_results_images" # 이미지 저장 폴더 설정

# 스크립트 시작 시 결과 폴더가 없으면 생성
os.makedirs(RESULTS_IMAGE_DIR, exist_ok=True)

for view in views:
    print(f"\nProcessing View: {view.upper()}")
    left_cam_final_pose = None
    
    # --- 1. LEFTCAM 처리 ---
    cam = 'leftcam'
    poses = recalculated_data_stage2.get(view, {}).get(cam)
    if poses:
        serial = camera_serials.get(view)
        calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
        img_files = glob.glob(os.path.join(IMAGE_DIR, f"{view}_*_{cam}_*.png"))

        if os.path.exists(calib_path) and img_files:
            print(f"  [LEFTCAM] Processing... Found calib and {len(img_files)} image file(s).")
            with open(calib_path) as f: calib = json.load(f)
            K, dist = np.array(calib["camera_matrix"]), np.array(calib["distortion_coeffs"])
            
            # 계산 로직 (기존과 동일)
            offset_applied_positions, all_quaternions = [], []
            for mid, offset in marker_offsets.get(view, {}).items():
                if mid in poses:
                    p = poses[mid]
                    tvec = np.array(list(p["position_m"].values()))
                    quat = np.array(list(p["rotation_quat"].values()))
                    Rm = R.from_quat(quat).as_matrix()
                    offset_applied_positions.append(tvec + Rm @ offset)
                    all_quaternions.append(quat)

            if offset_applied_positions:
                mean_offset_pos = np.mean(offset_applied_positions, axis=0)
                std_offset_pos = np.std(offset_applied_positions, axis=0)
                mean_quat = average_quaternion(np.array(all_quaternions))
                mean_rvec = R.from_quat(mean_quat).as_rotvec()
                
                left_cam_final_pose = {'tvec': mean_offset_pos, 'rvec': mean_rvec}

                # 요약 데이터 추가
                mean_proj, _ = cv2.projectPoints(mean_offset_pos, np.zeros(3), np.zeros(3), K, None)
                deg_rvec = np.rad2deg(mean_rvec.flatten())
                final_summary.append({
                    "view": view, "cam": cam, "tvec_x": mean_offset_pos[0], "tvec_y": mean_offset_pos[1], "tvec_z": mean_offset_pos[2],
                    "std_x": std_offset_pos[0], "std_y": std_offset_pos[1], "std_z": std_offset_pos[2],
                    "proj_x": mean_proj.ravel().astype(int)[0], "proj_y": mean_proj.ravel().astype(int)[1],
                    "rvec_x": deg_rvec[0], "rvec_y": deg_rvec[1], "rvec_z": deg_rvec[2]
                })
                print(f"  ➡️  {cam}: Pose calculated from {len(all_quaternions)} markers.")

                # ▼▼▼ [수정] LEFTCAM 시각화 함수 호출 위치 ▼▼▼
                save_visualization_image(
                    view=view, cam=cam, image_path=img_files[0], K=K, dist=dist,
                    poses=poses, marker_offsets=marker_offsets, final_pose=left_cam_final_pose,
                    output_dir=RESULTS_IMAGE_DIR
                )
    else:
        print(f"  ❌ [LEFTCAM] Skipped: No marker data found in Stage 2 results.")
        
    RIGHT_CAM_CORRECTION_OFFSET = np.array([-0.025, 0, 0]) 
    
    # --- 2. RIGHTCAM 처리 ---
    cam = 'rightcam'
    if left_cam_final_pose:
        serial = camera_serials.get(view)
        stereo_params = load_stereo_config(serial)
        
        if stereo_params:
            # --- 변수 이름 변경으로 가독성 향상 ---
            
            # 1. leftcam이 바라본 마커의 자세 (Marker -> LeftCam)
            rvec_marker_in_left, tvec_marker_in_left = left_cam_final_pose['rvec'], left_cam_final_pose['tvec']
            R_marker_in_left, _ = cv2.Rodrigues(rvec_marker_in_left)
            T_marker_to_left = np.eye(4)
            T_marker_to_left[:3, :3], T_marker_to_left[:3, 3] = R_marker_in_left, tvec_marker_in_left

            # 2. ZED 설정: leftcam 기준 rightcam의 위치 (RightCam -> LeftCam)
            t_right_in_left = np.array([p/1000.0 for p in [stereo_params['baseline'], stereo_params['ty'], stereo_params['tz']]])
            R_right_in_left = R.from_euler('zyx', [stereo_params['rz'], stereo_params['ry'], stereo_params['rx']]).as_matrix()
            T_right_to_left = np.eye(4)
            T_right_to_left[:3, :3], T_right_to_left[:3, 3] = R_right_in_left, t_right_in_left
            
            # 3. 역변환: rightcam 기준 leftcam의 위치 (LeftCam -> RightCam)
            T_left_to_right = np.linalg.inv(T_right_to_left)

            # 4. 최종 계산: rightcam이 바라본 마커의 자세 (Marker -> RightCam)
            T_marker_to_right = T_left_to_right @ T_marker_to_left
            
            # 최종 결과 추출
            R_marker_in_right, tvec_right = T_marker_to_right[:3, :3], T_marker_to_right[:3, 3]
            rvec_right, _ = cv2.Rodrigues(R_marker_in_right)
            
            print(f"   - Applying manual correction offset to rightcam: {RIGHT_CAM_CORRECTION_OFFSET.flatten()} m")
            tvec_right += RIGHT_CAM_CORRECTION_OFFSET
            
            # 요약 데이터 추가
            deg_rvec_right = np.rad2deg(rvec_right.flatten())
            # (나머지 요약 데이터 추가 로직은 기존과 동일)
            final_summary.append({
                "view": view, "cam": cam, "tvec_x": tvec_right[0], "tvec_y": tvec_right[1], "tvec_z": tvec_right[2],
                "std_x": 0, "std_y": 0, "std_z": 0, "proj_x": -1, "proj_y": -1,
                "rvec_x": deg_rvec_right[0], "rvec_y": deg_rvec_right[1], "rvec_z": deg_rvec_right[2]
            })
            print(f"  ➡️  {cam}: Pose calculated from stereo transformation.")

            # ▼▼▼ [수정] RIGHTCAM 시각화 함수 호출 위치 ▼▼▼
            calib_path_right = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            img_files_right = glob.glob(os.path.join(IMAGE_DIR, f"{view}_*_{cam}_*.png"))

            if os.path.exists(calib_path_right) and img_files_right:
                with open(calib_path_right) as f: calib = json.load(f)
                K_right, dist_right = np.array(calib["camera_matrix"]), np.array(calib["distortion_coeffs"])
                
                right_cam_final_pose = {'tvec': tvec_right, 'rvec': rvec_right}
                # RightCam은 변환된 최종 포즈만 시각화 (개별 마커는 LeftCam 기준이므로 그리지 않음)
                save_visualization_image(
                    view=view, cam=cam, image_path=img_files_right[0], K=K_right, dist=dist_right,
                    poses=recalculated_data_stage2.get(view, {}).get(cam, {}), # RightCam에서 감지된 마커도 함께 표시
                    marker_offsets=marker_offsets, final_pose=right_cam_final_pose,
                    output_dir=RESULTS_IMAGE_DIR
                )
    else:
        print(f"  ❌ [RIGHTCAM] Skipped: No valid pose from leftcam to transform.")
# --- 최종 요약 파일 저장 ---
if final_summary:
    df = pd.DataFrame(final_summary)
    df.to_json(FINAL_SUMMARY_OUTPUT_PATH, orient="records", indent=4)
    print(f"\n--- ✅ All stages complete. Final summary saved to: {FINAL_SUMMARY_OUTPUT_PATH} ---")
    print(df)
else:
    print("\n--- ❌ Processing finished, but no data was generated for the final summary. ---")