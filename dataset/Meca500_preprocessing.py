import os
import glob
import json
import cv2
import numpy as np
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

# =================================================================================
# ⚙️ 통합 설정 (Global Configuration)
# =================================================================================

# --- 입력 경로 ---
RAW_DATA_DIRS = [
    "./Meca_insertion/Meca_ArUco/ArUco_cap_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap2_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap3_250514"
]
ZED_CONF_DIR = "./All_camera_conf"
IMAGE_DIR = "./Meca_insertion/Meca_ArUco/ArUco_cap_250514"

# --- 출력 경로 ---
CALIB_DIR = "./Meca_insertion/Meca_calib_cam_from_conf"
CORRECTED_ARUCO_DIR = "./Meca_insertion/Meca_correct_ArUco" # 보정된 ArUco 데이터 저장 경로
RESULT_IMAGE_DIR = "./Meca_insertion/Meca_calib_results_images"
FINAL_SUMMARY_OUTPUT_PATH = "./Meca_insertion/aruco_final_summary.json"

# --- 파라미터 ---
MARKER_REAL_SIZE_M = 0.05  # 마커의 실제 한 변 길이 (미터)

# --- 카메라 및 뷰 설정 ---
CAMERA_SERIALS = {41182735: "front", 49429257: "right", 44377151: "left", 49045152: "top"}
# [수정] view 이름으로 serial을 찾기 위한 역방향 딕셔너리 생성
VIEW_TO_SERIAL = {v: k for k, v in CAMERA_SERIALS.items()}
VIEWS = ['front', 'left', 'right', 'top']
CAMS = ['leftcam', 'rightcam']


# --- 마커 오프셋 설정 ---
MARKER_OFFSETS = {
    "front": {"1": [-0.100, 0.125, 0.0065], "2": [-0.100, 0.025, 0.0065], "3": [0, -0.175, 0.0065], "4": [-0.100, -0.075, 0.0065], "5": [0.125, 0.025, 0.0065], "6": [0.125, 0.125, 0.0065], "7": [0, -0.075, 0.0065], "8": [0.125, -0.075, 0.0065]},
    "left": {"3": [0, -0.175, 0.0065], "4": [-0.100, -0.075, 0.0065], "5": [0.125, 0.025, 0.0065], "6": [0.125, 0.125, 0.0065], "7": [0.000, -0.075, 0.0065], "8": [0.125, -0.075, 0.0065]},
    "right": {"1": [-0.100, 0.125, 0.0065], "2": [-0.100, 0.025, 0.0065], "3": [0.000, -0.175, 0.0065], "4": [-0.100, -0.075, 0.0065], "7": [0.000, -0.075, 0.0065], "8": [0.125, -0.075, 0.0065]},
    "top": {"1": [-0.100, 0.125, 0.0065], "2": [-0.100, 0.025, 0.0065], "3": [0, -0.175, 0.0065], "4": [-0.100, -0.075, 0.0065], "5": [0.125, 0.025, 0.0065], "6": [0.125, 0.125, 0.0065], "7": [0, -0.075, 0.0065], "8": [0.125, -0.075, 0.0065]},
}
for view, markers in MARKER_OFFSETS.items():
    for mid, offset in markers.items():
        MARKER_OFFSETS[view][mid] = np.array(offset)

# =================================================================================
# 헬퍼 함수 (Helper Functions)
# =================================================================================
def parse_filename(filename):
    parts = filename.split('_'); return parts[0], parts[2]

def average_quaternion(quaternions):
    if len(quaternions) == 0: return np.array([0, 0, 0, 1])
    # Scipy의 평균 계산 기능을 사용하여 더 안정적이고 간결하게 처리
    return R.from_quat(quaternions).mean().as_quat()

def average_position(positions):
    if len(positions) == 0: return np.array([0, 0, 0])
    return np.mean(positions, axis=0)

def remove_outliers(positions, quaternions, pos_thresh=0.005, rot_thresh_deg=5.0):
    if len(positions) < 3: return positions, quaternions, np.ones(len(positions), dtype=bool)
    
    avg_pos = average_position(positions)
    avg_quat_R = R.from_quat(average_quaternion(quaternions))
    
    pos_dists = np.linalg.norm(positions - avg_pos, axis=1)
    # 위치 이상치 제거 (중위값 기준)
    median_pos_dist = np.median(pos_dists)
    pos_mask = pos_dists < median_pos_dist + pos_thresh
    
    rot_dists_deg = np.array([np.rad2deg((avg_quat_R.inv() * R.from_quat(q)).magnitude()) for q in quaternions])
    # 회전 이상치 제거 (중위값 기준)
    median_rot_dist = np.median(rot_dists_deg)
    rot_mask = rot_dists_deg < median_rot_dist + rot_thresh_deg

    valid_mask = pos_mask & rot_mask
    return positions[valid_mask], quaternions[valid_mask], valid_mask

# =================================================================================
# 📜 메인 기능 함수 (Main Function Blocks)
# =================================================================================

def extract_camera_calibration():
    """ZED 카메라의 .conf 파일에서 캘리브레이션 정보를 추출하여 JSON 파일로 저장합니다."""
    print("--- 캘리브레이션 추출 시작 ---")
    os.makedirs(CALIB_DIR, exist_ok=True)
    for serial, position in CAMERA_SERIALS.items():
        conf_path = os.path.join(ZED_CONF_DIR, f"SN{serial}.conf")
        if not os.path.exists(conf_path):
            print(f"경고: [{position}] 설정 파일 없음: {conf_path}")
            continue
        for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
            try:
                config = configparser.ConfigParser()
                # utf-8-sig로 BOM(Byte Order Mark) 문제 해결
                with open(conf_path, "r", encoding="utf-8-sig") as f: config.read_file(f)
                
                # ZED X One 카메라는 FHD1200, ZED 2i 카메라는 FHD 해상도 사용 가능
                section = f"{side.upper()}_CAM_FHD1200"
                
                cam = config[section]
                data = {
                    "camera_matrix": [[float(cam["fx"]), 0.0, float(cam["cx"])], [0.0, float(cam["fy"]), float(cam["cy"])], [0.0, 0.0, 1.0]],
                    "distortion_coeffs": [float(cam[k]) for k in ["k1", "k2", "p1", "p2", "k3"]]
                }
                filename = f"{position}_{serial}_{side_name}_calib.json"
                with open(os.path.join(CALIB_DIR, filename), "w") as f: json.dump(data, f, indent=4)
                print(f"[{position}/{side_name}] 캘리브레이션 저장 완료.")
            except Exception as e:
                print(f"오류: [{position}] {side_name} 처리 중 - {e}")
    print("--- 캘리브레이션 추출 완료 ---\n")

def stage1_average_raw_data():
    """1단계: 여러 Raw 데이터에서 이상치를 제거하고 평균 포즈와 코너를 계산하여 파일로 저장합니다."""
    print("--- STAGE 1: 원본 데이터 평균 계산 및 이상치 제거 시작 ---")
    # [수정] 요청하신대로 출력 경로 생성 로직 추가
    os.makedirs(CORRECTED_ARUCO_DIR, exist_ok=True)
    
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for base_dir in RAW_DATA_DIRS:
        for fname in glob.glob(os.path.join(base_dir, "*.json")):
            view, cam = parse_filename(os.path.basename(fname))
            with open(fname, 'r') as f:
                content = json.load(f)
                for mid, mdata in content.items():
                    if "corners_pixel" in mdata: raw_data[view][cam][mid].append(mdata)

    corrected_data_stage1 = defaultdict(lambda: defaultdict(dict))
    for view, cams_data in raw_data.items():
        for cam, markers_data in cams_data.items():
            for mid, entries in markers_data.items():
                if len(entries) < 2:
                    corrected_data_stage1[view][cam][mid] = entries[0]
                    continue
                
                pos = np.array([[m['position_m'][k] for k in 'xyz'] for m in entries])
                quat = np.array([[m['rotation_quat'][k] for k in 'xyzw'] for m in entries])
                corners = np.array([m['corners_pixel'] for m in entries], dtype=np.float32)
                
                pos_f, quat_f, mask = remove_outliers(pos, quat)
                if len(pos_f) == 0 or len(pos_f) < len(entries) / 3: # 필터링 후 데이터가 너무 적으면 제외
                    print(f"경고: 마커 {mid} ({view}_{cam}) 데이터가 불안정하여 제외됩니다. (원본 {len(entries)}개 -> 필터링 후 {len(pos_f)}개)")
                    continue
                
                corrected_data_stage1[view][cam][mid] = {
                    "position_m": {k: float(v) for k, v in zip('xyz', average_position(pos_f))},
                    "rotation_quat": {k: float(v) for k, v in zip('xyzw', average_quaternion(quat_f))},
                    "corners_pixel": np.mean(corners[mask], axis=0).tolist()
                }
            
            # [수정] 요청하신대로 보정된 데이터를 파일로 저장하는 로직 추가
            if corrected_data_stage1[view][cam]:
                output_path = os.path.join(CORRECTED_ARUCO_DIR, f"{view}_{cam}_corrected.json")
                with open(output_path, 'w') as f:
                    json.dump(corrected_data_stage1[view][cam], f, indent=4)
                print(f"보정된 데이터 저장 완료: {output_path}")

    print("--- STAGE 1 완료 ---\n")
    return corrected_data_stage1

def stage2_recalculate_pose_from_corners(stage1_data):
    """2단계: 1단계의 평균 코너 좌표를 이용해 solvePnP로 포즈를 정밀하게 재계산합니다."""
    print("--- STAGE 2: 코너 좌표 기반 포즈 재계산 시작 ---")
    marker_3d_pts = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], dtype=np.float32) * MARKER_REAL_SIZE_M / 2
    recalculated_data = defaultdict(lambda: defaultdict(dict))
    
    for view, cams_data in stage1_data.items():
        for cam, markers_data in cams_data.items():
            # [수정] VIEW_TO_SERIAL을 사용하여 view 이름으로 serial 번호를 찾음
            serial = VIEW_TO_SERIAL.get(view)
            if not serial:
                print(f"경고: {view}에 해당하는 serial 번호를 찾을 수 없습니다.")
                continue
            
            calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            if not os.path.exists(calib_path):
                print(f"경고: 캘리브레이션 파일 없음: {calib_path}")
                continue
            
            with open(calib_path) as f: calib = json.load(f)
            K, dist = np.array(calib["camera_matrix"]), np.array(calib["distortion_coeffs"])
            
            for mid, data in markers_data.items():
                corners = np.array(data["corners_pixel"], dtype=np.float32)
                ret, rvec, tvec = cv2.solvePnP(marker_3d_pts, corners, K, dist)
                if not ret: continue
                
                # Levenberg-Marquardt 최적화를 사용하여 포즈 정밀도 향상
                rvec, tvec = cv2.solvePnPRefineLM(marker_3d_pts, corners, K, dist, rvec, tvec)
                
                quat = R.from_rotvec(rvec.flatten()).as_quat()
                recalculated_data[view][cam][mid] = {
                    "position_m": {k: float(v) for k, v in zip('xyz', tvec.flatten())},
                    "rotation_quat": {k: float(v) for k, v in zip('xyzw', quat)},
                }
    print("--- STAGE 2 완료 ---\n")
    return recalculated_data

def stage3_visualize_and_summarize(stage2_data):
    """3단계: 최종 오프셋 적용, 시각화 및 요약."""
    print("--- STAGE 3: 최종 오프셋 적용, 시각화 및 요약 시작 ---")
    os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)
    final_summary = []
    
    for view in VIEWS:
        for cam in CAMS:
            poses = stage2_data.get(view, {}).get(cam)
            if not poses: continue
            
            # [수정] VIEW_TO_SERIAL을 사용하여 view 이름으로 serial 번호를 찾음
            serial = VIEW_TO_SERIAL.get(view)
            if not serial: continue
            
            calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            img_paths = glob.glob(os.path.join(IMAGE_DIR, f"{view}_*_{cam}_*.png"))
            if not (os.path.exists(calib_path) and img_paths):
                print(f"경고: {view}_{cam}의 캘리브레이션 파일 또는 이미지가 없습니다.")
                continue

            with open(calib_path) as f: calib = json.load(f)
            K, dist = np.array(calib["camera_matrix"]), np.array(calib["distortion_coeffs"])
            
            # 왜곡 보정된 이미지를 시각화에 사용
            img_vis = cv2.cvtColor(cv2.undistort(cv2.imread(img_paths[0]), K, dist, None, K), cv2.COLOR_BGR2RGB)

            offset_pos, all_quats = [], []
            for mid, offset in MARKER_OFFSETS[view].items():
                if mid not in poses: continue
                
                p = poses[mid]
                tvec = np.array([p["position_m"][k] for k in 'xyz'])
                quat = np.array([p["rotation_quat"][k] for k in 'xyzw'])
                
                # 오프셋을 적용하여 객체의 원점 위치 추정
                offset_pos.append(tvec + R.from_quat(quat).as_matrix() @ offset)
                all_quats.append(quat)
                
                # 개별 마커의 좌표축 그리기
                rvec = R.from_quat(quat).as_rotvec()
                cv2.drawFrameAxes(img_vis, K, None, rvec, tvec, 0.03) # 왜곡 보정된 이미지이므로 dist=None
                proj_pt, _ = cv2.projectPoints(tvec.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
                cv2.putText(img_vis, f"ID:{mid}", tuple(proj_pt.ravel().astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            if len(offset_pos) < 2: 
                print(f"경고: {view}_{cam}에서 유효 마커가 부족하여 평균 포즈를 계산할 수 없습니다.")
                continue
            
            # 여러 마커에서 추정된 객체 위치의 평균과 표준편차 계산
            mean_pos = average_position(np.array(offset_pos))
            std_pos = np.std(offset_pos, axis=0)
            
            # 모든 쿼터니언을 평균내어 객체의 평균 방향 계산
            mean_quat = average_quaternion(np.array(all_quats))
            mean_rvec = R.from_quat(mean_quat).as_rotvec()
            
            # 평균 포즈 좌표축 그리기 (더 크고 굵게)
            cv2.drawFrameAxes(img_vis, K, None, mean_rvec, mean_pos, 0.08, 4)
            mean_proj, _ = cv2.projectPoints(mean_pos.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
            xm, ym = mean_proj.ravel().astype(int)
            cv2.drawMarker(img_vis, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 25, 3)

            final_summary.append({
                "view": view, "cam": cam, "mean_x_m": mean_pos[0], "mean_y_m": mean_pos[1], "mean_z_m": mean_pos[2],
                "std_x_mm": std_pos[0]*1000, "std_y_mm": std_pos[1]*1000, "std_z_mm": std_pos[2]*1000,
                "rvec_x_deg": np.rad2deg(mean_rvec)[0], "rvec_y_deg": np.rad2deg(mean_rvec)[1], "rvec_z_deg": np.rad2deg(mean_rvec)[2]
            })
            
            plt.figure(figsize=(12, 9)); plt.imshow(img_vis)
            plt.title(f"Final Mean Pose ({view.upper()}-{cam}) | Pos Std (mm): {std_pos[0]*1000:.1f}, {std_pos[1]*1000:.1f}, {std_pos[2]*1000:.1f}", fontsize=14)
            plt.axis('off'); plt.tight_layout()
            
            output_path = os.path.join(RESULT_IMAGE_DIR, f"{view}_{cam}_final_pose.png")
            plt.savefig(output_path); plt.close()
            print(f"이미지 저장 완료: {output_path}")

    df = pd.DataFrame(final_summary)
    if not df.empty:
        df.to_json(FINAL_SUMMARY_OUTPUT_PATH, orient="records", indent=4)
        print(f"\n--- STAGE 3 완료 ---")
        print(f"최종 요약 파일 저장 완료: {FINAL_SUMMARY_OUTPUT_PATH}")
        print("--- 최종 요약 결과 ---")
        print(df)
    else:
        print("\n--- STAGE 3 완료 ---")
        print("처리할 유효 데이터가 없어 요약 파일을 생성하지 않았습니다.")

# =================================================================================
# ▶️ 메인 실행 블록 (Main Execution Block)
# =================================================================================
if __name__ == "__main__":
    extract_camera_calibration()
    corrected_data = stage1_average_raw_data()
    recalculated_data = stage2_recalculate_pose_from_corners(corrected_data)
    stage3_visualize_and_summarize(recalculated_data)
    print("\n🎉 --- 모든 작업이 완료되었습니다. --- 🎉")