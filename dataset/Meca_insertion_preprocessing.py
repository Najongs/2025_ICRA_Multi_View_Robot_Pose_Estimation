# -*- coding: utf-8 -*-
"""
ArUco 마커 데이터를 처리하여 최종 객체 자세를 추정하고 결과를 요약, 시각화하는 스크립트.

실행 순서:
1. 여러 세션에서 캡처된 Raw 데이터의 노이즈를 제거하고 평균을 계산합니다.
2. 평균낸 마커 코너 좌표를 사용하여 카메라 보정값 기준으로 자세를 다시 계산합니다.
3. 각 마커의 오프셋을 적용하여 최종 객체 중심의 자세를 추정하고, 이를 시각화하여 이미지 파일로 저장하며,
   모든 결과를 종합하여 JSON 파일로 출력합니다.
"""
import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

# =================================================================================
# 1. 통합 설정 (Configuration)
# =================================================================================

# --- Raw 데이터 경로 설정 ---
RAW_DATA_DIRS = [
    "./Meca_insertion/Meca_ArUco/ArUco_cap_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap2_250514",
    "./Meca_insertion/Meca_ArUco/ArUco_cap3_250514"
]

# --- 보정 및 시각화 관련 경로 설정 ---
CALIB_DIR = "./Meca_insertion/Meca_calib_cam_from_conf"
ZED_CONF_DIR = "./All_camera_conf"
IMAGE_DIR = "./Meca_insertion/Meca_ArUco/ArUco_cap_250514"  # 시각화에 사용할 대표 이미지 경로
RESULTS_IMAGE_DIR = "./Meca_insertion/Meca_calib_results_images"  # ★★★ 결과 이미지 저장 폴더
FINAL_SUMMARY_OUTPUT_PATH = "./Meca_insertion/aruco_final_summary.json"

# --- 파라미터 설정 ---
MARKER_REAL_SIZE_M = 0.05  # ArUco 마커의 실제 한 변 길이 (미터 단위)

# --- 카메라 및 뷰 매핑 정보 ---
camera_serials = {"front": 41182735, "right": 49429257, "left": 44377151, "top": 49045152}
camera_list = {v: k for k, v in camera_serials.items()} # serial: position 형태의 역방향 맵
views = ['front', 'left', 'right', 'top']
cams = ['leftcam', 'rightcam']

# --- 최종 객체 자세 추정을 위한 마커 오프셋 (객체 중심 기준 마커 위치) ---
marker_offsets = {
    "front": {"1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065])},
    "left": {"3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0.000, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065])},
    "right": {"1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0.000, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "7": np.array([0.000, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065])},
    "top": {"1": np.array([-0.100, 0.125, 0.0065]), "2": np.array([-0.100, 0.025, 0.0065]), "3": np.array([0, -0.175, 0.0065]), "4": np.array([-0.100, -0.075, 0.0065]), "5": np.array([0.125, 0.025, 0.0065]), "6": np.array([0.125, 0.125, 0.0065]), "7": np.array([0, -0.075, 0.0065]), "8": np.array([0.125, -0.075, 0.0065])},
}

# =================================================================================
# 2. 헬퍼 함수 (Helper Functions)
# =================================================================================
def parse_filename(filename):
    """파일 이름에서 view와 cam 정보를 추출합니다."""
    parts = os.path.basename(filename).split('_')
    view = parts[0]
    cam = parts[2]
    return view, cam

def average_quaternion(quaternions):
    """쿼터니언 배열의 평균을 계산합니다."""
    if len(quaternions) == 0: return np.array([0, 0, 0, 1])
    # Scipy의 Rotation 객체를 사용하여 평균 계산
    return R.from_quat(quaternions).mean().as_quat()

def average_position(positions):
    """3D 좌표 배열의 평균을 계산합니다."""
    if len(positions) == 0: return np.array([0, 0, 0])
    return np.mean(positions, axis=0)

def remove_outliers(positions, quaternions, pos_thresh=0.001, rot_thresh_deg=3):
    """위치 및 회전 데이터에서 이상치를 제거합니다."""
    if len(positions) < 2: return positions, quaternions, np.ones(len(positions), dtype=bool)
    
    avg_pos = average_position(positions)
    avg_quat = average_quaternion(quaternions)
    
    # 위치 이상치 제거
    pos_dists = np.linalg.norm(positions - avg_pos, axis=1)
    pos_mask = pos_dists < pos_thresh
    
    # 회전 이상치 제거 (평균 회전과의 각도 차이 기준)
    avg_rot = R.from_quat(avg_quat)
    angular_distances = np.array([np.rad2deg((avg_rot.inv() * R.from_quat(q)).magnitude()) for q in quaternions])
    rot_mask = angular_distances < rot_thresh_deg
    
    valid_mask = pos_mask & rot_mask
    return positions[valid_mask], quaternions[valid_mask], valid_mask

# =================================================================================
# 3. 주요 처리 함수 (Core Logic Functions)
# =================================================================================

def generate_calibration_files():
    """ZED SDK의 .conf 파일에서 카메라 보정 정보를 읽어 JSON 파일로 저장합니다."""
    print("--- Running Calibration File Generation ---")
    os.makedirs(CALIB_DIR, exist_ok=True)
    
    def load_fhd_calibration(conf_path, side):
        config = configparser.ConfigParser()
        with open(conf_path, "r", encoding="utf-8-sig") as f:
            config.read_file(f)
        
        section = f"{side.upper()}_CAM_FHD1200"
        cam = config[section]
        
        camera_matrix = [[float(cam["fx"]), 0.0, float(cam["cx"])],
                         [0.0, float(cam["fy"]), float(cam["cy"])],
                         [0.0, 0.0, 1.0]]
        distortion_coeffs = [float(cam[k]) for k in ["k1", "k2", "p1", "p2", "k3"]]
        
        return camera_matrix, distortion_coeffs

    for serial, position in camera_list.items():
        conf_path = os.path.join(ZED_CONF_DIR, f"SN{serial}.conf")
        if not os.path.exists(conf_path):
            print(f"[{position}] Warning: Config file not found: {conf_path}")
            continue

        for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
            try:
                cam_matrix, dist_coeffs = load_fhd_calibration(conf_path, side)
                data = {"camera_matrix": cam_matrix, "distortion_coeffs": dist_coeffs}
                filename = f"{position}_{serial}_{side_name}_calib.json"
                with open(os.path.join(CALIB_DIR, filename), "w") as f:
                    json.dump(data, f, indent=4)
                print(f"[{position}] Calibration saved: {filename}")
            except Exception as e:
                print(f"[{position}] Error processing {side_name}: {e}")
    print("--- Calibration File Generation Complete ---\n")


def process_raw_data():
    """STAGE 1: Raw 데이터를 읽어 이상치를 제거하고 평균값을 계산합니다."""
    print("--- STAGE 1: Averaging Raw Data and Removing Outliers ---")
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for base_dir in RAW_DATA_DIRS:
        for fname in glob.glob(os.path.join(base_dir, '*.json')):
            view, cam = parse_filename(fname)
            with open(fname, 'r') as f:
                content = json.load(f)
                for marker_id, marker_data in content.items():
                    if "corners_pixel" in marker_data:
                        raw_data[view][cam][marker_id].append(marker_data)

    corrected_data = defaultdict(lambda: defaultdict(dict))
    for view, cams_data in raw_data.items():
        for cam, markers_data in cams_data.items():
            for marker_id, entries in markers_data.items():
                if len(entries) < 2:
                    corrected_data[view][cam][marker_id] = entries[0]
                    continue
                
                positions = np.array([[m['position_m'][k] for k in 'xyz'] for m in entries])
                quaternions = np.array([[m['rotation_quat'][k] for k in 'xyzw'] for m in entries])
                corners = np.array([m['corners_pixel'] for m in entries], dtype=np.float32)

                pos_f, quat_f, mask = remove_outliers(positions, quaternions)
                
                if len(pos_f) == 0 or len(pos_f) < len(entries) / 2:
                    print(f"Warning: Marker {marker_id} in {view}_{cam} excluded due to excessive outliers.")
                    continue

                corrected_data[view][cam][marker_id] = {
                    "position_m": dict(zip('xyz', average_position(pos_f))),
                    "rotation_quat": dict(zip('xyzw', average_quaternion(quat_f))),
                    "corners_pixel": np.mean(corners[mask], axis=0).tolist()
                }
    print("--- STAGE 1 Complete ---\n")
    return corrected_data


def recalculate_poses(stage1_data):
    """STAGE 2: Stage 1의 평균 코너 좌표를 이용해 Pose를 다시 계산합니다."""
    print("--- STAGE 2: Recalculating Pose from Averaged Corners ---")
    marker_3d_points = np.array([
        [0, 0, 0], [MARKER_REAL_SIZE_M, 0, 0], 
        [MARKER_REAL_SIZE_M, MARKER_REAL_SIZE_M, 0], [0, MARKER_REAL_SIZE_M, 0]
    ], dtype=np.float32)

    recalculated_data = defaultdict(lambda: defaultdict(dict))
    for view, cams_data in stage1_data.items():
        for cam, markers_data in cams_data.items():
            serial = camera_serials.get(view)
            if not serial: continue
            
            calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            if not os.path.exists(calib_path): continue
            
            with open(calib_path) as f: calib_data = json.load(f)
            camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float64)
            dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float64)

            for marker_id, data in markers_data.items():
                corners_2d = np.array(data["corners_pixel"], dtype=np.float32).reshape((4, 1, 2))
                ret, rvec, tvec = cv2.solvePnP(marker_3d_points, corners_2d, camera_matrix, dist_coeffs)
                if not ret: continue
                
                # Refine the pose
                rvec, tvec = cv2.solvePnPRefineLM(marker_3d_points, corners_2d, camera_matrix, dist_coeffs, rvec, tvec)
                
                quat = R.from_rotvec(rvec.flatten()).as_quat()
                recalculated_data[view][cam][marker_id] = {
                    "position_m": dict(zip('xyz', tvec.flatten())),
                    "rotation_quat": dict(zip('xyzw', quat)),
                    "corners_pixel": data["corners_pixel"]
                }
    print("--- STAGE 2 Complete ---\n")
    return recalculated_data


def summarize_and_visualize(stage2_data):
    """STAGE 3: 최종 오프셋 적용, 시각화 및 요약 파일을 생성합니다."""
    print("--- STAGE 3: Applying Final Offsets, Visualizing, and Summarizing ---")
    final_summary = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for view in views:
        for cam in cams:
            poses = stage2_data.get(view, {}).get(cam)
            if not poses: continue
            
            serial = camera_serials.get(view)
            calib_path = os.path.join(CALIB_DIR, f"{view}_{serial}_{cam}_calib.json")
            img_files = glob.glob(os.path.join(IMAGE_DIR, f"{view}_*_{cam}_*.png"))
            if not os.path.exists(calib_path) or not img_files: continue

            with open(calib_path) as f: calib = json.load(f)
            K = np.array(calib["camera_matrix"], dtype=np.float64)
            dist = np.array(calib["distortion_coeffs"], dtype=np.float64)
            
            img = cv2.imread(img_files[0])
            undistorted_img = cv2.undistort(img, K, dist, None, K)
            
            offset_positions, all_quats, valid_ids = [], [], []

            for mid, offset in marker_offsets.get(view, {}).items():
                if mid not in poses or mid == "8": continue  # ID 8은 제외 (설정에 따라 변경 가능)
                
                p = poses[mid]
                tvec = np.array(list(p["position_m"].values()))
                quat = np.array(list(p["rotation_quat"].values()))
                Rm = R.from_quat(quat).as_matrix()
                
                # 마커 자세를 기준으로 오프셋을 적용하여 객체 중심 위치 계산
                tvec_with_offset = tvec + Rm @ offset
                offset_positions.append(tvec_with_offset)
                all_quats.append(quat)
                valid_ids.append(mid)
                
                # 시각화: 각 마커의 축 그리기
                rvec = R.from_quat(quat).as_rotvec()
                cv2.drawFrameAxes(undistorted_img, K, None, rvec, tvec, 0.05)
                marker_pos_2d, _ = cv2.projectPoints(tvec.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
                cv2.putText(undistorted_img, f"ID:{mid}", tuple(marker_pos_2d.ravel().astype(int)), font, 0.6, (255, 255, 0), 2)

            if not offset_positions: continue
            
            # 여러 마커에서 계산된 객체 중심 위치/자세의 평균 계산
            mean_pos = np.mean(offset_positions, axis=0)
            std_pos = np.std(offset_positions, axis=0)
            mean_quat = average_quaternion(np.array(all_quats))
            mean_rvec = R.from_quat(mean_quat).as_rotvec()
            
            # 시각화: 최종 평균 자세 축 및 중심점 그리기
            cv2.drawFrameAxes(undistorted_img, K, None, mean_rvec, mean_pos, 0.1, thickness=4)
            mean_proj, _ = cv2.projectPoints(mean_pos.reshape(1,3), np.zeros(3), np.zeros(3), K, None)
            xm, ym = mean_proj.ravel().astype(int)
            cv2.drawMarker(undistorted_img, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            # Matplotlib을 사용하여 결과 이미지 저장
            plt.figure(figsize=(12, 9))
            plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Final Mean Pose ({view.upper()}-{cam})")
            plt.axis('off')
            plt.tight_layout()
            
            output_image_path = os.path.join(RESULTS_IMAGE_DIR, f"{view}_{cam}_result.png")
            plt.savefig(output_image_path)
            plt.close() # 메모리 해제를 위해 창 닫기
            print(f"Saved visualization: {output_image_path}")
            
            # 요약 정보 추가
            deg_rvec = np.rad2deg(mean_rvec)
            final_summary.append({
                "view": view, "cam": cam, 
                "mean_x_m": mean_pos[0], "mean_y_m": mean_pos[1], "mean_z_m": mean_pos[2],
                "std_x_m": std_pos[0], "std_y_m": std_pos[1], "std_z_m": std_pos[2],
                "proj_x_px": xm, "proj_y_px": ym,
                "rvec_x_deg": deg_rvec[0], "rvec_y_deg": deg_rvec[1], "rvec_z_deg": deg_rvec[2]
            })
            
    return final_summary

# =================================================================================
# 4. 메인 실행 함수 (Main Execution)
# =================================================================================
def main():
    """전체 데이터 처리 파이프라인을 실행합니다."""
    # 결과 이미지 저장 폴더 생성
    os.makedirs(RESULTS_IMAGE_DIR, exist_ok=True)
    print(f"Result images will be saved to: {RESULTS_IMAGE_DIR}\n")

    # (필요시 실행) ZED .conf 파일로부터 .json 캘리브레이션 파일 생성
    generate_calibration_files()
    
    # STAGE 1: 데이터 로드 및 이상치 제거
    stage1_results = process_raw_data()
    
    # STAGE 2: Pose 재계산
    stage2_results = recalculate_poses(stage1_results)
    
    # STAGE 3: 최종 요약 및 시각화
    final_summary_data = summarize_and_visualize(stage2_results)
    
    # 최종 요약 파일 저장
    if final_summary_data:
        df = pd.DataFrame(final_summary_data)
        df.to_json(FINAL_SUMMARY_OUTPUT_PATH, orient="records", indent=4)
        print(f"\n--- All stages complete. Final summary saved to: {FINAL_SUMMARY_OUTPUT_PATH} ---")
        print("Final Summary DataFrame:")
        print(df)
    else:
        print("\n--- Processing finished, but no data was generated for the final summary. ---")

if __name__ == "__main__":
    main()