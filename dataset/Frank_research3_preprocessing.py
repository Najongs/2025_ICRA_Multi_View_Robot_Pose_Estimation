import os
import json
import glob
import configparser
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

# -------------------- ⚙️ 1. 전체 설정 (Global Configuration) --------------------

# --- 기본 경로 설정 ---
BASE_DIR = "./frank_research3"
ALL_CAMERA_CONF_DIR = "./All_camera_conf"

# --- Pose 1 경로 ---
POSE1_INPUT_BASE = os.path.join(BASE_DIR, "frank_research3_ArUco_pose1")
POSE1_CORRECTED_DIR = os.path.join(BASE_DIR, "frank_research3_correct_ArUco_pose1")
POSE1_SUMMARY_DIR = "./frank_research3/pose1_summary"

# --- Pose 2 경로 ---
POSE2_INPUT_BASE = os.path.join(BASE_DIR, "frank_research3_ArUco_pose2")
POSE2_CORRECTED_DIR = os.path.join(BASE_DIR, "frank_research3_correct_ArUco_pose2")
POSE2_SUMMARY_DIR = "./frank_research3/pose2_summary"

# --- 공통 경로 ---
CALIB_JSON_DIR = os.path.join(BASE_DIR, "Calib_cam_from_conf")
RESULT_IMAGE_DIR = os.path.join(BASE_DIR, "frank_research3_calib_results_images")

# --- 카메라 설정 ---
CAMERA_SERIALS = {
    41182735: "view1",
    49429257: "view2",
    44377151: "view3",
    49045152: "view4"
}
SERIAL_MAP = {v: k for k, v in CAMERA_SERIALS.items()}
VIEWS = ['view1', 'view2', 'view3', 'view4']
CAMS = ['leftcam', 'rightcam']

# --- 마커 오프셋 설정 ---
MARKER_OFFSETS = {
    "view1": {"1": [0.025, 0.20, -0.01], "2": [-0.175, 0.0, -0.01], "4": [-0.30, 0.0, -0.01], "5": [0.35, 0.0, -0.01], "6": [0.025, 0.325, -0.01]},
    "view2": {"2": [-0.175, 0.0, -0.01], "4": [-0.30, 0.00, -0.01], "7": [0.025, -0.225, -0.01], "8": [0.025, -0.325, -0.01]},
    "view3": {"3": [0.225, 0.0, -0.01], "5": [0.35, 0.0, -0.01], "7": [0.025, -0.225, -0.01], "8": [0.025, -0.325, -0.01]},
    "view4": {"1": [0.025, 0.20, -0.01], "2": [-0.175, 0.0, -0.01], "4": [-0.30, 0.01, -0.01], "6": [0.025, 0.325, -0.01], "7": [0.025, -0.225, -0.01], "8": [0.025, -0.325, -0.01]}
}
# 오프셋을 numpy 배열로 변환
for view_key, markers in MARKER_OFFSETS.items():
    for marker_key, offset in markers.items():
        MARKER_OFFSETS[view_key][marker_key] = np.array(offset)

# -------------------- 헬퍼 함수 (Helper Functions) --------------------
def parse_filename(filename):
    parts = filename.split('_')
    return parts[0], parts[2]

def average_quaternion(quaternions):
    M = np.zeros((4, 4))
    for q in quaternions:
        q = q.reshape(4, 1)
        M += q @ q.T
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_quat = eigvecs[:, np.argmax(eigvals)]
    return avg_quat / np.linalg.norm(avg_quat)

def angular_distance_deg(q1, q2):
    r1 = R.from_quat(q1); r2 = R.from_quat(q2)
    return np.rad2deg((r1.inv() * r2).magnitude())

def align_quaternions(quaternions):
    aligned = np.copy(quaternions)
    ref = aligned[0]
    for i in range(1, len(aligned)):
        if np.dot(ref, aligned[i]) < 0:
            aligned[i] *= -1
    return aligned

def average_position(positions):
    return np.mean(positions, axis=0)

# -------------------- 📜 기능 1: ArUco 데이터 보정 함수 --------------------
def correct_aruco_data(input_base_dir, output_dir):
    print(f"--- [ArUco 데이터 보정 시작] ---")
    print(f"입력: {input_base_dir}\n출력: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    input_dirs = glob.glob(os.path.join(input_base_dir, "ArUco_capture_dataset_*"))
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for base_dir in input_dirs:
        for fname in os.listdir(base_dir):
            if not fname.endswith('.json'): continue
            view, cam = parse_filename(fname)
            with open(os.path.join(base_dir, fname), 'r') as f:
                content = json.load(f)
                for marker_id, marker_data in content.items():
                    data[view][cam][marker_id].append(marker_data)
    
    for view, cams_data in data.items():
        for cam, markers_data in cams_data.items():
            corrected = {}
            for marker_id, entries in markers_data.items():
                if len(entries) < 2: continue
                positions = [np.array([m['position_m'][k] for k in 'xyz']) for m in entries]
                quaternions = [np.array([m['rotation_quat'][k] for k in 'xyzw']) for m in entries]

                aligned_quats = align_quaternions(np.array(quaternions))
                avg_quat_initial = average_quaternion(aligned_quats)

                filtered_pos = [p for p, q in zip(positions, quaternions) if angular_distance_deg(avg_quat_initial, q) <= 2.0]
                filtered_quat = [q for q in quaternions if angular_distance_deg(avg_quat_initial, q) <= 2.0]

                if not filtered_pos: continue
                
                avg_pos = average_position(filtered_pos)
                avg_quat = average_quaternion(align_quaternions(np.array(filtered_quat)))
                
                corrected[marker_id] = {
                    "position_m": {k: float(v) for k, v in zip('xyz', avg_pos)},
                    "rotation_quat": {k: float(v) for k, v in zip('xyzw', avg_quat)},
                    "corners_pixel": entries[0]["corners_pixel"]
                }
            
            if corrected:
                output_path = os.path.join(output_dir, f"{view}_{cam}_corrected.json")
                with open(output_path, 'w') as f: json.dump(corrected, f, indent=4)
    print(f"--- [ArUco 데이터 보정 완료] ---\n")

# -------------------- 📜 기능 2: 카메라 캘리브레이션 추출 함수 --------------------
def extract_camera_calibration():
    print(f"--- [카메라 캘리브레이션 추출 시작] ---")
    os.makedirs(CALIB_JSON_DIR, exist_ok=True)
    
    def load_fhd_calibration(conf_path, side):
        config = configparser.ConfigParser()
        with open(conf_path, "r", encoding="utf-8-sig") as f: config.read_file(f)
        section = f"{side.upper()}_CAM_FHD1200"
        cam = config[section]
        cam_matrix = [[float(cam["fx"]), 0.0, float(cam["cx"])], [0.0, float(cam["fy"]), float(cam["cy"])], [0.0, 0.0, 1.0]]
        dist_coeffs = [float(cam[k]) for k in ["k1", "k2", "p1", "p2", "k3"]]
        return cam_matrix, dist_coeffs

    for serial, position in CAMERA_SERIALS.items():
        conf_path = os.path.join(ALL_CAMERA_CONF_DIR, f"SN{serial}.conf")
        if not os.path.exists(conf_path):
            print(f"경고: [{position}] 설정 파일 없음: {conf_path}")
            continue
        for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
            try:
                cam_matrix, dist_coeffs = load_fhd_calibration(conf_path, side)
                data = {"camera_matrix": cam_matrix, "distortion_coeffs": dist_coeffs}
                filename = f"{position}_{serial}_{side_name}_calib.json"
                with open(os.path.join(CALIB_JSON_DIR, filename), "w") as f: json.dump(data, f, indent=4)
                print(f"[{position}] 저장 완료: {filename}")
            except Exception as e:
                print(f"오류: [{position}] {side_name} 처리 중 오류: {e}")
    print(f"--- [카메라 캘리브레이션 추출 완료] ---\n")

# -------------------- 📜 기능 3: 최종 포즈 계산 및 시각화/저장 함수 --------------------
def calculate_and_visualize_poses(pose_name, corrected_json_dir, image_base_dir, summary_output_dir):
    print(f"--- [최종 포즈 계산 및 시각화 시작: {pose_name.upper()}] ---")
    os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)

    summary = []
    
    for view in VIEWS:
        serial = SERIAL_MAP[view]
        for cam in CAMS:
            pose_path = os.path.join(corrected_json_dir, f"{view}_{cam}_corrected.json")
            calib_path = os.path.join(CALIB_JSON_DIR, f"{view}_{serial}_{cam}_calib.json")
            
            if not (os.path.exists(pose_path) and os.path.exists(calib_path)): continue

            with open(pose_path) as f: poses = json.load(f)
            with open(calib_path) as f: calib = json.load(f)

            K = np.array(calib["camera_matrix"], dtype=np.float64)
            dist = np.array(calib["distortion_coeffs"], dtype=np.float64)

            tvecs, quats, marker_ids = [], [], []
            for mid, offset in MARKER_OFFSETS[view].items():
                if mid not in poses: continue
                p = poses[mid]
                tvec = np.array([p["position_m"][k] for k in 'xyz'])
                quat = np.array([p["rotation_quat"][k] for k in 'xyzw'])
                Rm = R.from_quat(quat).as_matrix()
                tvec_with_offset = tvec + Rm.dot(offset)
                
                tvecs.append(tvec_with_offset)
                quats.append(quat)
                marker_ids.append(mid)
            
            if not tvecs: continue
            
            mean_tvec = average_position(np.array(tvecs))
            mean_quat = average_quaternion(np.array(quats))
            mean_rvec = R.from_quat(mean_quat).as_rotvec()
            
            # --- 시각화 ---
            image_dir = glob.glob(os.path.join(image_base_dir, "ArUco_capture_dataset_*"))[0]
            img_files = glob.glob(os.path.join(image_dir, f"{view}_*_{cam}_*.png"))
            if not img_files: continue
            
            img_rgb = cv2.cvtColor(cv2.imread(img_files[0]), cv2.COLOR_BGR2RGB)
            
            # 개별 마커 자세 시각화
            for (quat, mid) in zip(quats, marker_ids):
                p = poses[mid]
                tvec_marker = np.array([p["position_m"][k] for k in 'xyz']) # 오프셋 적용 전 원래 마커 위치
                rot_vec = R.from_quat(quat).as_rotvec()
                cv2.drawFrameAxes(img_rgb, K, dist, rot_vec, tvec_marker.reshape(3, 1), 0.05)
                marker_pos_2d, _ = cv2.projectPoints(tvec_marker.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist)
                x, y = marker_pos_2d.ravel().astype(int)
                cv2.putText(img_rgb, f"ID:{mid}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # 평균 자세 시각화
            cv2.drawFrameAxes(img_rgb, K, dist, mean_rvec, mean_tvec, 0.1, 3) # 더 굵고 크게
            proj_point, _ = cv2.projectPoints(mean_tvec.reshape(1,3), np.zeros(3), np.zeros(3), K, dist)
            xm, ym = proj_point.ravel().astype(int)
            cv2.drawMarker(img_rgb, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

            # --- 이미지 파일로 저장 ---
            plt.figure(figsize=(16, 9))
            plt.imshow(img_rgb)
            plt.title(f"[{pose_name.upper()}] {view.upper()} - {cam}", fontsize=16)
            plt.axis('off')
            
            output_image_path = os.path.join(RESULT_IMAGE_DIR, f"{pose_name}_{view}_{cam}_visualization.png")
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"이미지 저장 완료: {output_image_path}")

            summary.append([view, cam, *mean_tvec, *mean_rvec, xm, ym])
    
    # --- 요약 파일 저장 ---
    columns = ["view", "cam", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z", "proj_x", "proj_y"]
    df = pd.DataFrame(summary, columns=columns)
    summary_path = f"{pose_name}_aruco_pose_summary.json"
    df.to_json(summary_path, orient="records", indent=2)
    print(f"요약 파일 저장 완료: {summary_path}")
    print(f"--- [최종 포즈 계산 및 시각화 완료: {pose_name.upper()}] ---\n")

# -------------------- ▶️ 메인 실행 블록 (Main Execution) --------------------
if __name__ == "__main__":
    # 단계 1: 카메라 캘리브레이션 정보 추출 (공통 과정, 1회 실행)
    extract_camera_calibration()

    # 단계 2: Pose 1 데이터 처리 및 시각화
    correct_aruco_data(POSE1_INPUT_BASE, POSE1_CORRECTED_DIR)
    calculate_and_visualize_poses("pose1", POSE1_CORRECTED_DIR, POSE1_INPUT_BASE, POSE1_SUMMARY_DIR)

    # 단계 3: Pose 2 데이터 처리 및 시각화
    correct_aruco_data(POSE2_INPUT_BASE, POSE2_CORRECTED_DIR)
    calculate_and_visualize_poses("pose2", POSE2_CORRECTED_DIR, POSE2_INPUT_BASE, POSE2_SUMMARY_DIR)

    print("🎉 --- 모든 작업이 완료되었습니다. --- 🎉")