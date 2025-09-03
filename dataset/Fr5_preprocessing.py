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

# -------------------- 설정 (Configuration) --------------------

# 기본 디렉토리 경로
BASE_PROJECT_DIR = "./Fr5"
RAW_ARUCO_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_ArUco")
CORRECTED_ARUCO_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_correct_ArUco")
CALIB_CONF_DIR = "./All_camera_conf"
CALIB_JSON_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_calib_cam_from_conf")
IMAGE_SOURCE_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_ArUco")
RESULT_IMAGE_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_calib_results_images") # 결과 이미지 저장 경로
SUMMARY_DIR = os.path.join(BASE_PROJECT_DIR, "Fr5_aruco_pose_summary")

# 카메라 정보
CAMERA_SERIALS = {
    "top": 30779426,
    "right": 34850673,
    "left": 38007749
}
VIEWS = ['left', 'right', 'top']
CAMS = ['leftcam', 'rightcam']

# 각 View의 기준 좌표계로부터 마커까지의 상대 위치 (단위: 미터)
MARKER_OFFSETS = {
    "left": {
        "1": np.array([0.095, -0.135, -0.01]), "2": np.array([0.025, -0.135, -0.01]), # "3": np.array([-0.01, -0.295, -0.12]),
        "4": np.array([0.095, -0.215, -0.01]), "5": np.array([0.025, -0.215, -0.01]), # "6": np.array([-0.01, -0.375, -0.12]),
    },
    "right": {
        "1": np.array([0.095, -0.135, -0.01]), "2": np.array([0.025, -0.135, -0.01]), # "3": np.array([0.09, -0.375, -0.12]),
        "4": np.array([0.095, -0.215, -0.01]), "5": np.array([0.025, -0.215, -0.01]), # "6": np.array([0.09, -0.295, -0.12]),
    },
    "top": {
        "1": np.array([0.095, -0.135, -0.01]), "2": np.array([0.025, -0.135, -0.01]), "3": np.array([-0.055, -0.135, -0.01]),
        "4": np.array([0.095, -0.215, -0.01]), "5": np.array([0.025, -0.215, -0.01]), "6": np.array([-0.055, -0.215, -0.01]),
    }
}


# -------------------- 헬퍼 함수 (Helper Functions) --------------------

def parse_filename(filename):
    """파일 이름에서 view와 cam 정보를 추출합니다."""
    parts = filename.split('_')
    return parts[0], parts[2]

def average_quaternion(quaternions):
    """쿼터니언 배열의 평균을 계산합니다."""
    M = np.zeros((4, 4))
    for q in quaternions:
        q = q.reshape(4, 1)
        M += q @ q.T
    eigvals, eigvecs = np.linalg.eigh(M)
    avg_quat = eigvecs[:, np.argmax(eigvals)]
    return avg_quat / np.linalg.norm(avg_quat)

def angular_distance_deg(q1, q2):
    """두 쿼터니언 간의 각도 차이를 도(degree) 단위로 계산합니다."""
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    relative_rot = r1.inv() * r2
    return np.rad2deg(relative_rot.magnitude())

def average_position(positions):
    """위치 벡터 배열의 평균을 계산합니다."""
    return np.mean(positions, axis=0)


# -------------------- 단계 1: ArUco 데이터 보정 (Averaging & Outlier Removal) --------------------

def correct_aruco_data():
    """
    여러 프레임에서 감지된 ArUco 마커의 위치와 회전값의 평균을 계산합니다.
    회전값의 이상치(outlier)를 제거한 후, 보정된 데이터를 JSON 파일로 저장합니다.
    """
    print("--- 단계 1: ArUco 데이터 보정 시작 ---")
    os.makedirs(CORRECTED_ARUCO_DIR, exist_ok=True)
    
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # 데이터 로드
    for fname in os.listdir(RAW_ARUCO_DIR):
        if not fname.endswith('.json'):
            continue
        view, cam = parse_filename(fname)
        with open(os.path.join(RAW_ARUCO_DIR, fname), 'r') as f:
            content = json.load(f)
            for marker_id, marker_data in content.items():
                data[view][cam][marker_id].append(marker_data)

    # 마커 데이터 처리 및 저장
    for view in data:
        for cam in data[view]:
            corrected = {}
            for marker_id, entries in data[view][cam].items():
                if len(entries) < 2:
                    continue  # 최소 2개 이상의 데이터가 있어야 평균 계산

                positions = [np.array([m['position_m']['x'], m['position_m']['y'], m['position_m']['z']]) for m in entries]
                quaternions = [np.array([m['rotation_quat']['x'], m['rotation_quat']['y'], m['rotation_quat']['z'], m['rotation_quat']['w']]) for m in entries]

                # 초기 평균 회전값 계산
                avg_quat_initial = average_quaternion(np.array(quaternions))

                # 이상치 제거 (평균에서 1.0도 이상 차이나는 데이터)
                filtered_positions = []
                filtered_quaternions = []
                for pos, quat in zip(positions, quaternions):
                    if angular_distance_deg(avg_quat_initial, quat) <= 1.0:
                        filtered_positions.append(pos)
                        filtered_quaternions.append(quat)
                    else:
                        print(f"이상치 감지 및 제거: 마커 ID {marker_id} ({view}_{cam})")

                if not filtered_positions:
                    continue

                # 이상치 제거 후 최종 평균 계산
                avg_pos = average_position(filtered_positions)
                avg_quat = average_quaternion(np.array(filtered_quaternions))

                corrected[marker_id] = {
                    "position_m": {"x": float(avg_pos[0]), "y": float(avg_pos[1]), "z": float(avg_pos[2])},
                    "rotation_quat": {"x": float(avg_quat[0]), "y": float(avg_quat[1]), "z": float(avg_quat[2]), "w": float(avg_quat[3])},
                    "corners_pixel": entries[0]["corners_pixel"]
                }
            
            # 보정된 데이터 파일로 저장
            output_path = os.path.join(CORRECTED_ARUCO_DIR, f"{view}_{cam}_corrected.json")
            with open(output_path, 'w') as f:
                json.dump(corrected, f, indent=4)

    print(f"보정된 ArUco 데이터가 '{CORRECTED_ARUCO_DIR}'에 저장되었습니다.\n")


# -------------------- 단계 2: 카메라 캘리브레이션 정보 추출 --------------------

def extract_camera_calibration():
    """
    ZED 카메라의 .conf 파일에서 FHD 해상도 기준 카메라 매트릭스와 왜곡 계수를 추출하여
    JSON 파일로 저장합니다.
    """
    print("--- 단계 2: 카메라 캘리브레이션 정보 추출 시작 ---")
    os.makedirs(CALIB_JSON_DIR, exist_ok=True)

    def load_fhd_calibration(conf_path, side):
        config = configparser.ConfigParser()
        config.read(conf_path)
        section = f"{side.upper()}_CAM_FHD"
        cam = config[section]
        
        fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
        k1, k2, p1, p2, k3 = float(cam["k1"]), float(cam["k2"]), float(cam["p1"]), float(cam["p2"]), float(cam["k3"])

        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], [k1, k2, p1, p2, k3]

    for position, serial in CAMERA_SERIALS.items():
        conf_path = os.path.join(CALIB_CONF_DIR, f"SN{serial}.conf")
        if not os.path.exists(conf_path):
            print(f"경고: [{position}] 설정 파일 없음: {conf_path}")
            continue

        for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
            try:
                cam_matrix, dist_coeffs = load_fhd_calibration(conf_path, side)
                data = {"camera_matrix": cam_matrix, "distortion_coeffs": dist_coeffs}
                
                filename = f"{position}_{serial}_{side_name}_calib.json"
                output_path = os.path.join(CALIB_JSON_DIR, filename)
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"[{position}/{side_name}] 캘리브레이션 저장 완료: {filename}")
            except Exception as e:
                print(f"오류: [{position}/{side_name}] 처리 중 오류 발생: {e}")
    
    print(f"카메라 캘리브레이션 파일이 '{CALIB_JSON_DIR}'에 저장되었습니다.\n")


# -------------------- 단계 3: 평균 포즈 계산 및 시각화/저장 --------------------

def calculate_and_visualize_poses():
    """
    보정된 ArUco 데이터와 카메라 캘리브레이션 정보를 사용하여 각 카메라 뷰에 대한
    객체의 평균 위치(tvec)와 회전(rvec)을 계산합니다.
    결과를 이미지에 시각화하여 파일로 저장하고, 모든 뷰의 결과를 요약하여 JSON 파일로 저장합니다.
    """
    print("--- 단계 3: 평균 포즈 계산 및 결과 시각화 시작 ---")
    os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    
    summary = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_marker = 0.6
    thickness_marker = 2
    
    for view in VIEWS:
        serial = CAMERA_SERIALS[view]
        for cam in CAMS:
            pose_path = os.path.join(CORRECTED_ARUCO_DIR, f"{view}_{cam}_corrected.json")
            calib_path = os.path.join(CALIB_JSON_DIR, f"{view}_{serial}_{cam}_calib.json")

            if not (os.path.exists(pose_path) and os.path.exists(calib_path)):
                continue

            with open(pose_path) as f: poses = json.load(f)
            with open(calib_path) as f: calib = json.load(f)

            K = np.array(calib["camera_matrix"], dtype=np.float64)
            dist = np.array(calib["distortion_coeffs"], dtype=np.float64)
            dist_coeffs_for_calc = np.zeros((5, 1))
            
            detected_marker_ids = set(poses.keys())
            offset_marker_ids = set(MARKER_OFFSETS[view].keys())
            
            markers_to_use = detected_marker_ids.intersection(offset_marker_ids)
            ignored_markers = detected_marker_ids - offset_marker_ids
            
            print(f"\nProcessing [{view}/{cam}]:")
            print(f"  - Detected Markers: {sorted(list(detected_marker_ids))}")
            if ignored_markers:
                print(f"  - Ignored Markers (No Offset Defined): {sorted(list(ignored_markers))}")
            if not markers_to_use:
                print("  - No valid markers with defined offsets found. Skipping.")
                continue
            print(f"  - Markers to be Used for Calculation: {sorted(list(markers_to_use))}")

            
            tvecs, quats, marker_ids = [], [], []
            for mid in markers_to_use:
                offset = MARKER_OFFSETS[view][mid]
                p = poses[mid]
                tvec = np.array([p["position_m"]["x"], p["position_m"]["y"], p["position_m"]["z"]])
                quat = np.array([p["rotation_quat"][k] for k in ("x", "y", "z", "w")])
                
                # 오프셋을 적용하여 기준 좌표계의 위치 계산
                Rm = R.from_quat(quat).as_matrix()
                tvec_with_offset = tvec + Rm @ offset
                
                tvecs.append(tvec_with_offset)
                quats.append(quat)
                marker_ids.append(mid)

            if not tvecs:
                continue

            # 모든 마커의 tvec과 rvec을 평균내어 최종 포즈 계산
            mean_tvec = np.mean(tvecs, axis=0)
            mean_quat = average_quaternion(quats)
            mean_rvec = R.from_quat(mean_quat).as_rotvec()
            
            # --- 시각화 ---
            img_files = glob.glob(os.path.join(IMAGE_SOURCE_DIR, f"{view}_*_{cam}_*.png"))
            if not img_files:
                continue
            
            img_bgr = cv2.imread(img_files[0])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 개별 마커의 좌표축 그리기
            for quat, tvec, mid in zip(quats, tvecs, marker_ids):
                rot_vec = R.from_quat(quat).as_rotvec()
                cv2.drawFrameAxes(img_rgb, K, dist_coeffs_for_calc, rot_vec, tvec.reshape(3, 1), 0.05)
                marker_pos_2d, _ = cv2.projectPoints(tvec.reshape(1, 3), np.zeros((3,1)), np.zeros((3,1)), K, dist_coeffs_for_calc)
                x_marker, y_marker = marker_pos_2d.ravel().astype(int)
                cv2.putText(img_rgb, f"ID:{mid}", (x_marker + 10, y_marker - 10), font, font_scale_marker, (255, 255, 0), thickness_marker)
            
            # 평균 좌표계 시각화 (더 크게)
            cv2.drawFrameAxes(img_rgb, K, dist_coeffs_for_calc, mean_rvec.reshape(3, 1), mean_tvec.reshape(3, 1), 0.05)
            
            # 평균 위치 텍스트 표시
            mean_pos_2d, _ = cv2.projectPoints(mean_tvec.reshape(1,3), np.zeros(3), np.zeros(3), K, dist_coeffs_for_calc)
            xm, ym = map(int, mean_pos_2d.ravel())
            cv2.drawMarker(img_rgb, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            mean_rvec_deg = np.rad2deg(mean_rvec)
            rvec_text = f"Mean Rvec [deg]: {mean_rvec_deg[0]:.1f}, {mean_rvec_deg[1]:.1f}, {mean_rvec_deg[2]:.1f}"
            tvec_text = f"Mean Tvec [m]: {mean_tvec[0]:.3f}, {mean_tvec[1]:.3f}, {mean_tvec[2]:.3f}"
            cv2.putText(img_rgb, rvec_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img_rgb, tvec_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Matplotlib을 사용하여 이미지 저장
            plt.figure(figsize=(12, 9))
            plt.imshow(img_rgb)
            plt.title(f"Pose Estimation Result: {view.upper()} - {cam}", fontsize=16)
            plt.axis('off')
            
            # 요청하신 경로에 이미지 저장
            output_image_path = os.path.join(RESULT_IMAGE_DIR, f"{view}_{cam}_pose_visualization.png")
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
            plt.close() # 메모리 누수 방지를 위해 figure를 닫아줍니다.
            print(f"결과 이미지 저장 완료: {output_image_path}")

            # 요약 정보 추가
            summary.append([view, cam, *mean_tvec, *mean_rvec, xm, ym])
            print(f"{marker_ids}, {tvecs}, {quats}")
            
    # 최종 결과 요약 저장
    columns = ["view", "cam", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z", 'projected_x', 'projected_y']
    df = pd.DataFrame(summary, columns=columns)
    summary_path = "./Fr5/Fr5_aruco_pose_summary.json"
    df.to_json(summary_path, orient="records", indent=2)

    print(f"\n최종 포즈 요약 파일이 '{summary_path}'에 저장되었습니다.")


# -------------------- 메인 실행 블록 (Main Execution) --------------------

if __name__ == "__main__":
    # 1단계: ArUco 데이터 보정
    correct_aruco_data()
    
    # 2단계: 카메라 캘리브레이션 정보 추출
    extract_camera_calibration()
    
    # 3단계: 최종 포즈 계산 및 결과 저장
    calculate_and_visualize_poses()
    
    print("\n--- 모든 작업이 완료되었습니다. ---")