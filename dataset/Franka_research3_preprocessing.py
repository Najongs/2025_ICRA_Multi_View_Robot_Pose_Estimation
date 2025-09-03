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

# -------------------- ⚙️ 1. Global Configuration --------------------

# --- Base Paths ---
BASE_DIR = "./franka_research3"
ALL_CAMERA_CONF_DIR = "./All_camera_conf"

# --- Pose 1 Paths ---
POSE1_INPUT_BASE = os.path.join(BASE_DIR, "franka_research3_ArUco_pose1")
POSE1_CORRECTED_DIR = os.path.join(BASE_DIR, "franka_research3_correct_ArUco_pose1")

# --- Pose 2 Paths ---
POSE2_INPUT_BASE = os.path.join(BASE_DIR, "franka_research3_ArUco_pose2")
POSE2_CORRECTED_DIR = os.path.join(BASE_DIR, "franka_research3_correct_ArUco_pose2")

# --- Common Paths ---
CALIB_JSON_DIR = os.path.join(BASE_DIR, "Calib_cam_from_conf")
RESULT_IMAGE_DIR = os.path.join(BASE_DIR, "franka_research3_calib_results_images")

# --- Camera Settings ---
CAMERA_SERIALS = {
    41182735: "view1",
    49429257: "view2",
    44377151: "view3",
    49045152: "view4"
}
SERIAL_MAP = {v: k for k, v in CAMERA_SERIALS.items()}
VIEWS = ['view1', 'view2', 'view3', 'view4']
CAMS = ['leftcam', 'rightcam']

# --- Marker Offset Settings ---
MARKER_OFFSETS = {
    "view1": {"2": [-0.175, 0.0, -0.045], "4": [-0.30, 0.0, -0.045], "6": [0.025, 0.325, -0.045]},
    "view2": {"2": [-0.15, 0.0, -0.1], "4": [-0.275, 0.0, -0.1], "7": [0.05, -0.225, -0.1], "8": [0.05, -0.325, -0.1]},
    "view3": {"3": [0.225, 0.05, -0.045],  "5": [0.35, 0.05, -0.045],  "7": [0.025, -0.175, -0.045], "8": [0.025, -0.275, -0.045]},
    "view4": {"2": [-0.175, 0.025, -0.045], "4": [-0.30, 0.025, -0.045],"8": [0.025, -0.3, -0.045]}
}
# Convert offsets to numpy arrays
for view_key, markers in MARKER_OFFSETS.items():
    for marker_key, offset in markers.items():
        MARKER_OFFSETS[view_key][marker_key] = np.array(offset)

# -------------------- Helper Functions --------------------
def parse_filename(filename):
    parts = filename.split('_')
    # Assumes format like "view1_41182735_left_...json"
    return parts[0], parts[2]

def average_quaternion(quaternions):
    if len(quaternions) == 0: return None
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

# -------------------- 📜 Function 1: Correct ArUco Data --------------------
def correct_aruco_data(input_base_dir, output_dir):
    print(f"--- [Starting ArUco Data Correction] ---")
    print(f"Input: {input_base_dir}\nOutput: {output_dir}")
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
            # ✨ [DEBUG] Print which view/cam is being processed
            print(f"  [DEBUG] Correcting data for {view}/{cam}...")
            corrected = {}
            for marker_id, entries in markers_data.items():
                if len(entries) < 2: continue
                
                # ✨ [DEBUG] Print initial detection count
                print(f"    [DEBUG]   - Marker ID {marker_id}: Found {len(entries)} initial detections.")
                
                positions = [np.array([m['position_m'][k] for k in 'xyz']) for m in entries]
                quaternions = np.array([np.array([m['rotation_quat'][k] for k in 'xyzw']) for m in entries])

                # 💡 1. 쿼터니언을 맨 처음에 한 번만 정렬합니다.
                aligned_quats = align_quaternions(quaternions)
                
                # ✨ [DEBUG] Print initial detection count
                print(f" [DEBUG]  - Marker ID {marker_id}: Found {len(entries)} initial detections.")
                
                # 💡 2. 정렬된 쿼터니언으로 초기 평균을 계산합니다.
                avg_quat_initial = average_quaternion(aligned_quats)
                if avg_quat_initial is None: continue

                # 💡 3. 필터링 시에도 정렬된 쿼터니언을 사용합니다. (일관성 유지)
                #    (angular_distance_deg 함수는 q와 -q를 동일하게 처리하므로 원본을 써도 무방하지만,
                #     코드를 명확하게 하기 위해 정렬된 버전을 사용하는 것이 좋습니다.)
                filtered_indices = [i for i, q in enumerate(aligned_quats) if angular_distance_deg(avg_quat_initial, q) <= 2.0]
                
                if not filtered_indices:
                    print(f" [DEBUG]   - Marker ID {marker_id}: No detections kept after filtering.")
                    continue
                    
                filtered_pos = [positions[i] for i in filtered_indices]
                filtered_quat = [aligned_quats[i] for i in filtered_indices]

                # ✨ [DEBUG] Print detection count after filtering
                print(f" [DEBUG] - Marker ID {marker_id}: Kept {len(filtered_pos)} detections after filtering (threshold: 2.0 deg).")

                # 💡 4. 이미 정렬된 필터링 결과를 사용하므로, 다시 정렬할 필요가 없습니다.
                avg_pos = average_position(filtered_pos)
                avg_quat = average_quaternion(np.array(filtered_quat))
                
                if avg_quat is None: continue

                corrected[marker_id] = {
                    "position_m": {k: float(v) for k, v in zip('xyz', avg_pos)},
                    "rotation_quat": {k: float(v) for k, v in zip('xyzw', avg_quat)},
                    "corners_pixel": entries[0]["corners_pixel"]
                }
            
            if corrected:
                output_path = os.path.join(output_dir, f"{view}_{cam}_corrected.json")
                with open(output_path, 'w') as f: json.dump(corrected, f, indent=4)
    print(f"--- [ArUco Data Correction Complete] ---\n")

# -------------------- 📜 Function 2: Extract Camera Calibration --------------------
def extract_camera_calibration():
    print(f"--- [Starting Camera Calibration Extraction] ---")
    os.makedirs(CALIB_JSON_DIR, exist_ok=True)
    
    def load_fhd_calibration(conf_path, side):
        config = configparser.ConfigParser()
        with open(conf_path, "r", encoding="utf-8-sig") as f: config.read_file(f)
        section = f"{side.upper()}_CAM_FHD1200" # Section name in .conf file
        cam = config[section]
        cam_matrix = [[float(cam["fx"]), 0.0, float(cam["cx"])], [0.0, float(cam["fy"]), float(cam["cy"])], [0.0, 0.0, 1.0]]
        dist_coeffs = [float(cam[k]) for k in ["k1", "k2", "p1", "p2", "k3"]]
        return cam_matrix, dist_coeffs

    for serial, position in CAMERA_SERIALS.items():
        conf_path = os.path.join(ALL_CAMERA_CONF_DIR, f"SN{serial}.conf")
        if not os.path.exists(conf_path):
            print(f"Warning: [{position}] Config file not found: {conf_path}")
            continue
        for side, side_name in [("LEFT", "left"), ("RIGHT", "right")]:
            try:
                cam_matrix, dist_coeffs = load_fhd_calibration(conf_path, side)
                data = {"camera_matrix": cam_matrix, "distortion_coeffs": dist_coeffs}
                filename = f"{position}_{serial}_{side_name}cam_calib.json"
                with open(os.path.join(CALIB_JSON_DIR, filename), "w") as f: json.dump(data, f, indent=4)
                print(f"Saved: [{position}] {filename}")
            except Exception as e:
                print(f"Error processing [{position}] {side_name}: {e}")
    print(f"--- [Camera Calibration Extraction Complete] ---\n")

# -------------------- 📜 Function 3: Calculate and Visualize Poses --------------------
def calculate_and_visualize_poses(pose_name, corrected_json_dir, image_base_dir):
    print(f"--- [Starting Final Pose Calculation & Visualization: {pose_name.upper()}] ---")
    os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)
    
    summary = []
    dist_coeffs_for_calc = np.zeros((5, 1)) # Use zero distortion for drawing 3D axes
    for view in VIEWS:
        serial = SERIAL_MAP[view]
        for cam in CAMS:
            pose_path = os.path.join(corrected_json_dir, f"{view}_{cam}_corrected.json")
            calib_path = os.path.join(CALIB_JSON_DIR, f"{view}_{serial}_{cam}_calib.json")
            
            # ✨ [DEBUG] Check for necessary input files
            if not os.path.exists(pose_path):
                print(f"  [DEBUG] Skipping {view}/{cam}: Corrected pose file not found at {pose_path}")
                continue
            if not os.path.exists(calib_path):
                print(f"  [DEBUG] Skipping {view}/{cam}: Calibration file not found at {calib_path}")
                continue
            print(f"  [DEBUG] Processing {view}/{cam}...")

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
            
            # ✨ [DEBUG] Print which markers are being used for the average
            print(f"    [DEBUG]   - Using data from markers {marker_ids} to calculate average pose.")

            if not tvecs: continue
            
            mean_tvec = average_position(np.array(tvecs))
            aligned_quats = align_quaternions(np.array(quats))
            mean_quat = average_quaternion(aligned_quats)

            # mean_quat이 None이 아닌 경우에만 rvec으로 변환 (오류 방지)
            if mean_quat is not None:
                mean_rvec = R.from_quat(mean_quat).as_rotvec()
            else:
                # 쿼터니언 평균 계산 실패 시, 처리를 건너뛰거나 기본값을 사용
                print(f"    [WARNING] Could not compute average quaternion for {view}/{cam}. Skipping.")
                continue
            
            # --- Visualization ---
            image_dir_list = glob.glob(os.path.join(image_base_dir, "ArUco_capture_dataset_*"))
            if not image_dir_list: continue
            image_dir = image_dir_list[0]
            
            img_files = glob.glob(os.path.join(image_dir, f"{view}_*_{cam}_*.png"))
            # ✨ [DEBUG] Check if an image was found for visualization
            if not img_files:
                print(f"    [WARNING] No visualization image found for {view}/{cam}. Skipping visualization.")
                continue
            
            img_rgb = cv2.cvtColor(cv2.imread(img_files[0]), cv2.COLOR_BGR2RGB)
            
            # Visualize individual marker poses (with offset)
            for (quat, tvec, mid) in zip(quats, tvecs, marker_ids):
                rot_vec = R.from_quat(quat).as_rotvec()
                cv2.drawFrameAxes(img_rgb, K, dist_coeffs_for_calc, rot_vec, tvec.reshape(3, 1), 0.05)
                # Project center point to put text label
                marker_pos_2d, _ = cv2.projectPoints(tvec.reshape(1, 3), np.zeros(3), np.zeros(3), K, dist_coeffs_for_calc)
                x, y = marker_pos_2d.ravel().astype(int)
                cv2.putText(img_rgb, f"ID:{mid}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Visualize the final averaged pose
            cv2.drawFrameAxes(img_rgb, K, dist_coeffs_for_calc, mean_rvec, mean_tvec, 0.1, 3) # Thicker and larger
            proj_point, _ = cv2.projectPoints(mean_tvec.reshape(1,3), np.zeros(3), np.zeros(3), K, dist_coeffs_for_calc)
            xm, ym = proj_point.ravel().astype(int)
            cv2.drawMarker(img_rgb, (xm, ym), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)

            # --- Save Image to File ---
            plt.figure(figsize=(16, 9))
            plt.imshow(img_rgb)
            plt.title(f"[{pose_name.upper()}] {view.upper()} - {cam}", fontsize=16)
            plt.axis('off')
            
            output_image_path = os.path.join(RESULT_IMAGE_DIR, f"{pose_name}_{view}_{cam}_visualization.png")
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print(f"  Image saved: {output_image_path}")

            summary.append([view, cam, *mean_tvec, *mean_rvec, xm, ym])
    
    # --- Save Summary File ---
    columns = ["view", "cam", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z", "proj_x", "proj_y"]
    df = pd.DataFrame(summary, columns=columns)
    summary_path = f"./franka_research3/{pose_name}_aruco_pose_summary.json"
    df.to_json(summary_path, orient="records", indent=2)
    print(f"Summary file saved: {summary_path}")
    print(f"--- [Final Pose Calculation & Visualization Complete: {pose_name.upper()}] ---\n")

# -------------------- ▶️ Main Execution --------------------
if __name__ == "__main__":
    # Step 1: Extract camera calibration info (common process, run once)
    extract_camera_calibration()

    # Step 2: Process and visualize Pose 1 data
    correct_aruco_data(POSE1_INPUT_BASE, POSE1_CORRECTED_DIR)
    calculate_and_visualize_poses("pose1", POSE1_CORRECTED_DIR, POSE1_INPUT_BASE)

    # Step 3: Process and visualize Pose 2 data
    correct_aruco_data(POSE2_INPUT_BASE, POSE2_CORRECTED_DIR)
    calculate_and_visualize_poses("pose2", POSE2_CORRECTED_DIR, POSE2_INPUT_BASE)

    print("🎉 --- All tasks are complete. --- 🎉")