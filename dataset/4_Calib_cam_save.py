import os
import json
import configparser

# --- ⚙️ 1. 설정 변수 ---

# 첫 번째 이름 규칙 매핑
camera_list_dir = {
    41182735: "front",
    49429257: "right",
    44377151: "left",
    49045152: "top"
}

# 두 번째 이름 규칙 매핑
camera_list_dir2 = {
    41182735: "view1",
    49429257: "view2",
    44377151: "view3",
    49045152: "view4"
}

# 입출력 경로 설정
zed_conf_dir = "./All_camera_conf"
output_dir = "./Meca_insertion/Meca_insertion_calib_cam_from_conf"
output_dir2 = "./franka_research3/franka_research_calib_cam_from_conf"

# 두 개의 출력 폴더를 모두 생성
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

# --- 🛠️ 2. 헬퍼 함수 ---

def load_fhd_calibration(conf_path, side):
    """ .conf 파일에서 지정된 side의 FHD 해상도 카메라 보정값을 읽어옵니다. """
    config = configparser.ConfigParser()
    with open(conf_path, "r", encoding="utf-8-sig") as f:
        config.read_file(f)

    section = f"{side.upper()}_CAM_FHD1200"
    adv_section = f"{side.upper()}_DISTO"
    cam = config[section]

    fx, fy = float(cam["fx"]), float(cam["fy"])
    cx, cy = float(cam["cx"]), float(cam["cy"])
    k1, k2, k3 = float(cam["k1"]), float(cam["k2"]), float(cam["k3"])
    p1, p2 = float(cam["p1"]), float(cam["p2"])

    camera_matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    distortion_coeffs = [k1, k2, p1, p2, k3]

    adv_dist = {}
    if adv_section in config:
        adv = config[adv_section]
        for key in adv:
            adv_dist[key] = float(adv[key])

    return camera_matrix, distortion_coeffs, adv_dist

# ▼▼▼ [수정] 처리 로직을 별도 함수로 분리 ▼▼▼
def process_and_save_calibrations(camera_mapping, target_dir, conf_dir):
    """
    주어진 카메라 매핑과 경로 설정에 따라 보정 파일을 처리하고 저장합니다.
    """
    print(f"\n---  processing for target directory: '{target_dir}' ---")
    
    for serial, position_name in camera_mapping.items():
        conf_path = os.path.join(conf_dir, f"SN{serial}.conf")
        
        if not os.path.exists(conf_path):
            print(f"[{position_name}] 설정 파일 없음: {conf_path}")
            continue

        for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
            try:
                cam_matrix, dist_coeffs, adv_dist = load_fhd_calibration(conf_path, side)

                data = {
                    "camera_matrix": cam_matrix,
                    "distortion_coeffs": dist_coeffs,
                    "advanced_distortion": adv_dist
                }

                # 현재 매핑에 맞는 이름(e.g., 'front' 또는 'view1')으로 파일명 생성
                filename = f"{position_name}_{serial}_{side_name}_calib.json"
                output_path = os.path.join(target_dir, filename)
                
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=4)
                
                print(f"[{position_name}] 저장 완료: {filename}")

            except Exception as e:
                print(f"[{position_name}] {side_name} 처리 중 오류: {e}")


# --- 🚀 3. 메인 실행부 ---

# 첫 번째 설정으로 실행
process_and_save_calibrations(
    camera_mapping=camera_list_dir,
    target_dir=output_dir,
    conf_dir=zed_conf_dir
)

# 두 번째 설정으로 실행
process_and_save_calibrations(
    camera_mapping=camera_list_dir2,
    target_dir=output_dir2,
    conf_dir=zed_conf_dir
)

print("\n--- ✅ All tasks completed. ---")