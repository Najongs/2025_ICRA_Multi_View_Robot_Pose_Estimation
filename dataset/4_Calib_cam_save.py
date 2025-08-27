import os
import json
import configparser

# 시리얼 ↔ 위치 매핑
camera_list = {
    41182735: "front", # front # marker 1, 2, 4, 5, 6
    49429257: "right", # right # marker 2, 4, 7, 8
    44377151: "left", # left  # marker 3, 5, 7, 8
    49045152: "top"  # top   # marker 1, 2, 4, 6, 7, 8
}


zed_conf_dir = "./All_camera_conf"
output_dir = "./Meca_insertion/Meca_calib_cam_from_conf"
os.makedirs(output_dir, exist_ok=True)
def load_fhd_calibration(conf_path, side):
    config = configparser.ConfigParser()

    # UTF-8 BOM 제거하여 파일 읽기
    with open(conf_path, "r", encoding="utf-8-sig") as f:
        config.read_file(f)

    section = f"{side.upper()}_CAM_FHD1200"
    adv_section = f"{side.upper()}_DISTO"

    cam = config[section]

    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])
    k1 = float(cam["k1"])
    k2 = float(cam["k2"])
    p1 = float(cam["p1"])
    p2 = float(cam["p2"])
    k3 = float(cam["k3"])

    camera_matrix = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]
    distortion_coeffs = [k1, k2, p1, p2, k3]

    adv_dist = {}
    if adv_section in config:
        adv = config[adv_section]
        for key in adv:
            adv_dist[key] = float(adv[key])

    return camera_matrix, distortion_coeffs, adv_dist


# 전체 처리
for serial, position in camera_list.items():
    conf_path = os.path.join(zed_conf_dir, f"SN{serial}.conf")
    
    if not os.path.exists(conf_path):
        print(f"[{position}] 설정 파일 없음: {conf_path}")
        continue

    for side, side_name in [("LEFT", "leftcam"), ("RIGHT", "rightcam")]:
        try:
            cam_matrix, dist_coeffs, adv_dist = load_fhd_calibration(conf_path, side)

            data = {
                "camera_matrix": cam_matrix,
                "distortion_coeffs": dist_coeffs,
                "advanced_distortion": adv_dist
            }

            filename = f"{position}_{serial}_{side_name}_calib.json"
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(data, f, indent=4)
            print(f"[{position}] 저장 완료: {filename} (distortion: {dist_coeffs})")

        except Exception as e:
            print(f"[{position}] {side_name} 처리 중 오류: {e}")