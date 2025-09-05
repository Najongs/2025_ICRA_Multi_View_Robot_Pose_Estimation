import os
import glob
import json
import pandas as pd
from tqdm import tqdm

# --- ⚙️ 1. 설정 변수 ---

DATA_PATHS = [
    '../dataset/DREAM_real/panda-3cam_azure/panda-3cam_azure',
    '../dataset/DREAM_real/panda-3cam_kinect360/panda-3cam_kinect360',
    '../dataset/DREAM_real/panda-3cam_realsense/panda-3cam_realsense',
    '../dataset/DREAM_real/panda-orb/panda-orb',
]

# 추출할 키포인트와 조인트의 이름을 미리 정의합니다.
REQUIRED_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
REQUIRED_KEYPOINTS = ['panda_link0', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link6', 'panda_link7', 'panda_hand']


# --- 🛠️ 2. 단일 폴더 처리 함수 ---

def process_single_directory(base_path):
    """
    하나의 데이터 폴더에서 이미지, 조인트 각도, 키포인트 데이터를 매칭하여 CSV로 저장합니다.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Processing directory: {base_path}")
    print(f"{'='*60}")
    
    json_files = glob.glob(os.path.join(base_path, '*.json'))
    
    if not json_files:
        print(f"⚠️  Warning: No JSON files found in {base_path}. Skipping.")
        return

    print(f"✅ Found {len(json_files)} JSON files. Starting data matching...")

    all_records = []
    for json_path in tqdm(json_files, desc=f"Matching {os.path.basename(base_path)}"):
        try:
            base_name_without_ext = os.path.splitext(os.path.basename(json_path))[0]
            # ✅ 이미지 파일 이름 형식을 '.rgb.jpg'로 수정
            image_path = os.path.join(base_path, f"{base_name_without_ext}.rgb.jpg")

            if os.path.exists(image_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # --- 데이터 유효성 검사 ---
                # 1. 조인트 데이터가 올바른지 확인
                if not ('sim_state' in data and 'joints' in data['sim_state']):
                    continue
                joint_data = {joint['name']: joint['position'] for joint in data['sim_state']['joints']}
                if not all(name in joint_data for name in REQUIRED_JOINTS):
                    continue # 필요한 조인트가 하나라도 없으면 건너뛰기

                # 2. 키포인트 데이터가 올바른지 확인
                if not (data.get('objects') and isinstance(data['objects'], list) and len(data['objects']) > 0 and 'keypoints' in data['objects'][0]):
                    continue
                keypoints_data = {kp['name']: kp for kp in data['objects'][0]['keypoints']}
                if not all(name in keypoints_data for name in REQUIRED_KEYPOINTS):
                    continue # 필요한 키포인트가 하나라도 없으면 건너뛰기

                # --- 레코드 생성 ---
                record = {'image_path': image_path}
                
                # 1. 조인트 각도 데이터 추가
                for name in REQUIRED_JOINTS:
                    joint_num = name.replace('panda_joint', '')
                    record[f'joint_{joint_num}'] = joint_data[name]
                
                # 2. 키포인트 3D 위치 및 2D 투영 위치 데이터 추가
                for name in REQUIRED_KEYPOINTS:
                    keypoint = keypoints_data[name]
                    # 3D Location
                    record[f'kpt_{name}_loc_x'] = keypoint['location'][0]
                    record[f'kpt_{name}_loc_y'] = keypoint['location'][1]
                    record[f'kpt_{name}_loc_z'] = keypoint['location'][2]
                    # 2D Projected Location
                    record[f'kpt_{name}_proj_x'] = keypoint['projected_location'][0]
                    record[f'kpt_{name}_proj_y'] = keypoint['projected_location'][1]
                
                all_records.append(record)
        
        except Exception as e:
            print(f"\n⚠️  Error processing file {json_path}: {e}")

    if not all_records:
        print("❌ No matching data found in this directory.")
        return
        
    df = pd.DataFrame(all_records)
    output_csv_path = f"{base_path}_matched_data.csv"
    df.to_csv(output_csv_path, index=False)
    
    print("\n--- ✨ Directory processing complete ---")
    print(f"✅ Matched {len(df)} data pairs successfully.")
    print(f"✅ Results saved to: {output_csv_path}")
    print("\n--- Data Sample (first 5 rows, selected columns) ---")
    # 너무 길어지므로 일부 컬럼만 샘플로 출력
    sample_columns = ['image_path', 'joint_1', 'kpt_panda_hand_proj_x', 'kpt_panda_hand_proj_y']
    print(df[sample_columns].head())
    print("---------------------------------------------------\n")

# --- 🚀 3. 메인 실행부 ---
def main():
    for path in DATA_PATHS:
        process_single_directory(path)

# --- 4. 스크립트 실행 ---
if __name__ == '__main__':
    main()