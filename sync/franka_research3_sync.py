import os
import re
import yaml
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- ⚙️ 1. 설정 변수 ---

# 데이터 소스 경로
IMAGE_BASE_DIRS = [
    "../dataset/franka_research3/franka_research3_pose1",
    "../dataset/franka_research3/franka_research3_pose2"
]
JOINT_DATA_PATH = "../dataset/franka_research3/franka_research3_Joint_Angle"

# 최종 동기화 결과가 저장될 경로 및 파일명
OUTPUT_SYNC_CSV_PATH = "../dataset/franka_research3/fr3_matched_joint_angle.csv"

# 동기화 최대 허용 시간 차이 (초 단위)
MAX_TIME_DIFFERENCE_THRESHOLD = 0.05

# --- ✨ 수정된 부분: 이미지 타임스탬프에 더해줄 고정 딜레이 값 ---
IMAGE_TIMESTAMP_DELAY = 0.0333

# --- 🛠️ 2. 헬퍼 함수 ---

def process_yaml_to_df_records(yaml_path):
    """하나의 YAML 파일을 읽어 데이터 레코드(딕셔너리)의 리스트를 반환합니다."""
    records = []
    with open(yaml_path, 'r') as f:
        try:
            all_docs = list(yaml.safe_load_all(f))
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {yaml_path}: {e}")
            return []

    for doc in all_docs:
        if not doc: continue
        
        record = {}
        stamp = doc.get('header', {}).get('stamp', {})
        sec = stamp.get('sec', 0)
        nanosec = stamp.get('nanosec', 0)
        record['timestamp'] = float(f"{sec}.{nanosec:09d}"[:14])

        joint_names = doc.get('name', [])
        positions = doc.get('position', [])
        velocities = doc.get('velocity', [])
        efforts = doc.get('effort', [])

        for i, name in enumerate(joint_names):
            record[f'position_{name}'] = positions[i] if i < len(positions) else np.nan
            record[f'velocity_{name}'] = velocities[i] if i < len(velocities) else np.nan
            record[f'effort_{name}'] = efforts[i] if i < len(efforts) else np.nan
        
        records.append(record)
    return records

def find_image_files(base_dirs):
    """지정된 모든 상위 디렉토리에서 이미지 파일을 재귀적으로 찾습니다."""
    image_files = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, f))
    return image_files

def parse_image_timestamp(image_path):
    """이미지 파일명에서 타임스탬프를 float으로 추출합니다."""
    try:
        filename = os.path.basename(image_path)
        parts = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '').split('_')
        timestamp_str = parts[-1]
        return float(timestamp_str)
    except (IndexError, ValueError):
        return None

# --- 🚀 3. 메인 실행 함수 ---

def create_synchronized_dataset():
    """
    모든 로봇 데이터(YAML)와 이미지 데이터를 로드하고,
    타임스탬프를 기준으로 동기화하여 하나의 CSV 파일로 저장합니다.
    """
    
    # --- 단계 1: 모든 YAML 파일 로드 및 단일 로봇 데이터프레임 생성 ---
    print("--- 단계 1: 모든 로봇 관절 데이터(YAML) 로딩 및 통합 ---")
    all_joint_paths = glob.glob(os.path.join(JOINT_DATA_PATH, "joint_states_*.yaml"))
    
    if not all_joint_paths:
        print(f"❌ 에러: '{JOINT_DATA_PATH}' 경로에 YAML 파일이 없습니다.")
        return

    all_robot_records = []
    for yaml_path in tqdm(all_joint_paths, desc="YAML 파일 처리 중"):
        all_robot_records.extend(process_yaml_to_df_records(yaml_path))
        
    df_robot = pd.DataFrame(all_robot_records)
    df_robot.sort_values('timestamp', inplace=True, ignore_index=True)
    
    print(f"✅ 총 {len(df_robot)}개의 로봇 데이터 포인트를 {len(all_joint_paths)}개 파일로부터 통합했습니다.\n")

    # --- 단계 2: 모든 이미지 파일 경로 스캔 ---
    print("--- 단계 2: 모든 이미지 파일 스캔 ---")
    image_paths = find_image_files(IMAGE_BASE_DIRS)
    print(f"✅ 총 {len(image_paths)}개의 이미지 파일을 찾았습니다.\n")

    # --- 단계 3: 이미지와 로봇 데이터 타임스탬프 기준 동기화 ---
    print("--- 단계 3: 이미지와 로봇 데이터 동기화 (이미지 딜레이 +{IMAGE_TIMESTAMP_DELAY}초 적용) ---")
    synchronized_records = []
    robot_timestamps = df_robot['timestamp'].values

    for image_path in tqdm(image_paths, desc="이미지 매칭 중"):
        img_ts = parse_image_timestamp(image_path)
        if img_ts is None:
            continue

        # --- ✨ 수정된 부분: 이미지 타임스탬프에 딜레이를 더하여 매칭에 사용 ---
        img_ts_for_matching = img_ts + IMAGE_TIMESTAMP_DELAY

        # 가장 가까운 타임스탬프의 인덱스 찾기 (딜레이가 적용된 타임스탬프 기준)
        time_diffs = np.abs(robot_timestamps - img_ts_for_matching)
        closest_idx = np.argmin(time_diffs)
        min_time_diff = time_diffs[closest_idx]

        # 시간 차이가 설정된 임계값 이내인지 확인
        if min_time_diff < MAX_TIME_DIFFERENCE_THRESHOLD:
            matching_robot_row = df_robot.iloc[closest_idx]
            
            record = {
                'image_path': image_path,
                'image_timestamp': img_ts, # CSV에는 원본 타임스탬프 기록
                'robot_timestamp': matching_robot_row['timestamp'],
                'time_difference_s': min_time_diff # 딜레이가 반영된 시간 차이
            }
            record.update(matching_robot_row.to_dict())
            
            synchronized_records.append(record)

    # --- 단계 4: 최종 결과 저장 ---
    if not synchronized_records:
        print("\n❌ 매칭된 데이터가 없습니다. 결과 파일이 생성되지 않았습니다.")
        return
        
    df_sync = pd.DataFrame(synchronized_records)
    df_sync.sort_values('image_timestamp', inplace=True, ignore_index=True)
    
    output_dir = os.path.dirname(OUTPUT_SYNC_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    df_sync.to_csv(OUTPUT_SYNC_CSV_PATH, index=False)
    
    print("\n\n--- 🎉 동기화 완료 ---")
    print(f"✅ 총 {len(df_sync)}개의 이미지-로봇 쌍이 성공적으로 동기화되었습니다.")
    print(f"✅ 결과 저장 경로: {OUTPUT_SYNC_CSV_PATH}")
    print("\n--- 동기화 데이터 샘플 (딜레이 적용 후) ---")
    
    # --- ✨ 수정된 부분: 샘플 출력에 타임스탬프 정보를 추가하여 확인 용이하게 변경 ---
    sample_cols = ['image_timestamp', 'robot_timestamp', 'time_difference_s', 'position_fr3_joint1']
    display_cols = [col for col in sample_cols if col in df_sync.columns]
    print(df_sync[display_cols].head())


if __name__ == '__main__':
    create_synchronized_dataset()