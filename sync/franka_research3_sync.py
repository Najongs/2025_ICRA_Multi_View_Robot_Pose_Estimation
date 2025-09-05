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
MAX_TIME_DIFFERENCE_THRESHOLD = 0.02

# 이미지 타임스탬프에 더해줄 고정 딜레이 값
IMAGE_TIMESTAMP_DELAY = 0.0333

# --- 🛠️ 2. 헬퍼 함수 (기존과 동일) ---

def process_yaml_to_df_records(yaml_path):
    """하나의 YAML 파일을 읽어 데이터 레코드(딕셔셔리)의 리스트를 반환합니다."""
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
        record['robot_timestamp'] = float(f"{sec}.{nanosec:09d}"[:14]) # ✅ 컬럼명 명확화

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
        parts = os.path.splitext(filename)[0].split('_')
        timestamp_str = parts[-1]
        return float(timestamp_str)
    except (IndexError, ValueError):
        return None

# --- 🚀 3. 메인 실행 함수 (성능 개선 버전) ---

def create_synchronized_dataset_fast():
    """
    pandas.merge_asof를 사용하여 이미지와 로봇 데이터를 초고속으로 동기화합니다.
    """
    
    # --- 단계 1: 모든 로봇 데이터(YAML) 로드 및 데이터프레임 생성 ---
    print("--- 단계 1: 모든 로봇 관절 데이터(YAML) 로딩 및 통합 ---")
    all_joint_paths = glob.glob(os.path.join(JOINT_DATA_PATH, "joint_states_*.yaml"))
    
    if not all_joint_paths:
        print(f"❌ 에러: '{JOINT_DATA_PATH}' 경로에 YAML 파일이 없습니다.")
        return

    all_robot_records = []
    for yaml_path in tqdm(all_joint_paths, desc="YAML 파일 처리 중"):
        all_robot_records.extend(process_yaml_to_df_records(yaml_path))
        
    df_robot = pd.DataFrame(all_robot_records)
    df_robot.sort_values('robot_timestamp', inplace=True, ignore_index=True)
    print(f"✅ 총 {len(df_robot)}개의 로봇 데이터 포인트를 통합했습니다.\n")

    # --- 단계 2: 모든 이미지 데이터를 데이터프레임으로 생성 ---
    print("--- 단계 2: 모든 이미지 파일 스캔 및 데이터프레임 생성 ---")
    image_paths = find_image_files(IMAGE_BASE_DIRS)
    
    image_records = []
    for path in tqdm(image_paths, desc="이미지 파일 처리 중"):
        ts = parse_image_timestamp(path)
        if ts is not None:
            image_records.append({
                'image_timestamp': ts,
                'matching_timestamp': ts + IMAGE_TIMESTAMP_DELAY, # 딜레이를 더한 매칭용 타임스탬프
                'image_path': path
            })
            
    df_image = pd.DataFrame(image_records)
    df_image.sort_values('matching_timestamp', inplace=True, ignore_index=True)
    print(f"✅ 총 {len(df_image)}개의 유효한 이미지 데이터를 통합했습니다.\n")
    total_image_count = len(df_image)
    
    # --- 단계 3: `merge_asof`를 사용한 초고속 동기화 ---
    print(f"--- 단계 3: `merge_asof`로 동기화 (허용 오차: {MAX_TIME_DIFFERENCE_THRESHOLD}초) ---")
    
    df_sync = pd.merge_asof(
        left=df_image,
        right=df_robot,
        left_on='matching_timestamp',
        right_on='robot_timestamp',
        direction='nearest', # 가장 가까운 값 (절대값 기준)
        tolerance=MAX_TIME_DIFFERENCE_THRESHOLD
    )
    
    # 매칭되지 않은 행(NaN) 제거
    df_sync.dropna(subset=['robot_timestamp'], inplace=True)
    
    # 실제 시간 차이 계산
    df_sync['time_difference_s'] = (df_sync['matching_timestamp'] - df_sync['robot_timestamp']).abs()


    # --- 단계 4: 최종 결과 저장 ---
    if df_sync.empty:
        print("\n❌ 매칭된 데이터가 없습니다. 결과 파일이 생성되지 않았습니다.")
        return
        
    # 불필요한 매칭용 타임스탬프 컬럼 제거
    df_sync.drop(columns=['matching_timestamp'], inplace=True)
    
    output_dir = os.path.dirname(OUTPUT_SYNC_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    df_sync.to_csv(OUTPUT_SYNC_CSV_PATH, index=False)
    
    matched_count = len(df_sync)
    unmatched_count = total_image_count - matched_count
    
    print("\n\n--- 🎉 동기화 완료 ---")
    print("\n--- 매칭 결과 요약 ---")
    print(f"총 이미지 파일 수: {total_image_count}개")
    print(f"✅ 매칭 성공: {matched_count}개")
    print(f"❌ 매칭 실패: {unmatched_count}개")
    print(f"\n✅ 결과 저장 경로: {OUTPUT_SYNC_CSV_PATH}")
    print("\n--- 동기화 데이터 샘플 ---")
    
    sample_cols = ['image_timestamp', 'robot_timestamp', 'time_difference_s', 'position_fr3_joint1']
    display_cols = [col for col in sample_cols if col in df_sync.columns]
    print(df_sync[display_cols].head())


if __name__ == '__main__':
    create_synchronized_dataset_fast()