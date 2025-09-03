import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- ⚙️ 1. 설정 변수 ---

# 데이터가 포함된 기본 상위 디렉토리 목록
# 1th부터 7th까지의 모든 경로를 동적으로 생성합니다.
BASE_DIRS = [f"../dataset/Fr5/Fr5_{i}th_250526" for i in range(1, 8)]

# 최종 동기화 결과가 저장될 경로 및 파일명
OUTPUT_SYNC_CSV_PATH = "../dataset/Fr5/fr5_matched_joint_angle.csv"

# 동기화 최대 허용 시간 차이 (초 단위)
# 예: 0.05는 50ms를 의미하며, 이보다 시간 차이가 크면 매칭에서 제외됩니다.
MAX_TIME_DIFFERENCE_THRESHOLD = 0.05

# 이미지 타임스탬프에 더해줄 고정 딜레이 값 (초 단위)
IMAGE_TIMESTAMP_DELAY = 0.0333

# --- 🛠️ 2. 헬퍼 함수 ---

def find_files_by_extension(base_dirs, subfolder, extension):
    """지정된 하위 폴더에서 특정 확장자를 가진 모든 파일 경로를 찾습니다."""
    all_files = []
    for base_dir in base_dirs:
        search_path = os.path.join(base_dir, subfolder, f"*{extension}")
        all_files.extend(glob.glob(search_path))
    return all_files

def parse_timestamp_from_filename(file_path):
    """파일명에서 타임스탬프를 float 형태로 추출합니다."""
    try:
        filename = os.path.basename(file_path)
        # 확장자를 제거하고 '_'로 분리하여 마지막 부분을 타임스탬프로 간주합니다.
        timestamp_str = os.path.splitext(filename)[0].split('_')[-1]
        return float(timestamp_str)
    except (IndexError, ValueError):
        # 파일명 형식이 맞지 않을 경우 None을 반환합니다.
        return None

def read_joint_data_from_json(file_path):
    """JSON 파일에서 관절 데이터 리스트를 읽어옵니다."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None

# --- 🚀 3. 메인 실행 함수 ---

def create_synchronized_dataset():
    """
    이미지와 관절 데이터를 타임스탬프 기준으로 동기화하여 CSV 파일로 저장합니다.
    """

    # --- 단계 1: 모든 관절 데이터(JSON) 로드 및 단일 데이터프레임 생성 ---
    print("--- 단계 1: 모든 관절 데이터(JSON) 로딩 및 통합 ---")
    joint_file_paths = find_files_by_extension(BASE_DIRS, "joint", ".json")
    
    if not joint_file_paths:
        print(f"❌ 에러: 지정된 경로에서 관절 데이터(.json) 파일을 찾을 수 없습니다.")
        return

    joint_records = []
    for path in tqdm(joint_file_paths, desc="관절 데이터 파일 처리 중"):
        timestamp = parse_timestamp_from_filename(path)
        joint_angles = read_joint_data_from_json(path)
        
        if timestamp is not None and joint_angles is not None and len(joint_angles) == 6:
            record = {'joint_timestamp': timestamp, 'joint_path': path}
            # 각 관절 데이터를 별도의 열로 추가합니다.
            for i, angle in enumerate(joint_angles):
                record[f'joint_{i+1}'] = angle
            joint_records.append(record)
            
    df_joint = pd.DataFrame(joint_records)
    df_joint.sort_values('joint_timestamp', inplace=True, ignore_index=True)
    
    print(f"✅ 총 {len(df_joint)}개의 유효한 관절 데이터를 통합했습니다.\n")

    # --- 단계 2: 모든 이미지 파일 경로 스캔 ---
    print("--- 단계 2: 모든 이미지 파일 스캔 ---")
    # left, right, top 폴더의 모든 이미지 파일을 찾습니다.
    image_paths = []
    for subfolder in ["left", "right", "top"]:
        image_paths.extend(find_files_by_extension(BASE_DIRS, subfolder, ".jpg"))
    print(f"✅ 총 {len(image_paths)}개의 이미지 파일을 찾았습니다.\n")

    # --- 단계 3: 이미지와 관절 데이터 타임스탬프 기준 동기화 ---
    print(f"--- 단계 3: 이미지와 관절 데이터 동기화 (이미지 딜레이 +{IMAGE_TIMESTAMP_DELAY}초 적용) ---")
    synchronized_records = []
    joint_timestamps_np = df_joint['joint_timestamp'].values

    for image_path in tqdm(image_paths, desc="이미지 매칭 중"):
        img_ts = parse_timestamp_from_filename(image_path)
        if img_ts is None:
            continue

        # 이미지 타임스탬프에 딜레이를 더하여 매칭에 사용할 기준 시간을 계산합니다.
        adjusted_img_ts = img_ts + IMAGE_TIMESTAMP_DELAY

        # 가장 가까운 관절 타임스탬프의 인덱스를 찾습니다.
        time_diffs = np.abs(joint_timestamps_np - adjusted_img_ts)
        closest_idx = np.argmin(time_diffs)
        min_time_diff = time_diffs[closest_idx]

        # 시간 차이가 설정된 임계값 이내인지 확인합니다.
        if min_time_diff < MAX_TIME_DIFFERENCE_THRESHOLD:
            matching_joint_row = df_joint.iloc[closest_idx]
            
            # 매칭된 데이터를 저장할 레코드를 생성합니다.
            record = {
                'image_path': image_path,
                'image_timestamp': img_ts,
                'time_difference_s': min_time_diff
            }
            # 매칭된 관절 데이터의 모든 열을 레코드에 추가합니다.
            record.update(matching_joint_row.to_dict())
            
            synchronized_records.append(record)

    # --- 단계 4: 최종 결과 저장 ---
    if not synchronized_records:
        print("\n❌ 매칭된 데이터가 없습니다. 결과 파일이 생성되지 않았습니다.")
        return
        
    df_sync = pd.DataFrame(synchronized_records)
    # 이미지 타임스탬프 기준으로 최종 정렬합니다.
    df_sync.sort_values('image_timestamp', inplace=True, ignore_index=True)
    
    # 출력 폴더가 없으면 생성합니다.
    output_dir = os.path.dirname(OUTPUT_SYNC_CSV_PATH)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
    
    df_sync.to_csv(OUTPUT_SYNC_CSV_PATH, index=False)
    
    print("\n\n--- 🎉 동기화 완료 ---")
    print(f"✅ 총 {len(df_sync)}개의 이미지-관절 데이터 쌍이 성공적으로 동기화되었습니다.")
    print(f"✅ 결과 저장 경로: {OUTPUT_SYNC_CSV_PATH}")
    print("\n--- 동기화 데이터 샘플 (딜레이 적용 후) ---")
    
    # 결과를 확인하기 좋은 주요 컬럼들만 샘플로 출력합니다.
    sample_cols = ['image_timestamp', 'joint_timestamp', 'time_difference_s', 'joint_1', 'joint_2']
    print(df_sync[sample_cols].head())


if __name__ == '__main__':
    create_synchronized_dataset()
