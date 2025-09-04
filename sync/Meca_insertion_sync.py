import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- ⚙️ 1. 설정 변수 ---

# 데이터가 포함된 기본 상위 디렉토리 목록
# 1th부터 3th까지의 모든 경로를 동적으로 생성합니다.
BASE_DIRS = [f"../dataset/Meca_insertion/Meca_{i}th_*" for i in range(1, 4)]

# 최종 동기화 결과가 저장될 경로 및 파일명
OUTPUT_SYNC_CSV_PATH = "../dataset/Meca_insertion/Meca_insertion_matched_joint_angle.csv"

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

def read_joint_data_from_txt(file_path):
    try:
        # 파일의 첫 줄(헤더)은 건너뛰고, 데이터만 읽어옵니다.
        # 예제 데이터(timestamp + 12개 값)를 기준으로 열 이름을 동적으로 생성합니다.
        num_joint_angles = 7
        num_cartesian_pose = 5 # 예제 기준 12 - 7 = 5
        
        col_names = ['timestamp']
        col_names.extend([f'joint_{i}' for i in range(num_joint_angles)])
        col_names.extend([f'pose_{i}' for i in range(num_cartesian_pose)])

        # pandas를 사용하여 CSV 형식의 txt 파일 읽기
        df = pd.read_csv(file_path, skiprows=1, header=None, names=col_names)

        # DataFrame을 기존 JSON과 유사한 [{...}, {...}] 형태로 변환
        data_list = []
        for _, row in df.iterrows():
            record = {
                'timestamp': row['timestamp'],
                'joint_angles': row[[f'joint_{i}' for i in range(num_joint_angles)]].tolist(),
                'cartesian_pose': row[[f'pose_{i}' for i in range(num_cartesian_pose)]].tolist()
            }
            data_list.append(record)
        
        return data_list

    except (FileNotFoundError, pd.errors.EmptyDataError):
        # 파일이 존재하지 않거나 내용이 비어있는 경우
        return None
    except Exception as e:
        # 그 외 다른 오류가 발생한 경우
        print(f"Error processing file {file_path}: {e}")
        return None

# --- 🚀 3. 메인 실행 함수 ---

def create_synchronized_dataset():
    """
    이미지와 관절 데이터를 타임스탬프 기준으로 동기화하여 CSV 파일로 저장합니다.
    """

    # --- 단계 1: 모든 관절 데이터(txt) 로드 및 단일 데이터프레임 생성 ---
    print("--- 단계 1: 모든 관절 데이터(txt) 로딩 및 통합 ---")
    # find_files_by_extension 헬퍼 함수를 사용해 모든 robot_data.txt 파일 경로를 찾습니다.
    # robot_data.txt는 subfolder가 없으므로 두 번째 인자는 ''로 비워둡니다.
    joint_file_paths = find_files_by_extension(BASE_DIRS, '', 'robot_data.txt')

    if not joint_file_paths:
        print(f"❌ 에러: 지정된 경로에서 관절 데이터(robot_data.txt) 파일을 찾을 수 없습니다.")
        return

    all_joint_records = []
    for path in tqdm(joint_file_paths, desc="관절 데이터 파일 처리 중"):
        # read_joint_data_from_txt 함수로 파일 내의 모든 데이터를 읽어옵니다.
        records_from_file = read_joint_data_from_txt(path)
        
        if records_from_file:
            # 읽어온 데이터(리스트)를 전체 리스트에 추가합니다.
            all_joint_records.extend(records_from_file)

    # 통합된 데이터를 기반으로 DataFrame을 생성합니다.
    # DataFrame을 후속 처리에서 사용하기 편한 형태로 변환합니다.
    flat_records = []
    for record in all_joint_records:
        flat_record = {'joint_timestamp': record['timestamp']}
        # joint_angles 리스트를 joint_1, joint_2, ... 와 같이 별도의 열로 펼칩니다.
        for i, angle in enumerate(record['joint_angles']):
            flat_record[f'joint_{i+1}'] = angle
        flat_records.append(flat_record)

    if not flat_records:
        print("❌ 에러: 모든 관절 데이터 파일에서 유효한 데이터를 읽지 못했습니다.")
        return

    df_joint = pd.DataFrame(flat_records)
    df_joint.sort_values('joint_timestamp', inplace=True, ignore_index=True)

    print(f"✅ 총 {len(df_joint)}개의 유효한 관절 데이터를 통합했습니다.\n")

    # --- 단계 2: 모든 이미지 파일 경로 스캔 ---
    print("--- 단계 2: 모든 이미지 파일 스캔 ---")
    # left, right, top 폴더의 모든 이미지 파일을 찾습니다.
    image_paths = []
    for subfolder in ["front", "left", "right", "top"]:
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
