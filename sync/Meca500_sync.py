import os
import glob
import json
import pandas as pd
from tqdm import tqdm

# --- ⚙️ 1. 설정 변수 ---

# 데이터 경로
IMAGE_PATH = '../dataset/Meca500/image'
JSON_PATH = '../dataset/Meca500/angle'

# 저장될 CSV 파일 경로
OUTPUT_CSV_PATH = '../dataset/Meca500/Meca500_matched_joint_angle.csv'


# --- 🚀 2. 메인 실행 로직 ---

def create_matched_csv():
    """
    이미지 파일과 Angle JSON 파일을 인덱스 기준으로 매칭하여 CSV로 저장합니다.
    """
    print(f"JSON 파일 검색 경로: {JSON_PATH}")
    # JSON 경로에서 angle로 시작하고 .json으로 끝나는 모든 파일 목록을 가져옵니다.
    json_files = glob.glob(os.path.join(JSON_PATH, 'angle*.json'))
    
    if not json_files:
        print("❌ 에러: 해당 경로에서 Angle 파일을 찾을 수 없습니다. JSON_PATH를 확인해주세요.")
        return

    print(f"✅ 총 {len(json_files)}개의 Angle 파일을 찾았습니다. 데이터 매칭을 시작합니다.")

    all_records = []
    # tqdm을 사용하여 진행 상황을 표시합니다.
    for json_path in tqdm(json_files, desc="파일 매칭 중"):
        try:
            # 파일 이름에서 숫자 인덱스 추출 (예: 'angle123.json' -> '123')
            base_name = os.path.basename(json_path)
            index = base_name.replace('angle', '').replace('.json', '')
            
            # 인덱스를 이용해 해당하는 이미지 파일 경로 생성
            image_name = f'image{index}.jpg'
            image_path = os.path.join(IMAGE_PATH, image_name)
            
            # 해당하는 이미지 파일이 실제로 존재하는지 확인
            if os.path.exists(image_path):
                # JSON 파일 열고 관절 각도 데이터 읽기
                with open(json_path, 'r') as f:
                    joint_angles = json.load(f)
                
                # 데이터 유효성 확인 (리스트 형태, 6개 요소)
                if isinstance(joint_angles, list) and len(joint_angles) == 6:
                    record = {
                        'image_path': image_path,
                        'joint_1': joint_angles[0],
                        'joint_2': joint_angles[1],
                        'joint_3': joint_angles[2],
                        'joint_4': joint_angles[3],
                        'joint_5': joint_angles[4],
                        'joint_6': joint_angles[5],
                    }
                    all_records.append(record)
                    
        except Exception as e:
            print(f"\n⚠️ 파일 처리 중 오류 발생: {json_path} | 오류: {e}")

    if not all_records:
        print("❌ 매칭된 데이터가 없습니다. 파일 이름 형식을 확인해주세요. (예: image1.jpg, angle1.json)")
        return
        
    # 리스트를 Pandas DataFrame으로 변환
    df = pd.DataFrame(all_records)
    
    # DataFrame을 CSV 파일로 저장 (인덱스는 저장하지 않음)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print("\n--- 🎉 작업 완료 ---")
    print(f"✅ 총 {len(df)}개의 데이터 쌍을 성공적으로 매칭했습니다.")
    print(f"✅ 결과 저장 경로: {OUTPUT_CSV_PATH}")
    print("\n--- 데이터 샘플 ---")
    print(df.head())

# 스크립트 실행
if __name__ == '__main__':
    create_matched_csv()