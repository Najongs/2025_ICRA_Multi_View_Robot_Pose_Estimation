import os
import json
import glob

# 원본 JSON 파일들이 있는 대상 디렉토리
TARGET_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/franka_research3_to_DREAM"

# 변경된 JSON 파일을 저장할 새로운 출력 디렉토리 (원하시는 경로로 수정하세요)
OUTPUT_DIR = "/home/najo/NAS/DIP/2025_ICRA_Multi_View_Robot_Pose_Estimation/dataset/Converted_dataset/franka_research3_to_DREAM_modified"

def update_json_image_paths():
    # 출력 폴더가 존재하지 않으면 자동으로 생성합니다.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 해당 경로의 모든 json 파일 목록을 가져옵니다.
    json_files = glob.glob(os.path.join(TARGET_DIR, "*.json"))

    if not json_files:
        print(f"'{TARGET_DIR}' 경로에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 파일을 처리 중...")

    count = 0
    for file_path in json_files:
        try:
            # 1. 원본 JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. meta 내의 image_path 수정
            if "meta" in data and "image_path" in data["meta"]:
                # 현재 JSON 파일명 (확장자 제외, 예: zed_..._1756275914.348)
                file_name = os.path.basename(file_path)
                file_stem = os.path.splitext(file_name)[0]

                old_path = data["meta"]["image_path"]

                # "../dataset/franka_research3/" -> "../../franka_research3/" 로 변경
                modified_base = old_path.replace("../dataset/franka_research3/", "../../franka_research3/")

                # 파일명을 현재 JSON 파일명과 일치시키고 확장자를 .jpg로 변경
                dir_name = os.path.dirname(modified_base)
                new_image_path = os.path.join(dir_name, f"{file_stem}.jpg")

                data["meta"]["image_path"] = new_image_path

                # 3. 새로운 폴더에 변경된 내용 저장
                # 출력 디렉토리와 파일명을 결합하여 새 파일 경로 생성
                new_file_path = os.path.join(OUTPUT_DIR, file_name)
                
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                count += 1

        except Exception as e:
            print(f"파일 처리 중 오류 발생 ({os.path.basename(file_path)}): {e}")

    print(f"완료! 총 {count}개의 파일이 '{OUTPUT_DIR}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    update_json_image_paths()