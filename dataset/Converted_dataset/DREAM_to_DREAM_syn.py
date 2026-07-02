import os
import glob
import json

datasets = [
    ("../DREAM_syn/panda_synth_test_dr",    "DREAM_to_DREAM_syn/panda_synth_test_dr"),
    ("../DREAM_syn/panda_synth_test_photo",  "DREAM_to_DREAM_syn/panda_synth_test_photo"),
    ("../DREAM_syn/panda_synth_train_dr",    "DREAM_to_DREAM_syn/panda_synth_train_dr"),
]

for search_dir, output_dir in datasets:
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(search_dir, '**', '*.json'), recursive=True)

    print(f"\n[{search_dir}] 총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    # _camera_settings.json을 먼저 읽어서 K matrix 확보
    camera_settings_path = os.path.join(search_dir, '_camera_settings.json')
    with open(camera_settings_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)

    intrinsics = cam_data['camera_settings'][0]['intrinsic_settings']
    k_matrix = [
        [intrinsics['fx'], 0.0, intrinsics['cx']],
        [0.0, intrinsics['fy'], intrinsics['cy']],
        [0.0, 0.0, 1.0]
    ]
    print(f"✅ K matrix loaded from '{camera_settings_path}'")

    count = 0
    for json_path in json_files:
        if os.path.basename(json_path) == '_camera_settings.json':
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            sample = json.load(f)

        path_without_ext = os.path.splitext(json_path)[0]
        image_path = f"{path_without_ext}.rgb.jpg"
        # Compute image path relative to the output JSON file's directory
        original_filename = os.path.basename(json_path)
        output_json_dir = os.path.abspath(os.path.join(output_dir))
        abs_image_path = os.path.abspath(image_path)
        rel_image_path = os.path.relpath(abs_image_path, start=output_json_dir)

        sample['meta'] = {
            "K": k_matrix,
            "image_path": rel_image_path,
        }

        output_json_path = os.path.join(output_dir, original_filename)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=4)
        count += 1

    print(f"✅ {count}개 파일 변환 완료 → {output_dir}")