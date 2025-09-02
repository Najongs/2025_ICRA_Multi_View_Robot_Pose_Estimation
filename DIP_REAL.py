# inference_realtime.py

import pyzed.sl as sl
import cv2
import threading
import time
import os
import json
import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image

# 이전에 만든 models.py에서 모델 클래스들을 임포트합니다.
from model.model import DINOv2PoseEstimator 

# ==============================================================================
# 1. 준비: 모델 로딩 및 Helper 함수
# ==============================================================================

def load_model(model_path, model_name, device):
    """학습된 모델을 불러와 평가 모드로 설정합니다."""
    print("--- Loading model ---")
    model = DINOv2PoseEstimator(model_name)
    # DataParallel로 저장된 모델을 불러오기 위해 state_dict 키를 조정할 필요가 있을 수 있습니다.
    # 여기서는 간단하게 로드합니다.
    state_dict = torch.load(model_path, map_location=device)
    # DataParallel 접두사('module.') 제거
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded and set to {device} in eval mode.")
    return model

def draw_keypoints(image, keypoints, links):
    """이미지에 키포인트와 관절 연결선을 그립니다."""
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
    for start_idx, end_idx in links:
        start_point = tuple(map(int, keypoints[start_idx]))
        end_point = tuple(map(int, keypoints[end_idx]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    return image

# ==============================================================================
# 2. 핵심 수정: ZedCamera 스레드
# ==============================================================================

class ZedCamera(threading.Thread):
    def __init__(self, serial_number, view_name, model, transform, calib_data, device):
        super().__init__()
        self.serial_number = serial_number
        self.view_name = view_name
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.stop_signal = threading.Event()
        
        # 공유 리소스
        self.model = model
        self.transform = transform
        self.camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
        self.dist_coeffs = np.array(calib_data["distortion_coeffs"], dtype=np.float32)
        self.device = device

        self.processed_frame = None
        
        # ### 수정된 부분 1: 상태 플래그 추가 ###
        self.is_ready = False
        self.initialization_failed = False # 초기화 실패 여부를 저장할 플래그

        # 로봇 관절 연결 정보 (Base -> J1 -> J2 -> ...)
        self.robot_links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    def run(self):
        # 카메라 초기화
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720 # 실시간 처리를 위해 해상도 낮춤
        init_params.camera_fps = 30
        init_params.set_from_serial_number(self.serial_number)
        
        # ### 수정된 부분 2: 초기화 실패 시 플래그 설정 ###
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print(f"### ERROR: Failed to open camera {self.serial_number} ({self.view_name})")
            self.initialization_failed = True # 실패 플래그를 True로 설정
            return # 스레드 종료

        self.is_ready = True
        print(f"Camera {self.serial_number} ({self.view_name}) initialized.")
        
        image_sl = sl.Mat()

        while not self.stop_signal.is_set():
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                # ZED 카메라에서 왼쪽 뷰 이미지 가져오기
                self.zed.retrieve_image(image_sl, sl.VIEW.LEFT)
                frame_bgr = image_sl.get_data()[:, :, :3] # BGR, Alpha 채널 제외
                
                # 1. 전처리
                undistorted_frame = cv2.undistort(frame_bgr, self.camera_matrix, self.dist_coeffs)
                pil_image = Image.fromarray(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
                image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                # 2. 추론
                with torch.no_grad():
                    pred_heatmaps, _ = self.model(image_tensor)
                
                pred_heatmaps = pred_heatmaps[0].cpu() # (Num_Joints, H, W)
                
                # 3. 후처리: 히트맵에서 키포인트 추출
                keypoints = []
                h, w = pred_heatmaps.shape[1:]
                frame_h, frame_w, _ = undistorted_frame.shape
                for j in range(pred_heatmaps.shape[0]):
                    y, x = np.unravel_index(torch.argmax(pred_heatmaps[j]).numpy(), (h, w))
                    scaled_x = x * (frame_w / w)
                    scaled_y = y * (frame_h / h)
                    keypoints.append([scaled_x, scaled_y])
                keypoints = np.array(keypoints)

                # 4. 시각화: 원본 프레임에 키포인트 그리기
                self.processed_frame = draw_keypoints(undistorted_frame.copy(), keypoints, self.robot_links)

        self.zed.close()
        print(f"Camera {self.serial_number} ({self.view_name}) stopped.")

    def stop(self):
        self.stop_signal.set()

# ==============================================================================
# 3. 메인 실행 및 시각화 루프
# ==============================================================================

def main():
    # --- 설정 ---
    MODEL_PATH = './model/best_pose_estimator_model.pth'
    MODEL_NAME = 'vit_base_patch14_dinov2.lvd142m'
    CALIB_DIR = "./dataset/Fr5/Calib_cam_from_conf"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. 모델 및 전처리기 로드 ---
    model = load_model(MODEL_PATH, MODEL_NAME, device)
    
    dino_cfg = timm.create_model(MODEL_NAME, pretrained=True).default_cfg
    transform = transforms.Compose([
        transforms.Resize(dino_cfg['input_size'][-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=dino_cfg['mean'], std=dino_cfg['std'])
    ])
    
    # --- 2. 캘리브레이션 데이터 로드 ---
    camera_configs = {
        'right': {'serial': 34850673, 'cam_type': 'leftcam'},
        'left':  {'serial': 38007749, 'cam_type': 'leftcam'},
        # 'top':   {'serial': 30779426, 'cam_type': 'leftcam'},
        'top':   {'serial': 30695000, 'cam_type': 'leftcam'}
    }
    
    calib_data_all = {}
    for view, cfg in camera_configs.items():
        calib_path = os.path.join(CALIB_DIR, f"{view}_{cfg['serial']}_{cfg['cam_type']}_calib.json")
        try:
            with open(calib_path, 'r') as f:
                calib_data_all[view] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Calibration file not found for {view} camera: {calib_path}")
            # 캘리브레이션 파일이 없으면 해당 카메라를 사용 목록에서 제외할 수 있습니다.
            # 여기서는 일단 진행하고, 스레드에서 어차피 실패할 것입니다.
            # 또는 여기서 camera_configs.pop(view)를 호출하여 아예 시도조차 안하게 할 수 있습니다.

    # --- 3. 카메라 스레드 생성 및 시작 ---
    cameras = [
        ZedCamera(serial_number=cfg['serial'], view_name=view, model=model, 
                  transform=transform, calib_data=calib_data_all[view], device=device)
        for view, cfg in camera_configs.items()
    ]
    
    for cam in cameras:
        cam.start()

    # ### 수정된 부분 3: 모든 카메라가 초기화를 '시도'할 때까지 대기 ###
    while not all(cam.is_ready or cam.initialization_failed for cam in cameras):
        print("Waiting for all cameras to finish initialization attempt...")
        time.sleep(1)

    # ### 수정된 부분 4: 성공한 카메라와 실패한 카메라 분리 및 요약 정보 출력 ###
    active_cameras = [cam for cam in cameras if cam.is_ready]
    failed_cameras = [cam for cam in cameras if cam.initialization_failed]

    print("\n--- Camera Initialization Summary ---")
    for cam in active_cameras:
        print(f"[SUCCESS] {cam.view_name} camera (S/N: {cam.serial_number}) is ready.")
    for cam in failed_cameras:
        print(f"[FAILED]  {cam.view_name} camera (S/N: {cam.serial_number}) could not be opened.")
    print("-------------------------------------\n")

    if not active_cameras:
        print("No cameras are available. Exiting application.")
        return

    # --- 4. 메인 시각화 루프 ---
    try:
        SCREEN_MAX_WIDTH = 1800 
        SCREEN_MAX_HEIGHT = 950

        # ### 수정된 부분 5: 플레이스홀더 이미지 생성 ###
        # HD720 해상도 (1280x720)에 맞춰 검은색 배경 이미지 생성
        placeholder_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(placeholder_frame, "Camera Not Found", (400, 360), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        while True:
            active_frames = {cam.view_name: cam.processed_frame for cam in active_cameras}
            frames = {view: active_frames.get(view) for view in camera_configs.keys()}

            if all(frame is not None for frame in active_frames.values()):
                top_frame = frames.get('top') if frames.get('top') is not None else placeholder_frame.copy()
                left_frame = frames.get('left') if frames.get('left') is not None else placeholder_frame.copy()
                right_frame = frames.get('right') if frames.get('right') is not None else placeholder_frame.copy()

                # --- (프레임 크기 통일 및 합치기 과정은 기존과 동일) ---
                min_h = min(f.shape[0] for f in [top_frame, left_frame, right_frame] if f is not None)
                top_frame = cv2.resize(top_frame, (int(top_frame.shape[1] * min_h / top_frame.shape[0]), min_h))
                left_frame = cv2.resize(left_frame, (int(left_frame.shape[1] * min_h / left_frame.shape[0]), min_h))
                right_frame = cv2.resize(right_frame, (int(right_frame.shape[1] * min_h / right_frame.shape[0]), min_h))
                
                bottom_row = np.hstack((left_frame, right_frame))
                top_w = top_frame.shape[1]
                bottom_w = bottom_row.shape[1]
                
                if top_w > bottom_w:
                    bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, top_w - bottom_w, cv2.BORDER_CONSTANT, value=[0,0,0])
                elif bottom_w > top_w:
                    top_frame = cv2.copyMakeBorder(top_frame, 0, 0, 0, bottom_w - top_w, cv2.BORDER_CONSTANT, value=[0,0,0])

                canvas = np.vstack((top_frame, bottom_row))
                
                # ### 수정된 부분 2: 화면에 맞게 최종 캔버스 리사이징 ###
                canvas_h, canvas_w, _ = canvas.shape
                
                # 화면 비율을 유지하면서 축소할 비율 계산
                scale = min(SCREEN_MAX_WIDTH / canvas_w, SCREEN_MAX_HEIGHT / canvas_h)
                
                # 원본 이미지가 최대 크기보다 클 경우에만 리사이징
                if scale < 1.0:
                    new_w = int(canvas_w * scale)
                    new_h = int(canvas_h * scale)
                    display_canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    display_canvas = canvas
                
                cv2.imshow("Multi-camera Robot Pose Estimation", display_canvas)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    finally:
        # --- 5. 종료 처리 ---
        # 모든 스레드(성공/실패 무관)에 정지 신호 보냄
        print("Stopping all camera threads...")
        for cam in cameras:
            cam.stop()
        for cam in cameras:
            cam.join()
        cv2.destroyAllWindows()
        print("Application finished.")

if __name__ == "__main__":
    main()