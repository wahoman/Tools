import os
import json
import numpy as np
import cv2
from ultralytics import YOLO

# =========================================================
# [설정 영역] 이 부분만 환경에 맞게 수정하세요
# =========================================================

# 1. 모델 경로
MODEL_PATH = r"D:\hgyeo\testset_labeling\train2_0112\weights\best.pt"

# 2. 작업할 최상위 경로 (이 아래의 모든 images 폴더를 찾습니다)
ROOT_DIR = r"D:\hgyeo\testset_labeling\BCAS_Labeling\BCAS_DAY1"

# 3. 결과가 저장될 폴더 이름
OUTPUT_LABEL_FOLDER_NAME = "X-ray_Data_labels_7"

# 4. 추론 설정
IMG_SIZE = 896          # 학습 사이즈와 동일하게
CONF_THRESHOLD = 0.2    # (추천) 너무 낮은 신뢰도는 제외 (0.3 ~ 0.5)
IOU_THRESHOLD = 0.2     # 중복 박스 제거 기준

# 5. [핵심] 노이즈 제거 및 품질 옵션
# (1) 모폴로지 커널 크기: 클수록 튀어나온 부분을 더 강하게 깎아냅니다. (3, 5, 7 중 선택)
# 이미지를 보셨을 때 튀어나온 게 좀 크다면 5나 7을 추천합니다.
MORPH_KERNEL_SIZE = 7   

# (2) 외곽선 단순화 강도: 클수록 선이 더 단순/매끈해짐 (0.0005 ~ 0.002)
# 0.0005는 디테일 유지, 0.001은 적당히 매끈함
SMOOTHING_FACTOR = 0.0005 

# (3) 최소 면적: 노이즈 제거 후에도 남은 찌꺼기가 이 값보다 작으면 버림 (픽셀 수)
MIN_MASK_AREA = 50

# =========================================================

def find_images_folders(root_path):
    """지정된 경로 하위에서 'images' 폴더들을 모두 찾습니다."""
    target_folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'X-ray_Data_png' in dirnames:
            target_folders.append(os.path.join(dirpath, 'X-ray_Data_png'))
    return target_folders

def run_batch_labeling():
    # 1. 대상 폴더 찾기
    print(f"🔍 '{ROOT_DIR}' 경로 검색 중...")
    target_folders = find_images_folders(ROOT_DIR)
    
    if not target_folders:
        print("❌ 'images' 폴더를 찾지 못했습니다.")
        return

    print(f"📂 총 {len(target_folders)}개의 폴더를 찾았습니다.")

    # 2. 모델 로드
    print(f"🔥 모델 로드 중: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 모폴로지 연산용 커널 생성 (미리 만들어둠)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

    # 3. 폴더별 순차 처리
    for folder_idx, current_img_folder in enumerate(target_folders):
        print(f"\n▶ [{folder_idx+1}/{len(target_folders)}] 처리 시작: {current_img_folder}")
        
        # 저장 경로 생성
        parent_dir = os.path.dirname(current_img_folder)
        label_output_dir = os.path.join(parent_dir, OUTPUT_LABEL_FOLDER_NAME)
        os.makedirs(label_output_dir, exist_ok=True)
        
        # 이미지 파일 리스트
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(current_img_folder) if f.lower().endswith(valid_exts)]
        
        if not image_files:
            print("   ⚠️ 이미지 파일이 없습니다.")
            continue

        success_count = 0

        for i, img_file in enumerate(image_files):
            img_path = os.path.join(current_img_folder, img_file)
            json_path = os.path.join(label_output_dir, os.path.splitext(img_file)[0] + ".json")
            
            try:
                # 이미지 로드 (한글 경로 대응)
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None: continue
                h, w = img.shape[:2]

                # 추론 실행
                results = model.predict(
                    source=img,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE,
                    retina_masks=True,  # [필수] 고화질 마스크 모드
                    verbose=False,
                    device=0
                )

                result = results[0]
                if not result.masks: continue

                shapes = []
                
                # 데이터 추출
                boxes_cls = result.boxes.cls.cpu().numpy()
                masks_data = result.masks.data.cpu().numpy() # 비트맵 마스크 가져오기

                # 객체별 처리 루프
                for j, mask_tensor in enumerate(masks_data):
                    # 1. 마스크 크기 맞추기 (모델 출력 -> 원본 크기)
                    # cv2.resize는 (width, height) 순서임에 주의
                    mask_img = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_uint8 = (mask_img * 255).astype(np.uint8)

                    # 2. [노이즈 제거] 모폴로지 열기 (Opening)
                    # 튀어나온 픽셀을 깎아내고(Erosion), 다시 채움(Dilation) -> 돌출부 제거됨
                    cleaned_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

                    # 3. 외곽선(Polygon) 추출
                    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours: continue
                    
                    # 4. 가장 큰 덩어리만 선택 (혹시 파편이 분리되었다면 본체만 가져옴)
                    main_contour = max(contours, key=cv2.contourArea)

                    # 5. 면적 체크 (너무 작으면 저장 안 함)
                    if cv2.contourArea(main_contour) < MIN_MASK_AREA:
                        continue

                    # 점 개수 부족하면 패스 (최소 삼각형 이상)
                    if len(main_contour) < 3: continue

                    # 6. [매끄럽게] 다각형 단순화 (Smoothing)
                    epsilon = SMOOTHING_FACTOR * cv2.arcLength(main_contour.astype(np.float32), True)
                    smooth_contour = cv2.approxPolyDP(main_contour.astype(np.float32), epsilon, True)
                    
                    # 형태 변환 (N, 1, 2) -> (N, 2)
                    smooth_contour = smooth_contour.reshape(-1, 2)

                    # 클래스 정보
                    class_id = int(boxes_cls[j])
                    class_name = model.names[class_id]

                    # 저장 데이터 구성
                    shape_data = {
                        "label": class_name,
                        "points": smooth_contour.tolist(),
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape_data)

                # 유효한 라벨이 하나도 없으면 JSON 생성 안 함
                if not shapes: continue

                # JSON 구조 생성
                relative_image_path = os.path.relpath(img_path, label_output_dir).replace("\\", "/")
                
                labelme_data = {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": relative_image_path,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w
                }
                
                # 파일 쓰기
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                if success_count % 50 == 0:
                    print(f"      🚀 {success_count}장 완료...")

            except Exception as e:
                print(f"      ❌ 에러 발생 ({img_file}): {e}")
                continue
                
    print("\n" + "="*50)
    print("🎉 모든 작업이 완료되었습니다!")
    print("="*50)

if __name__ == "__main__":
    run_batch_labeling()