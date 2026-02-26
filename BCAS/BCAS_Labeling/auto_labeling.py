import os
import json
import numpy as np
import cv2
from ultralytics import YOLO

# =========================================================
# 1. 사용자 설정
# =========================================================
MODEL_PATH = r"D:\hgyeo\testset_labeling\train18_exp_Laptop_print_final\weights\best.pt"

# ★ 최상위 루트 경로 (이 아래에 있는 모든 images 폴더를 찾습니다)
ROOT_DIR = r"\\Sstl_nas\ai\5. BCAS_Labeling\BCAS_Labeling\DAY15"

# 저장될 라벨 폴더 이름 (images 폴더와 같은 위치에 생성됨)
OUTPUT_LABEL_FOLDER_NAME = "object_json_labels"

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.2
IMG_SIZE = 896
# =========================================================

def find_images_folders(root_path):
    """
    루트 경로 하위를 뒤져서 'images'라는 이름을 가진 모든 폴더 경로를 찾습니다.
    """
    target_folders = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'images' in dirnames:
            # images 폴더의 전체 경로를 추가
            target_folders.append(os.path.join(dirpath, 'images'))
    return target_folders

def run_batch_labeling():
    # 1. 처리할 폴더 목록 찾기
    print(f"🔍 '{ROOT_DIR}' 아래에서 'images' 폴더를 검색 중...")
    target_folders = find_images_folders(ROOT_DIR)
    
    if not target_folders:
        print("❌ 'images' 폴더를 하나도 찾지 못했습니다. 경로를 확인해주세요.")
        return

    print(f"📂 총 {len(target_folders)}개의 'images' 폴더를 발견했습니다.")
    for idx, f in enumerate(target_folders):
        print(f"   [{idx+1}] {f}")
    print("="*100)

    # 2. 모델 로드 (한 번만 로드해서 계속 사용)
    print(f"🔥 모델 로드 중: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 3. 폴더별 순차 처리 시작
    total_folders = len(target_folders)
    
    for folder_idx, current_img_folder in enumerate(target_folders):
        print(f"\n▶ 폴더 처리 시작 ({folder_idx+1}/{total_folders}): {current_img_folder}")
        
        # ----------------------------------------------------------------
        # 라벨 저장 경로 설정 (images 폴더의 형제 폴더로 생성)
        # 예: .../Case1/images  ->  .../Case1/json_labels
        # ----------------------------------------------------------------
        parent_dir = os.path.dirname(current_img_folder)
        label_output_dir = os.path.join(parent_dir, OUTPUT_LABEL_FOLDER_NAME)
        os.makedirs(label_output_dir, exist_ok=True)
        
        # 이미지 파일 리스트업
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(current_img_folder) if f.lower().endswith(valid_exts)]
        
        if not image_files:
            print(f"   ⚠️ 이미지 파일이 없습니다. 넘어갑니다.")
            continue

        print(f"   📍 저장소: {label_output_dir}")
        print(f"   📂 이미지: {len(image_files)}장")

        success_count = 0

        for i, img_file in enumerate(image_files):
            img_path = os.path.join(current_img_folder, img_file)
            json_path = os.path.join(label_output_dir, os.path.splitext(img_file)[0] + ".json")
            
            try:
                # 이미지 로드 (한글 경로 호환)
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
                    retina_masks=True,
                    verbose=False,
                    device=0
                )

                result = results[0]
                if not result.masks: continue

                # 데이터 추출
                shapes = []
                masks_xy = result.masks.xy
                boxes_cls = result.boxes.cls.cpu().numpy()

                for j, contour in enumerate(masks_xy):
                    if len(contour) < 3: continue
                    class_id = int(boxes_cls[j])
                    class_name = model.names[class_id]
                    
                    shape_data = {
                        "label": class_name,
                        "points": contour.tolist(),
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape_data)

                # ★ 상대 경로 계산 (AnyLabeling 호환용)
                # json 파일 위치에서 이미지 파일 위치로 가는 상대 경로
                relative_image_path = os.path.relpath(img_path, label_output_dir)
                relative_image_path = relative_image_path.replace("\\", "/")

                # JSON 저장
                labelme_data = {
                    "version": "5.0.0",
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": relative_image_path,
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w
                }
                
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                if success_count % 100 == 0:
                    print(f"       {success_count}장 완료...")

            except Exception as e:
                print(f"      ❌ 에러 ({img_file}): {e}")
                continue

        print(f"   ✅ [완료] {current_img_folder} -> {success_count}장 라벨링 됨.")

    print("\n" + "="*50)
    print("🎉 모든 폴더의 작업이 끝났습니다!")
    print("="*100)

if __name__ == "__main__":
    run_batch_labeling()