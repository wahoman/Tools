import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# =========================================================
# 1. 설정
# =========================================================
# 원본 데이터 루트 (이 안의 파일들이 이동됩니다)
SRC_ROOT_DIR = r"D:/hgyeo/BCAS_TIP/TIP_output"

# 결과 데이터셋 루트
DST_ROOT_DIR = r"D:/hgyeo/BCAS_TIP/TIP_Dataset_Final"

# 분할 비율
TRAIN_RATIO = 0.8

# 이미지 확장자
IMG_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def main():
    # 1. 원본 경로 확인
    if not os.path.exists(SRC_ROOT_DIR):
        print(f"❌ 원본 경로가 존재하지 않습니다: {SRC_ROOT_DIR}")
        return

    # 클래스 폴더 목록 가져오기
    class_folders = [d for d in os.listdir(SRC_ROOT_DIR) if os.path.isdir(os.path.join(SRC_ROOT_DIR, d))]
    print(f"[*] 감지된 클래스: {len(class_folders)}개")
    print(f"[*] 구조: train/클래스명/images & labels 로 이동합니다.")

    total_train = 0
    total_valid = 0

    # 2. 각 클래스별로 순회
    for class_name in tqdm(class_folders, desc="Moving Classes"):
        src_class_path = os.path.join(SRC_ROOT_DIR, class_name)
        src_images_path = os.path.join(src_class_path, 'images')
        src_labels_path = os.path.join(src_class_path, 'labels')

        # 해당 클래스가 저장될 목적지 기본 경로 설정
        # 예: TIP_Dataset_Final/train/Adaptor/images
        train_img_dir = os.path.join(DST_ROOT_DIR, 'train', class_name, 'images')
        train_lbl_dir = os.path.join(DST_ROOT_DIR, 'train', class_name, 'labels')
        valid_img_dir = os.path.join(DST_ROOT_DIR, 'valid', class_name, 'images')
        valid_lbl_dir = os.path.join(DST_ROOT_DIR, 'valid', class_name, 'labels')

        # 목적지 폴더 생성 (클래스별로 생성됨)
        for d in [train_img_dir, train_lbl_dir, valid_img_dir, valid_lbl_dir]:
            create_dir(d)

        # 이미지 리스트 확보
        image_files = []
        for ext in IMG_EXTENSIONS:
            image_files.extend(glob(os.path.join(src_images_path, ext)))
        
        # 짝(Pair) 맞추기
        valid_pairs = []
        for img_path in image_files:
            basename = os.path.basename(img_path)
            name_only = os.path.splitext(basename)[0]
            lbl_path = os.path.join(src_labels_path, f"{name_only}.txt")
            
            if os.path.exists(lbl_path):
                valid_pairs.append((img_path, lbl_path))
        
        # 셔플 및 분할
        random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * TRAIN_RATIO)
        
        train_set = valid_pairs[:split_idx]
        valid_set = valid_pairs[split_idx:]

        # --- 이동 함수 ---
        def move_files(file_list, dst_img_dir, dst_lbl_dir):
            for src_img, src_lbl in file_list:
                fname_img = os.path.basename(src_img)
                fname_lbl = os.path.basename(src_lbl)

                # 파일 이동 (파일명 변경 없이 그대로 이동)
                shutil.move(src_img, os.path.join(dst_img_dir, fname_img))
                shutil.move(src_lbl, os.path.join(dst_lbl_dir, fname_lbl))

        # Train 이동
        move_files(train_set, train_img_dir, train_lbl_dir)
        total_train += len(train_set)

        # Valid 이동
        move_files(valid_set, valid_img_dir, valid_lbl_dir)
        total_valid += len(valid_set)
        
        # (선택) 빈 폴더 정리: 원본 폴더가 비었으면 삭제
        try:
            if not os.listdir(src_images_path): os.rmdir(src_images_path)
            if not os.listdir(src_labels_path): os.rmdir(src_labels_path)
            if not os.listdir(src_class_path): os.rmdir(src_class_path)
        except:
            pass

    print("\n" + "="*50)
    print("✅ 클래스별 폴더 분할 이동 완료!")
    print(f"📂 저장 경로: {DST_ROOT_DIR}")
    print(f"   ㄴ train/{{클래스명}}/images")
    print(f"   ㄴ valid/{{클래스명}}/images")
    print(f"📊 Train: {total_train}장, Valid: {total_valid}장")
    print("="*50)

if __name__ == "__main__":
    main()