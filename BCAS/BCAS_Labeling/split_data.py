import os
import shutil
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

# ================= 설정 부분 =================
SOURCE_DIR = r'D:\hgyeo\BCAS\BCAS_organized_0225'
TARGET_DIR = r'D:\hgyeo\BCAS\BCAS_organized_0225_split'
TRAIN_RATIO = 0.8
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ============================================

def copy_file(task):
    """
    폴더 생성 확인 로직을 빼고 오직 복사만 무식하게 밀어붙입니다.
    """
    src, dst = task
    try:
        shutil.copy2(src, dst) 
        return True
    except Exception:
        return False

def main():
    source_root = Path(SOURCE_DIR)
    target_root = Path(TARGET_DIR)
    copy_tasks = []
    
    print(f"📂 데이터 스캔 중: {SOURCE_DIR}")
    classes = [d for d in source_root.iterdir() if d.is_dir()]
    
    # 병목 원인이었던 폴더 생성을 한 번에 미리 다 해둡니다.
    print("📁 필요 폴더 사전 생성 중...")
    for split in ['train', 'valid']:
        for cls_dir in classes:
            (target_root / split / cls_dir.name / 'images').mkdir(parents=True, exist_ok=True)
            (target_root / split / cls_dir.name / 'labels').mkdir(parents=True, exist_ok=True)

    for class_dir in classes:
        class_name = class_dir.name
        images_dir = class_dir / 'images'
        labels_dir = class_dir / 'labels'
        
        if not images_dir.exists():
            continue
            
        all_images = [f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * TRAIN_RATIO)
        train_imgs = all_images[:split_idx]
        valid_imgs = all_images[split_idx:]
        
        def plan_tasks(file_list, split_type):
            target_img_base = target_root / split_type / class_name / 'images'
            target_lbl_base = target_root / split_type / class_name / 'labels'

            for img_path in file_list:
                copy_tasks.append((img_path, target_img_base / img_path.name))
                
                if labels_dir.exists():
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        copy_tasks.append((label_path, target_lbl_base / label_path.name))

        plan_tasks(train_imgs, 'train')
        plan_tasks(valid_imgs, 'valid')

    print(f"🚀 총 {len(copy_tasks)}개의 파일을 멀티프로세싱으로 복사합니다...")
    print(f"🔥 CPU 코어 {multiprocessing.cpu_count()}개를 풀가동합니다.")

    # 고객님 원래 방식인 ProcessPoolExecutor 사용
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(copy_file, copy_tasks), total=len(copy_tasks), unit="file"))

    success_count = sum(results)
    print("-" * 50)
    print("🎉 분할 복사 완료!")
    print(f"성공: {success_count} / {len(copy_tasks)}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()