import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
from collections import defaultdict

# ================= [설정 영역] =================
# 원본 데이터 최상위 폴더 (DAY1, DAY2... 가 들어있는 곳)
SOURCE_DIR = r'D:\_team_ai\BCAS'

# 결과물이 저장될 폴더 (없으면 자동으로 만듭니다)
TARGET_DIR = r'D:\hgyeo\BCAS\bcas_organized_Laptop'

# 🎯 추출할 특정 클래스 지정 (여러 개라면 콤마로 구분하여 추가 가능)
TARGET_CLASSES = {'Laptop'}

# 이미지 확장자 목록
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# ==============================================

def process_single_pair(args):
    """
    labels 폴더에 있는 txt를 읽고, 
    대응되는 images 폴더의 이미지를 찾아 분류하여 복사하는 함수
    """
    txt_path_str, target_root_str = args
    
    try:
        txt_path = Path(txt_path_str)
        target_root = Path(target_root_str)
        
        # ---------------------------------------------------------
        # 1. 이미지 파일 찾기 로직
        # ---------------------------------------------------------
        if txt_path.parent.name == 'labels':
            img_dir = txt_path.parent.parent / 'images'
        else:
            img_dir = txt_path.parent

        image_path = None
        for ext in IMG_EXTENSIONS:
            temp_path = img_dir / f"{txt_path.stem}{ext}"
            if temp_path.exists():
                image_path = temp_path
                break
            
            temp_path_upper = img_dir / f"{txt_path.stem}{ext.upper()}"
            if temp_path_upper.exists():
                image_path = temp_path_upper
                break
        
        if image_path is None:
            return False

        # ---------------------------------------------------------
        # 2. txt 파일 내용 읽기 및 특정 클래스만 필터링 (✨핵심 변경 부분✨)
        # ---------------------------------------------------------
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return False

        class_map = defaultdict(list)
        
        for line in lines:
            parts = line.strip().split()
            if not parts: 
                continue 
            
            # 첫 번째 단어가 클래스 이름
            class_name = parts[0]
            
            # 🎯 지정한 클래스(TARGET_CLASSES)에 포함될 때만 저장
            if class_name in TARGET_CLASSES:
                class_map[class_name].append(line)

        # 타겟 클래스가 이 파일에 하나도 없다면 작업 종료 (복사 안 함)
        if not class_map:
            return False

        # ---------------------------------------------------------
        # 3. 결과 폴더에 저장 (복사 & 새 파일 생성)
        # ---------------------------------------------------------
        for class_name, filtered_lines in class_map.items():
            save_dir_img = target_root / class_name / 'images'
            save_dir_lbl = target_root / class_name / 'labels'
            
            os.makedirs(save_dir_img, exist_ok=True)
            os.makedirs(save_dir_lbl, exist_ok=True)
            
            shutil.copy2(image_path, save_dir_img / image_path.name)
            
            new_txt_path = save_dir_lbl / txt_path.name
            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)

        return True

    except Exception:
        return False

def main():
    multiprocessing.freeze_support()

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 결과 저장 폴더를 생성했습니다: {TARGET_DIR}")

    print("📂 원본 데이터(DAY 폴더들) 스캔 중...")
    source_path = Path(SOURCE_DIR)
    
    all_txt_files = [str(p) for p in source_path.rglob('*.txt') if p.is_file()]
    all_txt_files = [f for f in all_txt_files if 'classes.txt' not in f.lower()]

    print(f"-> 총 {len(all_txt_files)}개의 라벨 파일을 발견했습니다.")
    print(f"🎯 추출 대상 클래스: {', '.join(TARGET_CLASSES)}")
    print(f"🔥 CPU 코어 {multiprocessing.cpu_count()}개로 병렬 처리(복사)를 시작합니다.")

    tasks = [(f, TARGET_DIR) for f in all_txt_files]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_pair, tasks), total=len(tasks), unit="file"))

    success_count = sum(results)
    
    print("-" * 50)
    print("🎉 작업 완료!")
    print(f"처리된 세트(이미지+라벨) 수: {success_count}개")
    print(f"조건에 안 맞아 건너뛴 파일(이미지 없음 or 타겟 클래스 없음): {len(all_txt_files) - success_count}개")
    print(f"저장 위치: {TARGET_DIR}")

if __name__ == "__main__":
    main()