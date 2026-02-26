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
TARGET_DIR = r'D:\hgyeo\BCAS\BCAS_organized_0225'

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
        # 1. 이미지 파일 찾기 로직 (핵심 변경 부분)
        # ---------------------------------------------------------
        # 현재 txt 파일의 부모 폴더 이름이 'labels'인지 확인
        # 구조: .../DAY1/labels/파일.txt
        
        # 부모 폴더가 labels가 아니면, 혹시 같은 폴더에 있을 수도 있으니 현재 폴더 유지
        # 하지만 질문하신 구조(DAY/labels, DAY/images)를 우선적으로 처리
        
        if txt_path.parent.name == 'labels':
            # ../labels/.. -> ../images/.. 로 경로 변경
            img_dir = txt_path.parent.parent / 'images'
        else:
            # labels 폴더 안에 있는게 아니라면, 같은 폴더에 있다고 가정
            img_dir = txt_path.parent

        image_path = None
        # 해당 이미지 폴더에서 확장자만 바꿔서 파일 존재 여부 확인
        for ext in IMG_EXTENSIONS:
            temp_path = img_dir / f"{txt_path.stem}{ext}" # stem은 확장자 뺀 파일명
            if temp_path.exists():
                image_path = temp_path
                break
            
            # 대문자 확장자(.JPG) 대응
            temp_path_upper = img_dir / f"{txt_path.stem}{ext.upper()}"
            if temp_path_upper.exists():
                image_path = temp_path_upper
                break
        
        # 이미지가 없으면 작업 불가 (건너뜀)
        if image_path is None:
            return False

        # ---------------------------------------------------------
        # 2. txt 파일 내용 읽기 및 클래스 분류
        # ---------------------------------------------------------
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return False

        # 클래스별로 줄 모으기
        # 예: {'Razor': [...], 'Laptop': [...]}
        class_map = defaultdict(list)
        
        for line in lines:
            parts = line.strip().split()
            if not parts: 
                continue 
            
            # 첫 번째 단어가 클래스 이름
            class_name = parts[0]
            class_map[class_name].append(line)

        # ---------------------------------------------------------
        # 3. 결과 폴더에 저장 (복사 & 새 파일 생성)
        # ---------------------------------------------------------
        for class_name, filtered_lines in class_map.items():
            # 저장 경로: 타겟 / 클래스명 / images (또는 labels)
            save_dir_img = target_root / class_name / 'images'
            save_dir_lbl = target_root / class_name / 'labels'
            
            # 폴더가 없으면 생성 (exist_ok=True)
            os.makedirs(save_dir_img, exist_ok=True)
            os.makedirs(save_dir_lbl, exist_ok=True)
            
            # (A) 이미지 복사 (원본 유지)
            shutil.copy2(image_path, save_dir_img / image_path.name)
            
            # (B) 라벨 파일 생성 (해당 클래스 라인만)
            new_txt_path = save_dir_lbl / txt_path.name
            with open(new_txt_path, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)

        return True

    except Exception:
        return False

def main():
    multiprocessing.freeze_support()

    # 결과 폴더가 아예 없으면 생성
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"📁 결과 저장 폴더를 생성했습니다: {TARGET_DIR}")

    print("📂 원본 데이터(DAY 폴더들) 스캔 중...")
    source_path = Path(SOURCE_DIR)
    
    # 모든 하위 폴더의 .txt 파일 찾기
    all_txt_files = [str(p) for p in source_path.rglob('*.txt') if p.is_file()]
    
    # (선택) classes.txt 같은 설정 파일 제외
    all_txt_files = [f for f in all_txt_files if 'classes.txt' not in f.lower()]

    print(f"-> 총 {len(all_txt_files)}개의 라벨 파일을 발견했습니다.")
    print(f"🔥 CPU 코어 {multiprocessing.cpu_count()}개로 병렬 처리(복사)를 시작합니다.")

    tasks = [(f, TARGET_DIR) for f in all_txt_files]

    # 병렬 처리 실행
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_pair, tasks), total=len(tasks), unit="file"))

    success_count = sum(results)
    
    print("-" * 50)
    print("🎉 작업 완료!")
    print(f"처리된 세트(이미지+라벨) 수: {success_count}개")
    print(f"매칭되는 이미지가 없어 건너뛴 파일: {len(all_txt_files) - success_count}개")
    print(f"저장 위치: {TARGET_DIR}")

if __name__ == "__main__":
    main()