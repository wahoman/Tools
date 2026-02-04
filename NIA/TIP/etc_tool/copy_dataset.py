import os
import shutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ==========================================
# 설정 구간
# ==========================================
SOURCE_ROOT = Path(r"\\Sstl_nas\ai\datasets\NIA_new_colormapping")
DEST_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image")
MAX_WORKERS = 16  # 일꾼 수
# ==========================================

def process_class_dir(args):
    """
    하나의 클래스 폴더를 처리하는 함수 (PNG -> TXT 1:1 매칭 최적화)
    """
    class_dir, split, dest_root = args
    worker_name = threading.current_thread().name
    
    src_images_dir = class_dir / 'images'
    src_labels_dir = class_dir / 'labels'
    
    if not src_images_dir.exists():
        return 0

    # 타겟 경로 설정
    dst_images_dir = dest_root / split / class_dir.name / 'images'
    dst_labels_dir = dest_root / split / class_dir.name / 'labels'

    # 이미지 파일 리스트 가져오기
    try:
        # iterdir()는 모든 파일을 가져오므로, 아래 루프에서 png인지 체크합니다.
        all_files = list(src_images_dir.iterdir())
    except Exception:
        return 0

    count = 0
    valid_files_in_folder = 0

    # [로그] 작업 시작 알림 (파일 있는 경우만)
    if len(all_files) > 0:
        tqdm.write(f"[{worker_name}] 스캔 중.. 📂: {class_dir.name}")

    for img_file in all_files:
        if not img_file.is_file():
            continue

        # 조건 1: 확장자가 .png 인지 확인 (대소문자 무시)
        if img_file.suffix.lower() != '.png':
            continue
            
        # 조건 2: 언더바(_)가 정확히 6개인지 확인
        if img_file.name.count('_') == 6:
            valid_files_in_folder += 1
            try:
                # 라벨 파일 경로 직접 생성 (검색 X -> 지목 O)
                # 예: 이미지.png -> 이미지.txt
                label_file = src_labels_dir / f"{img_file.stem}.txt"
                
                # 라벨 파일이 실제로 있을 때만 복사 진행
                if label_file.exists():
                    # 폴더 생성 (최초 1회만 실행되게 되지만, 루프 안에 있어도 exist_ok=True라 안전)
                    dst_images_dir.mkdir(parents=True, exist_ok=True)
                    dst_labels_dir.mkdir(parents=True, exist_ok=True)

                    # 1. 이미지 복사 (png)
                    shutil.copy2(img_file, dst_images_dir / img_file.name)

                    # 2. 라벨 복사 (txt)
                    shutil.copy2(label_file, dst_labels_dir / label_file.name)
                    
                    count += 1
            except Exception as e:
                tqdm.write(f"[{worker_name}] ❌ 에러 발생 ({img_file.name}): {e}")

    # 복사된 게 있을 때만 완료 로그 출력 (너무 시끄러우면 주석 처리)
    if count > 0:
        tqdm.write(f"[{worker_name}] ✅ 완료: {class_dir.name} ({count}쌍 복사)")
    
    return count

def main():
    print(f"소스: {SOURCE_ROOT}")
    print(f"타겟: {DEST_ROOT}")
    print(f"대상: .png 이미지 & .txt 라벨 (언더바 6개 조건)")
    print(f"일꾼: {MAX_WORKERS}명")
    print("-" * 50)

    splits = ['train', 'valid']
    all_tasks = []

    # 작업 목록 생성
    print("폴더 목록을 불러오는 중...")
    for split in splits:
        current_split_path = SOURCE_ROOT / split
        if not current_split_path.exists():
            continue
            
        class_dirs = [d for d in current_split_path.iterdir() if d.is_dir()]
        for class_dir in class_dirs:
            all_tasks.append((class_dir, split, DEST_ROOT))

    print(f"총 {len(all_tasks)}개의 클래스 폴더 처리 시작!")
    print("-" * 50)

    # 멀티스레딩 실행
    total_files_copied = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_class_dir, all_tasks), 
                            total=len(all_tasks), 
                            desc="전체 진행률",
                            unit="class"))
        
        total_files_copied = sum(results)

    print("-" * 50)
    print(f"모든 작업 완료! 총 {total_files_copied} 쌍의 파일이 복사되었습니다.")

if __name__ == "__main__":
    main()