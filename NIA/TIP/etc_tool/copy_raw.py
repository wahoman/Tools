import os
import shutil
import sqlite3
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ==========================================
# [설정] 경로 설정
# ==========================================
# 1. 합쳐진 PNG가 있는 폴더 (소스)
# 구조 예상: D:\...\bare_image_png\Knife\images\*.png  또는  ...\Knife\*.png
INPUT_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image_png")

# 2. Raw 파일을 저장할 새로운 폴더 (타겟)
OUTPUT_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image_raw")

# 3. DB 경로 (Raw 파일 위치 찾는 용도)
DB_PATH = r"D:\hgyeo\colormapping\raw_files.db"

# 4. 일꾼 수 (SSD라면 16~32 추천)
MAX_WORKERS = 16
# ==========================================

FILENAME_PATTERN = re.compile(r'_\d{8}_(\d{8})_.*?_(\d+)\.png$', re.IGNORECASE)

def get_raw_path_from_db(db_cursor, raw_filename):
    try:
        db_cursor.execute("SELECT filepath FROM files WHERE filename = ?", (raw_filename,))
        result = db_cursor.fetchone()
        return result[0] if result else None
    except Exception:
        return None

def process_class_dir(args):
    # split 인자가 사라졌습니다.
    class_dir, input_root, output_root, db_path = args
    worker_name = threading.current_thread().name
    
    # 1. 이미지 폴더 위치 확인
    # 합치면서 구조가 [Class/images/*.png] 인지 [Class/*.png] 인지 확인
    src_images_dir = class_dir / 'images'
    if not src_images_dir.exists():
        src_images_dir = class_dir # images 폴더가 없으면 클래스 폴더 자체를 봅니다.

    # 저장 경로 계산 (bare_image_raw/클래스명/raws)
    # relative_to: bare_image_png/Knife -> Knife
    try:
        relative_path = class_dir.relative_to(input_root)
    except ValueError:
        relative_path = class_dir.name

    dst_raws_dir = output_root / relative_path / 'raws'
    
    # PNG 찾기
    try:
        png_files = [f for f in src_images_dir.iterdir() if f.is_file() and f.suffix.lower() == '.png']
    except Exception:
        return (0, 0)

    if not png_files:
        return (0, 0)

    # DB 연결 (스레드별)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    copied_count = 0
    skipped_count = 0
    
    # 타겟 폴더 생성
    dst_raws_dir.mkdir(parents=True, exist_ok=True)

    for png_file in png_files:
        match = FILENAME_PATTERN.search(png_file.name)
        if not match:
            continue

        sLID, d_str = match.groups()
        d_index = int(d_str)

        if d_index == 0:
            continue
        
        detector_index = d_index - 1
        search_raw_filename = f"{sLID}_{detector_index}.raw"
        
        # DB에서 Raw 파일 경로 찾기
        source_raw_path_str = get_raw_path_from_db(cursor, search_raw_filename)

        if not source_raw_path_str:
            continue

        source_raw_path = Path(source_raw_path_str)
        if not source_raw_path.exists():
            continue

        # 복사 실행
        dest_filename = png_file.with_suffix('.raw').name
        dest_file_path = dst_raws_dir / dest_filename

        if dest_file_path.exists():
            skipped_count += 1
        else:
            try:
                shutil.copy2(source_raw_path, dest_file_path)
                copied_count += 1
            except Exception as e:
                tqdm.write(f"[{worker_name}] 에러: {e}")

    conn.close()
    return (copied_count, skipped_count)

def main():
    print(f"소스 폴더 (PNG): {INPUT_ROOT}")
    print(f"타겟 폴더 (Raw): {OUTPUT_ROOT}")
    
    if not INPUT_ROOT.exists():
        print(f"❌ 오류: 소스 폴더가 없습니다 -> {INPUT_ROOT}")
        return

    if not os.path.exists(DB_PATH):
        print("❌ 오류: DB 파일이 없습니다.")
        return

    print("-" * 50)
    print("클래스 폴더 스캔 중...")

    # [핵심 변경] train/valid 구분 없이 바로 클래스 폴더를 찾습니다.
    class_dirs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("❌ 처리할 클래스 폴더가 없습니다. 경로를 확인하세요.")
        return

    # 작업 목록 생성
    all_tasks = []
    for class_dir in class_dirs:
        all_tasks.append((class_dir, INPUT_ROOT, OUTPUT_ROOT, DB_PATH))

    print(f"총 {len(all_tasks)}개의 클래스 폴더 작업을 시작합니다.")
    print("-" * 50)

    total_copied = 0
    total_skipped = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_class_dir, all_tasks), 
                            total=len(all_tasks), 
                            desc="Raw 가져오기",
                            unit="class"))
        
        for c, s in results:
            total_copied += c
            total_skipped += s

    print("-" * 50)
    print(f"작업 완료!")
    print(f" - 저장 위치: {OUTPUT_ROOT}")
    print(f" - 총 복사됨: {total_copied}")
    print(f" - 이미 있음(스킵): {total_skipped}")

if __name__ == "__main__":
    main()