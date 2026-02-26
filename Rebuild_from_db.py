import sqlite3
import os
import shutil
import time
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 설정 (경로 확인 필수)
# ==========================================

# [DB 파일] 아까 만든 DB 파일명
DB_FILE_PATH = r"D:\hgyeo\data_version\dataset_1114_optimized.db"

# [타겟] 실제 파일이 복사될 위치
TARGET_RESTORE_FOLDER = r"D:\hgyeo\1114"

# [이미지 창고 1] NIA (NAS) - 언더바가 많은 파일
STORAGE_NIA = r"\\Sstl_nas\ai\hgyeo\DATA\NIA"

# [이미지 창고 2] APIDS (Local) - 언더바가 적은 파일 (2~3개)
STORAGE_APIDS = r"D:\hgyeo\APIDS_16class_learning_data\images"

# [분기 기준] 언더바(_) 개수가 이 숫자 '이하'면 APIDS, '초과'면 NIA
UNDERSCORE_THRESHOLD = 3 

# [성능 설정] CPU 코어 자동 감지 후 -2 적용
try:
    total_cores = os.cpu_count()
    # 최소 1개는 보장, 코어가 많으면 2개 남기고 풀가동
    MAX_WORKERS = max(1, total_cores - 2)
except Exception:
    MAX_WORKERS = 4 # 감지 실패시 기본값
    total_cores = "?"

def process_single_item(args):
    """
    개별 파일 하나를 처리하는 함수 (스레드에서 실행됨)
    """
    row, target_root, path_nia, path_apids = args
    rel_path, filename, label_content = row
    
    # 1. 타겟 폴더 생성 (이미지 경로 기준)
    # 예: D:\hgyeo\1114\train\class_A\images
    target_img_dir = target_root / rel_path
    target_img_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_img_dir / filename
    
    # 이미 파일이 존재하면 스킵 (이어하기 기능)
    if target_file.exists():
        return "SKIPPED"

    # 2. 이미지 소스 찾기 (라우팅 로직)
    underscore_count = filename.count('_')
    
    if underscore_count <= UNDERSCORE_THRESHOLD:
        source_file = path_apids / filename
        source_name = "APIDS"
    else:
        source_file = path_nia / filename
        source_name = "NIA"

    # 3. 진짜 파일 복사 (Copy)
    if source_file.exists():
        try:
            shutil.copy2(source_file, target_file)
        except Exception as e:
            return f"ERROR_COPY: {filename} ({e})"
    else:
        return f"MISSING: {filename} (in {source_name})"

    # 4. 라벨 파일 생성
    # DB에 라벨 내용이 있는 경우에만 생성
    if label_content:
        try:
            parts = list(target_img_dir.parts)
            # 대소문자 무시하고 'images' 찾기
            parts_lower = [p.lower() for p in parts]
            
            if 'images' in parts_lower:
                # 뒤에서부터 images를 찾아 labels로 변경
                idx = len(parts) - 1 - parts_lower[::-1].index('images')
                parts[idx] = 'labels'
                target_label_dir = Path(*parts)
                target_label_dir.mkdir(parents=True, exist_ok=True)
                
                label_file = target_label_dir / Path(filename).with_suffix('.txt')
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write(label_content)
        except Exception as e:
            return f"ERROR_LABEL: {filename} ({e})"

    return "SUCCESS"

def main():
    print(f"🚀 [복원 시작] 실제 파일 복사 모드")
    print(f"   - 타겟: {TARGET_RESTORE_FOLDER}")
    print(f"   - 분류: 언더바 {UNDERSCORE_THRESHOLD}개 이하->APIDS, 초과->NIA")
    print(f"   - 성능: 전체 코어 {total_cores}개 중 {MAX_WORKERS}개 사용 (2개 여유)")
    
    # DB 연결 및 데이터 조회
    if not os.path.exists(DB_FILE_PATH):
        print(f"❌ 오류: DB 파일이 없습니다 -> {DB_FILE_PATH}")
        return

    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT relative_path, filename, label_content FROM dataset")
    rows = cursor.fetchall()
    conn.close()
    
    total_files = len(rows)
    if total_files == 0:
        print("❌ DB에 데이터가 없습니다.")
        return

    print(f"📦 총 {total_files}개의 파일을 복원합니다.")
    
    target_root = Path(TARGET_RESTORE_FOLDER)
    path_nia = Path(STORAGE_NIA)
    path_apids = Path(STORAGE_APIDS)
    
    # 진행 상황 집계용
    stats = {"SUCCESS": 0, "SKIPPED": 0, "MISSING": 0, "ERROR": 0}
    processed_count = 0
    start_time = time.time()

    # 멀티스레드 실행
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 작업 패키징
        futures = [
            executor.submit(process_single_item, (row, target_root, path_nia, path_apids))
            for row in rows
        ]
        
        # 결과 처리
        for future in as_completed(futures):
            result = future.result()
            processed_count += 1
            
            if result == "SUCCESS":
                stats["SUCCESS"] += 1
            elif result == "SKIPPED":
                stats["SKIPPED"] += 1
            elif result.startswith("MISSING"):
                stats["MISSING"] += 1
                # print(f"🚫 {result}") # 너무 많이 뜨면 주석 처리하세요
            else: # ERROR
                stats["ERROR"] += 1
                print(f"❌ {result}")

            # 1000개마다 진행률 표시
            if processed_count % 1000 == 0:
                elapsed = time.time() - start_time
                speed = processed_count / elapsed
                percent = (processed_count / total_files) * 100
                print(f"▶ {percent:.1f}% 완료 ({processed_count}/{total_files}) - 속도: {speed:.1f}장/초")

    print("-" * 50)
    print("🎉 복원 작업 완료!")
    print(f"✅ 성공: {stats['SUCCESS']}개")
    print(f"⏭️ 스킵(이미 존재): {stats['SKIPPED']}개")
    print(f"🚫 누락(창고에 없음): {stats['MISSING']}개")
    print(f"⚠️ 에러: {stats['ERROR']}개")

if __name__ == "__main__":
    main()