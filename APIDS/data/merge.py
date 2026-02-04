import os
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# ========================================================
# 설정
# ========================================================
SOURCE_ROOT = r"F:\APIDS\data\APIDS_78class_learning_data_classified"
DEST_ROOT = r"F:\APIDS\data\APIDS_DATA5"

MAX_WORKERS = 20
LOG_INTERVAL = 5000 
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# 전역 변수 및 잠금 장치
counter_lock = threading.Lock()
total_processed = 0
start_time = 0

def copy_worker(src_path, dest_path):
    global total_processed
    
    try:
        # 1. 중복 검사 (이미 있으면 패스)
        if os.path.exists(dest_path):
            result = "skip"
        else:
            # 2. 복사 수행 (메타데이터 없이 내용만 복사 -> 속도 최적화)
            shutil.copyfile(src_path, dest_path)
            result = "copy"
    except Exception:
        result = "error"

    # 3. 카운트 및 로그 출력 (잠금 장치로 동시 접근 제어)
    with counter_lock:
        total_processed += 1
        current = total_processed
        
        # 정확히 5000장 단위일 때만 출력
        if current % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            # 현재 속도 계산 (장/초)
            speed = current / elapsed if elapsed > 0 else 0
            print(f"👉 누적 {current}장 처리 완료... (평균 속도: {speed:.1f}장/초)")

def run_real_time_copy():
    global start_time
    
    # 목적지 폴더 생성
    os.makedirs(DEST_ROOT, exist_ok=True)

    print(f"🚀 [실시간 복사] 발견 즉시 복사합니다.")
    print(f"📂 원본: {SOURCE_ROOT}")
    print(f"📂 타겟: {DEST_ROOT}")
    print(f"⚠️ 참고: 초반에는 RAM 버퍼로 인해 빠르다가, 점차 SSD 실제 속도로 맞춰집니다.")
    print("-" * 50)

    start_time = time.time()
    
    # 스레드 풀 생성
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    # os.walk로 파일을 찾자마자 던짐 (리스트 대기 시간 0초)
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file.lower().endswith(IMAGE_EXTS):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(DEST_ROOT, file)
                
                # 일꾼에게 바로 작업 지시
                executor.submit(copy_worker, src_path, dest_path)

    # 더 이상 찾을 파일이 없으면, 남은 작업이 끝날 때까지 대기
    print("✅ 파일 탐색 종료. 남은 복사 작업을 마무리 중입니다...")
    executor.shutdown(wait=True)

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"🎉 작업 끝!")
    print(f"총 처리 파일: {total_processed}장")
    print(f"총 소요 시간: {elapsed:.1f}초")

if __name__ == "__main__":
    run_real_time_copy()