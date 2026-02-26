import os
import time
from multiprocessing import Pool, cpu_count, freeze_support

# =========================================================
# [설정] 경로, 허용된 클래스 목록, 로그 파일 저장 위치
# =========================================================
base_dir = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY13-2\knitting"

# 오류 결과를 저장할 텍스트 파일 경로 (바탕화면 BCAS 폴더 안에 생성됩니다)
log_file_path = r"C:\Users\hgy84\Desktop\BCAS\error_log.txt"

# 허용된 클래스 목록 (정확히 일치해야 함)
allowed_classes = {
    "Knitting", "Matchbox", "Printer-Cartridge", "Razor", 
    "Laptop", "Scissors", "Knives", "Wrenches"
}
# =========================================================

def check_single_file(file_path):
    """
    개별 텍스트 파일을 검사하는 Worker 함수 (원본 훼손 X, 읽기 전용)
    """
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                    
                class_name = parts[0]
                
                # 클래스 이름이 허용 목록에 없으면 에러 목록에 추가
                if class_name not in allowed_classes:
                    errors.append(f"⚠️ 파일: {file_path} -> {line_num}번째 줄, 잘못된 클래스명: '{class_name}'")
                    
    except Exception as e:
        errors.append(f"❌ 파일 읽기 에러 ({file_path}): {e}")
        
    return errors

def main():
    # 윈도우 환경 멀티프로세싱 필수
    freeze_support()

    print(f"📂 탐색 대상: {base_dir}")
    print(f"✅ 허용된 클래스: {', '.join(allowed_classes)}")
    print("🔍 지정된 폴더의 labels 경로만 스캔 중...")

    target_files = []

    # 1. 파일 스캔 (DAY 폴더 직속 하위의 labels 폴더만 탐색)
    if os.path.exists(base_dir):
        for day_folder in os.listdir(base_dir):
            day_path = os.path.join(base_dir, day_folder)
            
            if os.path.isdir(day_path):
                labels_dir = os.path.join(day_path, "labels")
                
                if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
                    for f_name in os.listdir(labels_dir):
                        if f_name.endswith('.txt'):
                            target_files.append(os.path.join(labels_dir, f_name))
    else:
        print("❌ 기본 경로를 찾을 수 없습니다.")
        return

    total_files = len(target_files)
    if total_files == 0:
        print("❌ 검사할 .txt 파일이 없습니다.")
        return

    num_cores = cpu_count()
    print(f"🚀 병렬 검사 시작! (총 {total_files}개 파일 / CPU 코어 {num_cores}개)")
    print("=" * 70)

    start_time = time.time()
    
    # 텍스트 파일에 저장할 에러 내역을 모아둘 리스트
    all_errors_collected = []

    # 2. 멀티프로세싱 검사 진행
    with Pool(processes=num_cores) as pool:
        for i, errors_found in enumerate(pool.imap_unordered(check_single_file, target_files), 1):
            
            # 에러가 발견되면 리스트에 추가하고 화면에도 일부 출력
            if errors_found:
                for err in errors_found:
                    print(f"\n{err}")
                    all_errors_collected.append(err)
                
            # 진행률 업데이트
            if i % 1000 == 0 or i == total_files:
                print(f"\r >>> 진행률: {i}/{total_files} ({(i/total_files)*100:.1f}%) 탐색 완료", end="")

    end_time = time.time()
    print("\n" + "=" * 70)
    print(f"🎉 검사 완료! (소요 시간: {end_time - start_time:.2f}초)")
    
    # 3. 에러 발생 시 로그 파일(txt) 생성 및 저장
    if all_errors_collected:
        print(f"🚨 총 {len(all_errors_collected)}건의 잘못된 클래스가 발견되었습니다.")
        
        try:
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(f"=== 라벨 클래스 오류 검사 결과 (총 {len(all_errors_collected)}건) ===\n")
                log_file.write(f"검사 일시: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("-" * 70 + "\n")
                
                for error_msg in all_errors_collected:
                    log_file.write(error_msg + "\n")
                    
            print(f"💾 전체 오류 내역이 파일로 저장되었습니다: {log_file_path}")
            
        except Exception as e:
            print(f"❌ 로그 파일 저장 중 오류 발생: {e}")
            
    else:
        print("✅ 완벽합니다! 잘못된 클래스 이름이 하나도 없습니다.")

if __name__ == "__main__":
    main()