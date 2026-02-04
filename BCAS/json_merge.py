import os
import json
import shutil
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial
import time

# =========================================================
# [설정] 경로를 지정해주세요
# =========================================================
folder_a = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\Laptop_json_labels" 
folder_b = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\Object_json_labels"
output_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\json_labels"
# =========================================================

def process_single_file(filename, dir_a, dir_b, dir_out):
    """
    개별 파일을 처리하는 작업자(Worker) 함수입니다.
    이 함수가 여러 CPU 코어에서 동시에 실행됩니다.
    """
    path_a = os.path.join(dir_a, filename)
    path_b = os.path.join(dir_b, filename)
    path_out = os.path.join(dir_out, filename)
    
    result_status = "SKIP"

    try:
        exists_a = os.path.exists(path_a)
        exists_b = os.path.exists(path_b)

        # CASE 1: 두 폴더에 모두 파일이 존재할 때 (병합 대상)
        if exists_a and exists_b:
            with open(path_a, 'r', encoding='utf-8') as f: data_a = json.load(f)
            with open(path_b, 'r', encoding='utf-8') as f: data_b = json.load(f)

            # [핵심 로직] 병합
            data_a['shapes'].extend(data_b['shapes'])
            
            with open(path_out, 'w', encoding='utf-8') as f:
                json.dump(data_a, f, indent=2, ensure_ascii=False)
            
            result_status = "MERGED"

        # CASE 2: A에만 있을 때 (복사)
        elif exists_a:
            shutil.copy(path_a, path_out)
            result_status = "COPY_A"

        # CASE 3: B에만 있을 때 (복사)
        elif exists_b:
            shutil.copy(path_b, path_out)
            result_status = "COPY_B"
            
    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"

    return result_status

def main():
    # 윈도우 환경에서 멀티프로세싱 사용 시 필수
    freeze_support()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 출력 폴더 생성: {output_folder}")

    print("🔍 파일 목록 스캔 중...")
    # 두 폴더의 파일 목록 가져오기
    files_a = set(os.listdir(folder_a)) if os.path.exists(folder_a) else set()
    files_b = set(os.listdir(folder_b)) if os.path.exists(folder_b) else set()

    # 모든 유니크한 파일명 (JSON만 필터링)
    all_files = [f for f in (files_a | files_b) if f.endswith('.json')]
    total_files = len(all_files)

    if total_files == 0:
        print("❌ 처리할 JSON 파일이 없습니다.")
        return

    # CPU 코어 수 확인 (최대한 활용)
    num_cores = cpu_count()
    print(f"🚀 병렬 처리 시작! (총 {total_files}개 파일 / CPU 코어 {num_cores}개 사용)")
    print("=" * 60)

    start_time = time.time()

    # 결과 카운트용
    stats = {"MERGED": 0, "COPY_A": 0, "COPY_B": 0, "ERROR": 0}

    # 경로 인자를 고정한 함수 생성 (partial 사용)
    worker_func = partial(process_single_file, dir_a=folder_a, dir_b=folder_b, dir_out=output_folder)

    # Pool을 사용하여 병렬 실행
    with Pool(processes=num_cores) as pool:
        # imap_unordered가 순서 상관없이 처리되는대로 결과를 뱉어서 조금 더 효율적임
        for i, res in enumerate(pool.imap_unordered(worker_func, all_files), 1):
            
            if res.startswith("ERROR"):
                print(f"\n❌ {res}")
                stats["ERROR"] += 1
            else:
                stats[res] += 1

            # 진행 상황 표시 (너무 자주 출력하면 느려지므로 100개 단위나 1% 단위로 출력)
            if i % 100 == 0 or i == total_files:
                print(f"\r >>> 진행률: {i}/{total_files} ({(i/total_files)*100:.1f}%)", end="")

    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"🎉 작업 완료! (소요 시간: {end_time - start_time:.2f}초)")
    print(f" - 🧩 병합됨 (A+B) : {stats['MERGED']}개")
    print(f" - 📄 복사됨 (A)   : {stats['COPY_A']}개")
    print(f" - 📄 복사됨 (B)   : {stats['COPY_B']}개")
    print(f" - ⚠️ 에러 발생    : {stats['ERROR']}개")
    print(f"📂 결과 폴더: {output_folder}")

if __name__ == "__main__":
    main()