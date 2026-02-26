import os
import glob
from concurrent.futures import ThreadPoolExecutor

# --- 실행 부분 ---
target_folder = r'C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY1\labels' 

# 1. 개별 파일을 검사하는 함수 (스레드들이 각각 이 함수를 실행함)
def check_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # [핵심 최적화] readlines() 대신 제너레이터 표현식 사용
            # 파일을 한 번에 다 읽지 않고, 한 줄씩 읽어오며 유효한 줄 개수만 셈
            valid_line_count = sum(1 for line in f if line.strip())
            
            # 조건에 맞으면 파일명과 라벨 수를 반환
            if valid_line_count > 1:
                return os.path.basename(file_path), valid_line_count
                
    except Exception as e:
        print(f"파일 읽기 오류 ({os.path.basename(file_path)}): {e}")
        
    # 조건에 안 맞으면 아무것도 반환하지 않음
    return None

def find_multi_label_files_fast(folder_path):
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    if not txt_files:
        print(f"'{folder_path}' 폴더 안에 .txt 파일이 없습니다.")
        return

    print(f"검색 시작: 총 {len(txt_files)}개의 파일을 검사합니다. (병렬 처리 진행 중...)\n")
    print("--- 2개 이상의 라벨이 있는 파일 목록 ---")

    count = 0
    
    # 2. ThreadPoolExecutor를 활용한 멀티스레딩
    # max_workers는 동시에 일할 작업자 수 (8~16 정도가 적당, 컴퓨터 사양에 따라 조절 가능)
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 여러 파일(txt_files)을 check_file 함수에 동시에 던져서 처리
        results = executor.map(check_file, txt_files)
        
        # 결과 취합 및 출력
        for result in results:
            if result is not None:
                file_name, line_count = result
                print(f"{file_name} (라벨 수: {line_count})")
                count += 1

    print("----------------------------------------")
    print(f"결과: 총 {count}개의 파일이 다중 라벨을 포함하고 있습니다.")

# 폴더 존재 여부 확인 후 실행
if os.path.exists(target_folder):
    find_multi_label_files_fast(target_folder)
else:
    print(f"오류: '{target_folder}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")