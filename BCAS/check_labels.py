import os
import glob

# --- 실행 부분 ---
# 실제 labels 폴더의 경로를 입력하세요. (현재 위치에 있다면 './labels')
target_folder = 'C:/Users/hgy84/Desktop/BCAS/BCAS_Labeling/DAY9-2/labels' 

def find_multi_label_files(folder_path):
    # 폴더 내의 모든 .txt 파일 경로를 리스트로 가져옴
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # 결과가 없을 경우
    if not txt_files:
        print(f"'{folder_path}' 폴더 안에 .txt 파일이 없습니다.")
        return

    print(f"검색 시작: 총 {len(txt_files)}개의 파일을 검사합니다.\n")
    print("--- 2개 이상의 라벨이 있는 파일 목록 ---")

    count = 0
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 공백이나 빈 줄을 제외하고 실제 데이터가 있는 줄만 리스트로 만듦
                valid_lines = [line for line in lines if line.strip()]
                
                # 줄(라벨) 개수가 1개를 초과하는 경우 출력
                if len(valid_lines) > 1:
                    file_name = os.path.basename(file_path)
                    print(f"{file_name} (라벨 수: {len(valid_lines)})")
                    count += 1
                    
        except Exception as e:
            print(f"파일 읽기 오류 ({file_path}): {e}")

    print("----------------------------------------")
    print(f"결과: 총 {count}개의 파일이 다중 라벨을 포함하고 있습니다.")



# 폴더가 실제로 존재하는지 확인 후 실행
if os.path.exists(target_folder):
    find_multi_label_files(target_folder)
else:
    print(f"오류: '{target_folder}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")