import os
import glob
from concurrent.futures import ProcessPoolExecutor

# 1. 대상 폴더별 라벨을 추출하는 작업자(Worker) 함수
def extract_classes_from_folder(labels_folder_path):
    unique_classes = set()
    
    # 해당 labels 폴더 안의 모든 txt 파일 검색
    txt_files = glob.glob(os.path.join(labels_folder_path, '*.txt'))
    
    for file_path in txt_files:
        try:
            # 원본을 훼손하지 않기 위해 'r' (읽기 전용) 모드로 열기
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 띄어쓰기를 기준으로 첫 번째 단어(클래스명)만 추출
                        # maxsplit=1로 설정하여 뒤의 수많은 좌표값들은 파싱하지 않아 속도 향상
                        class_name = line.split(maxsplit=1)[0]
                        unique_classes.add(class_name)
        except Exception as e:
            pass # 손상된 파일이나 읽을 수 없는 파일은 무시하고 진행
            
    # 결과를 반환할 때 폴더의 이름(labels의 상위 폴더명)과 정렬된 클래스 목록 반환
    parent_folder_name = os.path.basename(os.path.dirname(labels_folder_path))
    return parent_folder_name, sorted(list(unique_classes))


# 2. 메인 실행 블록 (윈도우 멀티프로세싱 필수 구문)
if __name__ == '__main__':
    # 최상위 경로 설정
    root_dir = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling"
    
    # 작업할 labels 폴더들의 경로를 모두 수집
    target_folders = []
    
    if os.path.exists(root_dir):
        for item in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, item)
            
            # 폴더인 경우에만 진입
            if os.path.isdir(subfolder_path):
                labels_dir = os.path.join(subfolder_path, 'labels')
                
                # 안에 labels 폴더가 실제로 존재하는지 확인
                if os.path.isdir(labels_dir):
                    target_folders.append(labels_dir)
                    
        print(f"총 {len(target_folders)}개의 폴더를 찾았습니다. 멀티프로세싱으로 클래스 분석을 시작합니다...\n")
        print("="*50)
        
        # 3. ProcessPoolExecutor를 이용한 멀티프로세싱 병렬 처리
        # 컴퓨터의 CPU 코어 수에 맞춰 자동으로 프로세스를 할당합니다.
        with ProcessPoolExecutor() as executor:
            # target_folders 리스트의 경로들을 extract_classes_from_folder 함수에 병렬로 던짐
            results = executor.map(extract_classes_from_folder, target_folders)
            
            # 결과 출력
            for folder_name, classes in results:
                if classes:
                    print(f"📁 [{folder_name}] 폴더에 포함된 클래스:")
                    print(f"   -> {', '.join(classes)}\n")
                else:
                    print(f"📁 [{folder_name}] 폴더: 클래스(데이터)가 없습니다.\n")
                    
        print("="*50)
        print("모든 검사가 완료되었습니다.")
        
    else:
        print(f"지정한 경로 '{root_dir}'를 찾을 수 없습니다.")