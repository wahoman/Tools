import os
import glob

# 1. 파일 경로 (r을 붙여서 경로 에러 방지)
folder_path = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY1\labels"

# 2. 바꿀 새로운 이름 (여기에 원하는 클래스명이나 숫자를 넣으세요)
new_name = "Laptop" 

# 해당 폴더의 모든 txt 파일 가져오기
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

print(f"총 {len(txt_files)}개의 파일을 처리합니다...")

count = 0
for file_path in txt_files:
    try:
        # 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified_lines = []
        
        for line in lines:
            parts = line.strip().split() # 공백 기준으로 자르기
            
            # 빈 줄이 아니라면 내용 수정
            if parts:
                parts[0] = new_name  # ★기존 내용 무시하고 무조건 교체★
                new_line = " ".join(parts) + "\n" # 다시 합치기
                modified_lines.append(new_line)
            else:
                # 빈 줄은 그대로 유지 (혹은 삭제하려면 이 부분을 빼세요)
                modified_lines.append(line)

        # 파일 덮어쓰기
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)
            
        count += 1
        # 진행상황 출력이 너무 많으면 아래 줄 주석 처리
        # print(f"[수정 완료] {os.path.basename(file_path)}")

    except Exception as e:
        print(f"[에러 발생] {file_path}: {e}")

print(f"\n작업 완료! 총 {count}개의 파일이 '{new_name}'(으)로 일괄 변경되었습니다.")