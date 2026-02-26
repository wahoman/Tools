import os
import glob

# 라벨 파일이 있는 폴더 경로 (r을 붙여서 경로 문자열 오류 방지)
folder_path = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-2\labels"

# 해당 폴더 내의 모든 txt 파일 경로 가져오기
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

# 중복 없는 클래스 이름을 저장할 집합(Set)
unique_classes = set()

# 모든 txt 파일을 하나씩 읽기
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 빈 줄은 건너뜀
            
            # 띄어쓰기를 기준으로 줄을 나누고 첫 번째 요소를 클래스 이름으로 가져옴
            parts = line.split()
            if parts:
                class_name = parts[0]
                unique_classes.add(class_name)

# 결과 출력
print("=== 추출된 전체 클래스 목록 ===")
for cls in sorted(unique_classes):
    print(f"- {cls}")

print(f"\n총 {len(unique_classes)}개의 고유한 클래스가 발견되었습니다.")