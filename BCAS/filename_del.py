import os

# 1. 대상 폴더 경로 설정 (경로 앞에 r을 붙여 백슬래시 오류 방지)
folder_path = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\MS\DAY12-2\object_json_labels"

# 2. 지우고 싶은 파일명에 포함된 특정 단어
keyword = "Knitting"

# 3. 삭제된 파일 개수를 세기 위한 변수
deleted_count = 0

# 폴더 내의 모든 파일 및 폴더 목록 가져오기
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        
        # 파일명에 'Knitting'이 포함되어 있는지 확인
        if keyword in filename:
            file_path = os.path.join(folder_path, filename)
            
            # 해당 경로가 실제 파일인지 확인 (폴더가 아닌 경우에만 삭제)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"✅ 삭제 완료: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 삭제 실패 ({filename}): {e}")
                    
    print(f"\n총 {deleted_count}개의 '{keyword}' 관련 파일이 삭제되었습니다.")
else:
    print("지정하신 폴더 경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")