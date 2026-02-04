import os
import json
import glob

# =========================================================
# [설정] 경로를 지정해주세요
# =========================================================
# 수정할 JSON 파일들이 있는 폴더
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-2\print_json_labels"

# 변경할 경로의 앞부분 (Prefix)
# 기존: "../images/파일명.png"
# 변경: "../print_images/파일명.png"
NEW_PATH_PREFIX = "../print_images/"
# =========================================================

def fix_image_paths(folder_path):
    # 1. JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print("❌ 처리할 JSON 파일이 없습니다.")
        return

    print(f"📂 폴더: {folder_path}")
    print(f"🚀 총 {len(json_files)}개의 파일 경로를 수정합니다...")
    print("-" * 60)

    count = 0
    for json_file in json_files:
        try:
            # 2. JSON 읽기
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 3. imagePath 수정 로직
            original_path = data.get("imagePath", "")
            
            # 경로에서 '파일 이름'만 추출 (예: abc.png)
            filename = os.path.basename(original_path)
            
            # 새로운 경로 조합 (../print_images/ + abc.png)
            new_path = os.path.join(NEW_PATH_PREFIX, filename).replace("\\", "/") # 윈도우 역슬래시 이슈 방지

            # 변경사항이 있을 때만 저장
            if original_path != new_path:
                data['imagePath'] = new_path
                
                # 4. JSON 저장
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                count += 1
                # print(f"수정됨: {filename}") # 너무 많으면 주석 처리

        except Exception as e:
            print(f"❌ 오류 발생 ({os.path.basename(json_file)}): {e}")

    print("-" * 60)
    print(f"✨ 수정 완료! 총 {count}개의 파일 내 경로를 '{NEW_PATH_PREFIX}...'로 변경했습니다.")

if __name__ == "__main__":
    fix_image_paths(json_folder)