import os
import json

# =========================================================
# [설정] JSON 파일들이 들어있는 폴더 경로를 입력하세요.
# =========================================================
LABEL_FOLDER = r"C:\Users\hgy84\Desktop\BCAS\DAY3\labels"

# 이미지 확장자 (보통 .png 또는 .jpg)
# JSON 안에 적힌 확장자를 그대로 쓰려면 None으로 두세요. (자동 감지)
# 강제로 지정하려면 ".png" 처럼 적으세요.
FORCE_EXTENSION = ".png" 
# =========================================================

def sync_image_path():
    print(f"📂 작업 경로: {LABEL_FOLDER}")
    
    if not os.path.exists(LABEL_FOLDER):
        print("❌ 폴더가 존재하지 않습니다.")
        return

    json_files = [f for f in os.listdir(LABEL_FOLDER) if f.endswith('.json')]
    count = 0

    print(f"🔍 총 {len(json_files)}개의 JSON 파일을 발견했습니다. 동기화 시작...\n")

    for filename in json_files:
        file_path = os.path.join(LABEL_FOLDER, filename)
        
        try:
            # 1. JSON 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. 현재 JSON 파일 이름에서 확장자 제거 (예: A.json -> A)
            current_name_no_ext = os.path.splitext(filename)[0]

            # 3. 기존 imagePath 정보 분석
            old_image_path = data.get('imagePath', '')
            
            # 기존 경로에 폴더 정보가 있었다면 유지 (예: ../images/old.png -> ../images/)
            dir_prefix = os.path.dirname(old_image_path)
            
            # 확장자 결정 (강제 지정 또는 기존 확장자 유지)
            if FORCE_EXTENSION:
                ext = FORCE_EXTENSION
            else:
                _, ext = os.path.splitext(old_image_path)
                if not ext: ext = ".png" # 정보가 없으면 기본 .png

            # 4. 새로운 imagePath 생성
            # 예: prefix가 있으면 "../images/새이름.png", 없으면 "새이름.png"
            if dir_prefix:
                # 윈도우 경로(\)를 리눅스/웹 표준(/)으로 변경
                new_image_path = os.path.join(dir_prefix, current_name_no_ext + ext).replace("\\", "/")
            else:
                new_image_path = current_name_no_ext + ext

            # 5. 변경사항이 있을 때만 저장
            if data['imagePath'] != new_image_path:
                print(f"   🔄 수정: {filename}")
                print(f"      ㄴ 기존: {data['imagePath']}")
                print(f"      ㄴ 변경: {new_image_path}")
                
                data['imagePath'] = new_image_path
                
                # (선택사항) imageData는 경로 의존성을 높이기 위해 비워두는 경우가 많습니다.
                # 필요하다면 아래 주석을 해제하세요.
                # data['imageData'] = None 

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                count += 1
        
        except Exception as e:
            print(f"   ❌ 에러 ({filename}): {e}")

    print("\n" + "="*50)
    print(f"🎉 완료! 총 {count}개의 파일 내용이 수정되었습니다.")
    print("="*50)

if __name__ == "__main__":
    sync_image_path()