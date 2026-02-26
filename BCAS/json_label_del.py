import os

# 1. 폴더 경로 설정
folder_path = r'C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY13-2\object_json_labels'

def delete_files_with_word(path, target_word):
    if not os.path.exists(path):
        print(f"❌ 폴더를 찾을 수 없습니다: {path}")
        return

    files = os.listdir(path)
    count = 0

    print(f"🚀 '{target_word}'가 포함된 파일 삭제를 시작합니다...")
    print("-" * 50)

    for filename in files:
        # 파일명에 target_word(Knitting)가 포함되어 있는지 확인
        if target_word in filename:
            file_path = os.path.join(path, filename)
            
            try:
                # 파일 삭제 실행
                os.remove(file_path)
                print(f"🗑️ 삭제됨: {filename}")
                count += 1
            except Exception as e:
                print(f"❌ 삭제 실패({filename}): {e}")

    print("-" * 50)
    print(f"✅ 총 {count}개의 파일을 삭제했습니다.")

if __name__ == "__main__":
    # 실행 (단어는 대소문자를 구분하니 주의하세요)
    delete_files_with_word(folder_path, "Printer-Cartridge")