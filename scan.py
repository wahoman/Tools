import os
from collections import Counter

# =========================================================
# [설정] 경로 지정 (백슬래시 2개 주의!)
# =========================================================
target_folder = "D:\\"

# [설정] 검사에서 제외할 폴더 이름들 (여기에 추가하면 됩니다)
IGNORE_FOLDERS = {'Analyzer', 'TIPTool', '$RECYCLE.BIN', 'System Volume Information'}
# =========================================================

def scan_extensions(folder_path):
    if not os.path.exists(folder_path):
        print("❌ 경로를 찾을 수 없습니다.")
        return

    print(f"🔍 경로 스캔 중...: {folder_path}")
    print(f"🚫 제외된 폴더: {', '.join(IGNORE_FOLDERS)}")
    print("잠시만 기다려주세요...")

    ext_counts = Counter()
    total_files = 0

    for root, dirs, files in os.walk(folder_path):
        # -------------------------------------------------
        # [핵심] 제외할 폴더는 탐색 리스트에서 삭제 (하위로 진입 안 함)
        # -------------------------------------------------
        dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

        for file in files:
            try:
                # 확장자 분리 및 카운트
                _, ext = os.path.splitext(file)
                ext = ext.lower() if ext else "[확장자 없음]"
                ext_counts[ext] += 1
                total_files += 1
            except:
                pass # 권한 문제 등으로 접근 불가한 파일은 패스

    print("\n" + "=" * 40)
    print(f"📊 스캔 결과 (총 파일: {total_files:,}개)")
    print("=" * 40)

    if total_files == 0:
        print("파일이 없습니다.")
    else:
        # 개수 많은 순서대로 출력 (상위 20개만 보기 좋게 출력)
        for ext, count in ext_counts.most_common(20):
            print(f"📄 {ext:<15} : {count:,} 개")
            
    print("=" * 40)

if __name__ == "__main__":
    scan_extensions(target_folder)