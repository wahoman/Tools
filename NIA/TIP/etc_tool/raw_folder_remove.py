import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# [설정] 삭제 작업을 수행할 루트 경로
# ==========================================
TARGET_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image")
TARGET_FOLDER_NAME = "raws"  # 삭제할 폴더 이름
# ==========================================

def clean_dataset():
    print(f"대상 경로: {TARGET_ROOT}")
    print(f"삭제 대상: 각 클래스 내부의 '{TARGET_FOLDER_NAME}' 폴더")
    print("-" * 50)

    if not TARGET_ROOT.exists():
        print("❌ 오류: 해당 경로를 찾을 수 없습니다.")
        return

    splits = ['train', 'valid']
    deleted_count = 0
    checked_count = 0

    # 탐색 시작
    for split in splits:
        split_path = TARGET_ROOT / split
        if not split_path.exists():
            continue

        # 클래스 폴더 목록 가져오기
        class_dirs = [d for d in split_path.iterdir() if d.is_dir()]

        print(f"[{split}] 폴더 내 {len(class_dirs)}개 클래스 정리 중...")

        for class_dir in tqdm(class_dirs, desc=f"Cleaning {split}"):
            checked_count += 1
            
            # 삭제할 타겟 폴더 경로 (예: .../Apple/raws)
            target_dir = class_dir / TARGET_FOLDER_NAME

            # 폴더가 존재하면 삭제 (파일 포함 전체 삭제)
            if target_dir.exists() and target_dir.is_dir():
                try:
                    shutil.rmtree(target_dir)
                    deleted_count += 1
                except Exception as e:
                    print(f"\n❌ 삭제 실패 ({class_dir.name}): {e}")

    print("-" * 50)
    print("정리 완료!")
    print(f" - 총 검사한 클래스: {checked_count} 개")
    print(f" - 삭제된 '{TARGET_FOLDER_NAME}' 폴더: {deleted_count} 개")
    print(f"이제 '{TARGET_ROOT}' 안에는 images와 labels만 남았습니다.")

if __name__ == "__main__":
    clean_dataset()