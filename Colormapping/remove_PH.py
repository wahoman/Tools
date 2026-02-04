import os
from pathlib import Path
from tqdm import tqdm

def remove_ph_suffix(root_path):
    """
    root_path (예: E:\\NIA) 하위의 train/valid 안 images 폴더에 있는
    모든 파일 이름에서 '_PH' 접미사를 제거합니다.
    """
    renamed_count = 0
    skipped_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        if os.path.basename(dirpath).lower() == "images":
            class_name = Path(dirpath).parent.name
            print(f"\n▶ 현재 폴더 처리 중: {class_name}\\images")

            for filename in filenames:
                old_path = Path(dirpath) / filename
                stem, ext = old_path.stem, old_path.suffix

                if stem.lower().endswith("_ph"):
                    new_stem = stem[:-3]  # "_PH" 제거
                    new_name = new_stem + ext
                    new_path = Path(dirpath) / new_name

                    try:
                        # 중복 방지
                        if new_path.exists():
                            skipped_count += 1
                            tqdm.write(f"[경고] 이미 존재 → 건너뜀: {new_path}")
                            continue

                        old_path.rename(new_path)
                        renamed_count += 1
                    except Exception as e:
                        tqdm.write(f"[에러] {old_path} → {new_path} 이름 변경 실패: {e}")
                else:
                    skipped_count += 1

    print("\n✨ 작업 완료!")
    print(f"- 이름 변경된 파일 수: {renamed_count}")
    print(f"- 건너뛴 파일 수: {skipped_count}")

if __name__ == "__main__":
    ROOT = r"\\Sstl_nas\ai\datasets\NIA_new_colormapping"   # train/valid 모두 포함
    remove_ph_suffix(ROOT)
