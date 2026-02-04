import shutil
from pathlib import Path

# 원본 루트(라벨을 여기서 가져옴)
SRC_ROOT = Path(r"\\Sstl_nas\ai\datasets\NIA")

# 대상 루트(이미 images만 복사돼 있음)
DST_ROOT = Path(r"\\Sstl_nas\ai\datasets\NIA_new_colormapping")

copied = 0
missing = 0
classes_done = 0

for split in ["train", "valid"]:
    dst_split = DST_ROOT / split
    if not dst_split.exists():
        continue

    # 대상 쪽의 클래스 디렉터리들만 돈다 (images 기준)
    for class_dir in dst_split.iterdir():
        if not class_dir.is_dir():
            continue

        dst_images = class_dir / "images"
        if not dst_images.exists():
            continue

        # 원본 라벨 폴더 경로
        src_labels = SRC_ROOT / split / class_dir.name / "labels"
        if not src_labels.exists():
            print(f"[WARN] 원본 라벨 폴더 없음: {src_labels}")
            continue

        # 대상 라벨 폴더 생성
        dst_labels = class_dir / "labels"
        dst_labels.mkdir(parents=True, exist_ok=True)

        # 대상에 있는 이미지 목록 기준으로 라벨 매칭
        for img_path in dst_images.iterdir():
            if not img_path.is_file():
                continue
            stem = img_path.stem
            src_label_path = src_labels / f"{stem}.txt"
            if src_label_path.exists():
                shutil.copy2(src_label_path, dst_labels / src_label_path.name)
                copied += 1
            else:
                # 라벨이 없는 이미지
                missing += 1

        classes_done += 1
        print(f"[OK] {split}/{class_dir.name} 라벨 복사 완료")

print("\n==== 요약 ====")
print(f"클래스 처리 수 : {classes_done}")
print(f"복사한 라벨 수 : {copied}")
print(f"라벨 없는 이미지 수 : {missing}")
print("✅ labels 복사 완료 (images 구조와 동일하게 생성)")
