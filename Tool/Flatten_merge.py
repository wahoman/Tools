import os, shutil
from pathlib import Path

def merge_label(dst_txt: Path, src_txt: Path):
    """dst_txt가 이미 있으면 줄 병합, 없으면 그대로 이동"""
    if not dst_txt.exists():
        shutil.move(src_txt, dst_txt)
        return

    # 중복 줄 제거하며 병합
    with dst_txt.open() as f:
        exist = {ln.rstrip() for ln in f if ln.strip()}

    with src_txt.open() as f:
        new = [ln.rstrip() for ln in f if ln.strip() and ln.rstrip() not in exist]

    if new:
        with dst_txt.open("a") as f:
            for ln in new:
                f.write(ln + "\n")

    src_txt.unlink()          # 병합 끝난 원본 삭제

def flatten_split(src_split: Path, dst_split: Path):
    dst_img = dst_split / "images"
    dst_lbl = dst_split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    for cls_dir in src_split.iterdir():
        if not cls_dir.is_dir() or cls_dir.name in ("images", "labels"):
            continue

        # ── 이미지 ───────────────────
        for img in (cls_dir / "images").glob("*"):
            tgt = dst_img / img.name
            if tgt.exists():
                img.unlink()           # 같은 이름 이미지는 무시(삭제)
            else:
                shutil.move(img, tgt)

        # ── 라벨 ────────────────────
        for lbl in (cls_dir / "labels").glob("*"):
            merge_label(dst_lbl / lbl.name, lbl)

def move_and_flatten_yolo(src_root: str | Path, dst_root: str | Path):
    """src_root/train, src_root/valid → dst_root/train, dst_root/valid 평탄화"""
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    for split in ("train", "valid"):
        src_split = src_root / split
        dst_split = dst_root / split
        if src_split.exists():
            print(f"▶ {split}  병합 중…")
            flatten_split(src_split, dst_split)
            print(f"✅ {split}  완료")

# ──── 사용 예 ────
SRC = "/home/hgyeo/Desktop/BCAS/data"
DST = "/home/hgyeo/Desktop/BCAS/data_merge"

move_and_flatten_yolo(SRC, DST)
