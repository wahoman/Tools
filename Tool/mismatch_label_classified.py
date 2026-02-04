#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO 데이터셋 클래스별 자동 분류기 (통합 버전)

[설정]
COPY_MODE = True  : 원본_classified 폴더를 만들어 복사 (원본 보존)
COPY_MODE = False : 원본 폴더 내부에서 이동 및 정리 (원본 변경)
"""

from pathlib import Path
from collections import defaultdict
import shutil, yaml, sys, csv

# ═════ 사용자 설정 ════════════════════════════════════════════════════════
# 1. 작업할 원본 데이터 폴더 경로
SRC_ROOT = Path(r"C:\Users\hgy84\Desktop\0520\0512_class_split\base_data_by_class")

# 2. YAML 파일 경로 (클래스 이름 매핑용)
YAML     = Path(r"C:\Users\hgy84\Desktop\0520\NIA.yaml")

# 3. 모드 설정 (True: 복사 / False: 이동)
COPY_MODE = True  

# 기타 설정
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ════════════════════════════════════════════════════════════════════════


def load_yaml(path: Path) -> dict[int, str]:
    if not path.exists():
        sys.exit(f"❌ YAML 경로가 없습니다: {path}")
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = d.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: (v if v else f"cls_{i}") for i, v in enumerate(names)}
    return {}

def find_img_for_txt(txt: Path) -> Path | None:
    # labels 폴더와 형제인 images 폴더 찾기
    # ../labels/file.txt -> ../images/file.jpg
    img_dir = txt.parent.parent / "images"
    if not img_dir.exists():
        # 혹시 구조가 다르다면 txt 옆에 있는지 확인 (fallback)
        img_dir = txt.parent 
    
    stem = txt.stem
    for ext in IMG_EXTS:
        cand = img_dir / (stem + ext)
        if cand.exists():
            return cand
    return None

def main():
    if not SRC_ROOT.exists():
        sys.exit(f"❌ 원본 경로가 없습니다: {SRC_ROOT}")

    # ── [1] 목적지 경로 설정 ────────────────────────────────
    if COPY_MODE:
        # 복사 모드: 원본폴더명_classified 생성
        DST_ROOT = SRC_ROOT.parent / (SRC_ROOT.name + "_classified")
        mode_str = "복사(Copy)"
        if DST_ROOT.exists():
            # 안전을 위해 기존 결과 폴더 삭제 후 재생성 (선택사항)
            shutil.rmtree(DST_ROOT)
        DST_ROOT.mkdir(parents=True, exist_ok=True)
    else:
        # 이동 모드: 원본 폴더 그 자체
        DST_ROOT = SRC_ROOT
        mode_str = "이동(Move)"

    print(f"🚀 작업 시작: {mode_str} 모드")
    print(f"   원본: {SRC_ROOT}")
    print(f"   대상: {DST_ROOT}\n")

    id2name = load_yaml(YAML)
    stats = defaultdict(lambda: defaultdict(int)) # 통계용

    # ── [2] 파일 순회 및 처리 ────────────────────────────────
    for split in ("train", "valid", "test"):
        src_split_dir = SRC_ROOT / split
        if not src_split_dir.exists():
            continue

        # 처리 중 리스트가 변하지 않게 list로 감쌈
        # 라벨 파일 기준 탐색 (labels 폴더 안에 있는 것만)
        txt_files = list((src_split_dir / "labels").glob("*.txt"))
        
        print(f"📂 {split} 처리 중... ({len(txt_files)}개 파일)")

        for txt in txt_files:
            # 이미지 찾기
            img = find_img_for_txt(txt)
            if img is None:
                print(f"⚠️  이미지 없음: {txt.name}")
                continue

            # 라벨 내용 읽기 및 클래스 분류
            try:
                lines = [ln.strip() for ln in txt.read_text(encoding='utf-8').splitlines() if ln.strip()]
            except:
                continue # 빈 파일 등 예외

            if not lines:
                continue

            # 파일 내에 존재하는 클래스 ID 집합
            id_set = set()
            line_groups = defaultdict(list)
            
            for ln in lines:
                try:
                    cid = int(ln.split()[0])
                    id_set.add(cid)
                    line_groups[cid].append(ln)
                except:
                    pass

            # ── [3] 클래스별 폴더로 분배 ────────────────────────
            for cid in id_set:
                cname = id2name.get(cid, f"cls_{cid}")
                
                # 목표 폴더: DST_ROOT / split / class_name / images|labels
                target_img_dir = DST_ROOT / split / cname / "images"
                target_lbl_dir = DST_ROOT / split / cname / "labels"
                
                target_img_dir.mkdir(parents=True, exist_ok=True)
                target_lbl_dir.mkdir(parents=True, exist_ok=True)

                # 3-1. 이미지 처리
                dst_img = target_img_dir / img.name
                if not dst_img.exists():
                    shutil.copy2(img, dst_img) # 이동 모드라도 일단 복사(안전)
                    stats[cname][split] += 1
                
                # 3-2. 라벨 처리 (해당 클래스 라벨만 추출해서 저장)
                dst_txt = target_lbl_dir / txt.name
                with open(dst_txt, "a", encoding="utf-8") as f:
                    for ln in line_groups[cid]:
                        f.write(ln + "\n")

            # ── [4] 이동 모드일 경우 원본 삭제 (Clean up) ────────
            if not COPY_MODE:
                # 원본 라벨 삭제
                try: txt.unlink() 
                except: pass
                
                # 원본 이미지 삭제 (단, 다른 txt가 이 이미지를 안 쓸 때만)
                # 보통 YOLO 구조에선 1:1 대응이므로 바로 지워도 되지만 안전하게
                try: img.unlink() 
                except: pass

    # ── [5] 통계 및 마무리 ──────────────────────────────────
    
    # 이동 모드일 경우 빈 껍데기 폴더(images, labels)가 남을 수 있음 -> 정리
    if not COPY_MODE:
        for split in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                d = SRC_ROOT / split / sub
                if d.exists() and not any(d.iterdir()):
                    try: d.rmdir() # 비어있으면 삭제
                    except: pass

    print("\n📊 클래스별 분류 통계:")
    print(f"{'Class':<20} {'Train':>7} {'Valid':>7} {'Total':>7}")
    print("-" * 45)
    
    # CSV 저장 (어느 모드든 통계는 생성)
    csv_path = DST_ROOT / "class_stats.csv"
    
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Class Name", "Train", "Valid", "Total"])
            
            for cname in sorted(stats.keys()):
                tr = stats[cname]["train"]
                va = stats[cname]["valid"]
                tot = tr + va
                print(f"{cname:<20} {tr:7} {va:7} {tot:7}")
                writer.writerow([cname, tr, va, tot])
    except Exception as e:
        print(f"⚠️ CSV 저장 실패: {e}")

    print("-" * 45)
    print(f"✅ 작업 완료! 저장 위치: {DST_ROOT}")
    if COPY_MODE:
        print(f"   (원본 폴더는 보존되었습니다: {SRC_ROOT})")
    else:
        print(f"   (원본 폴더가 재구성되었습니다)")

if __name__ == "__main__":
    main()