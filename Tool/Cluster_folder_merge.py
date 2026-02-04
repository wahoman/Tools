#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_strict_and_clean.py
- 기능: Source(data2) 데이터를 Target(data1)으로 병합
- 핵심: 파일 단위 이동이 아니라, '라벨 줄(Line) 단위'로 읽어서 맞는 폴더에 넣음.
- 장점: 
  1. 폴더 안에 잘못 섞인 다른 ID가 있어도 자동으로 제자리(맞는 클래스 폴더)로 찾아감.
  2. 이미지 중복 시 덮어쓰지 않고 유지.
  3. 모든 처리가 끝나면 Source의 빈 폴더 삭제.
"""

import shutil
import os
from pathlib import Path
from tqdm import tqdm
import yaml

# ==========================================
# ⚙️ 사용자 설정
# ==========================================

# 1. 옮길 데이터 (Source, 사라질 곳)
SRC_ROOT = Path("/home/hgyeo/Desktop/BCAS/BCAS_Origin/NIA 추가한거")

# 2. 합칠 데이터 (Target, 모일 곳)
DST_ROOT = Path("/home/hgyeo/Desktop/BCAS/BCAS_Origin/기존 학습하던거")

# 3. data.yaml 경로 (ID와 폴더명 매핑을 위해 필수)
#    - ID 35가 'Scissors-A' 폴더로 가야 함을 알기 위해 필요합니다.
YAML_PATH = Path("/home/hgyeo/Desktop/yaml/1208.yaml")

# ==========================================
# 🛠️ 로직 시작
# ==========================================

def load_id_map(yaml_path: Path):
    """YAML을 읽어서 {ID: '클래스명'} 딕셔너리 반환"""
    if not yaml_path.exists():
        print(f"❌ YAML 파일이 없습니다: {yaml_path}")
        return None
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        
        names = d.get('names', {})
        # 리스트인 경우 {0: 'name', ...} 변환
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        # 딕셔너리인 경우 {0: 'name', ...} 그대로 사용 (Key를 int로 변환)
        elif isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        else:
            return {}
    except Exception as e:
        print(f"⚠️ YAML 로드 실패: {e}")
        return None

def main():
    if not SRC_ROOT.exists():
        print("❌ Source 경로 없음")
        return

    # 1. ID -> Class Name 매핑 로드
    id_map = load_id_map(YAML_PATH)
    if not id_map:
        print("❌ ID 매핑 정보를 불러올 수 없어 종료합니다.")
        return
    
    print(f"📋 Loaded {len(id_map)} classes from YAML.")
    print(f"🚀 Strict Merge Start: {SRC_ROOT.name} -> {DST_ROOT.name}")

    # 2. Source 내의 모든 라벨 파일 검색 (재귀)
    #    폴더 구조 무시하고 모든 txt를 찾아서 내용물 기준으로 재배치
    src_labels = list(SRC_ROOT.rglob("labels/*.txt"))
    
    for src_lbl_path in tqdm(src_labels, desc="Processing Files"):
        # split(train/valid) 찾기
        # 경로 예: .../train/ClassA/labels/abc.txt -> 'train' 추출
        try:
            # SRC_ROOT 기준으로 상대 경로를 구한 뒤 첫 번째 파트가 train/valid
            rel_path = src_lbl_path.relative_to(SRC_ROOT)
            split = rel_path.parts[0] 
            if split not in ['train', 'valid', 'test']:
                # 바로 아래에 labels가 있는 구조 등 예외 처리
                split = 'train' 
        except:
            split = 'train'

        # 3. 라벨 파일 읽기
        try:
            with open(src_lbl_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue

        if not lines:
            src_lbl_path.unlink() # 빈 파일 삭제
            continue

        # 해당 라벨 파일에 대응하는 이미지 찾기
        # (현재 labels 폴더와 같은 레벨의 images 폴더 가정)
        img_name = src_lbl_path.stem
        src_img_dir = src_lbl_path.parent.parent / "images"
        
        found_img = None
        for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            cand = src_img_dir / (img_name + ext)
            if cand.exists():
                found_img = cand
                break
        
        if found_img is None:
            # 이미지가 없으면 라벨도 의미 없음 -> 삭제
            src_lbl_path.unlink()
            continue

        # 4. 라벨 내용 분석 (ID별 분류)
        #    한 파일 안에 여러 ID가 섞여 있을 수 있음
        content_by_id = {}
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            try:
                class_id = int(parts[0])
                if class_id not in content_by_id:
                    content_by_id[class_id] = []
                content_by_id[class_id].append(line.strip())
            except:
                continue

        # 5. 분류된 내용을 Target 폴더로 분배 (핵심)
        for cid, cls_lines in content_by_id.items():
            # 이 ID가 가야할 폴더명 찾기
            class_name = id_map.get(cid)
            if not class_name:
                print(f"⚠️ Unknown Class ID {cid} found in {src_lbl_path.name}. Skipping.")
                continue

            # Target 경로 설정
            target_class_dir = DST_ROOT / split / class_name
            target_lbl_dir = target_class_dir / "labels"
            target_img_dir = target_class_dir / "images"

            target_lbl_dir.mkdir(parents=True, exist_ok=True)
            target_img_dir.mkdir(parents=True, exist_ok=True)

            # (1) 라벨 쓰기 (Append)
            dst_lbl_file = target_lbl_dir / src_lbl_path.name
            
            # 파일이 없으면 생성, 있으면 추가(append)
            mode = "a" if dst_lbl_file.exists() else "w"
            prefix = "\n" if mode == "a" and dst_lbl_file.stat().st_size > 0 else ""
            
            with open(dst_lbl_file, mode, encoding="utf-8") as f_out:
                f_out.write(prefix + "\n".join(cls_lines))

            # (2) 이미지 복사 (Copy)
            #     주의: 이미지가 이미 있으면 복사하지 않음 (중복 방지)
            dst_img_file = target_img_dir / found_img.name
            if not dst_img_file.exists():
                shutil.copy2(str(found_img), str(dst_img_file))

        # 6. 처리가 끝난 Source 파일 삭제
        #    이미지는 라벨이 모두 처리되었으면 삭제
        src_lbl_path.unlink() # 라벨 원본 삭제
        
        # 이미지는 다른 클래스에서도 참조했을 수 있으므로, 
        # 같은 이름의 txt가 해당 폴더에 더 이상 없을 때만 삭제
        # (하지만 여기선 loop 내에서 처리하므로, 안전하게는 놔두고 나중에 빈폴더 정리로 처리)
        # -> 일단 안전하게 found_img.unlink() 실행 (복사했으므로)
        if found_img.exists():
            found_img.unlink()

    # 7. 빈 폴더 정리
    print("🧹 Cleaning up empty source folders...")
    for root, dirs, files in os.walk(SRC_ROOT, topdown=False):
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except:
                pass # 비어있지 않으면 패스
    
    # 최상위 루트가 비었으면 삭제
    if SRC_ROOT.exists() and not any(SRC_ROOT.iterdir()):
        SRC_ROOT.rmdir()

    print("\n✅ Strict Merge Completed!")

if __name__ == "__main__":
    main()