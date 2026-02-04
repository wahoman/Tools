#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_and_relabel_final.py
1. YAML 연동: 클래스 이름으로 ID 자동 매핑
2. 폴더명/숫자 혼용 지원
3. 자기 자신 포함 병합 지원 (Safety Logic 추가)
   - 타겟 폴더가 소스 리스트에 있어도 안전하게 ID만 업데이트
   - 이미지 중복 -> 기존 유지
   - 라벨 중복 -> 내용 이어쓰기
"""

import shutil
import os
import yaml
from pathlib import Path

# ==========================================
# ⚙️ 사용자 설정
# ==========================================

BASE_DIR = Path("/home/hgyeo/Desktop/BCAS/BCAS_Origin/APIDS 추가한거")
YAML_PATH = Path("/home/hgyeo/Desktop/yaml/1208.yaml")

# 자기 자신("Bolt-Cutter")을 리스트에 넣어도 이제 안전합니다!
MERGE_PLAN = {
    # 예시: Bolt-Cutter 폴더 + cluster_0 + Monkey-Wrench 폴더 -> 모두 Bolt-Cutter로 통합
    # "Bolt cutter": ["Vise plier-A", "Bolt cutter"],
    
    # "Plastic Pistol": ["Plastic Pistol", "Plastic Pistol-B"],
    # "Pistol": ["Pistol","Plastic Pistol",],  

    # "Smart phone": ["Smart phone","Smart phone1" ],  

    # "Pistol": ["Plastic pistol","Plastic Pistol" ],  
    # # "Nipper": ["Nipper", "Scissors-C"], 

    # "Grenade(Type-A)": ["Grenade(Type-A)", "Plastic Grenade"], 

    # "Awl": ["Awl", "Driver"], 

    # "Battery(Type-A)": ["Battery(Type-A)", "Battery(Type-B)", "Battery(Type-G)"], 

    # "Battery(Type-C)": ["Battery(Type-C)", "Battery(Type-D)"], 

    # "Knife-A": ["Knife-A", "Knife-E"], 

    # "LAGs products(Type-F)": ["LAGs products(Type-E)", "LAGs products(Type-F)"], 

    # "LAGs products(Type-A)": ["LAGs products(Type-A)", "LAGs products(Type-C)"], 
    # "Hex key-A": ["Hex key-A", "Hex key-B"],
    "Bullet": ["Ammunition-A", "Ammunition-B","Ammunition-C","Ammunition-D"], 
    "Ax": ["Axe-A", "Axe-B","Axe-C"],
    "Magazin": ["Magazine"],

    
}

# ==========================================
# 🚀 메인 로직
# ==========================================

def load_class_mapping(yaml_path: Path):
    if not yaml_path.exists():
        print(f"❌ YAML 파일을 찾을 수 없습니다: {yaml_path}")
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    name_to_id = {}
    if isinstance(names, list):
        for idx, name in enumerate(names):
            name_to_id[name] = idx
    elif isinstance(names, dict):
        for idx, name in names.items():
            name_to_id[name] = int(idx)
    return name_to_id

def get_source_folder_name(item):
    if isinstance(item, int):
        return f"cluster_{item}"
    return str(item)

def update_label_content(content: str, new_class_id: int) -> str:
    """텍스트 내용의 맨 앞 숫자를 new_class_id로 교체"""
    lines = content.strip().split('\n')
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))
    return "\n".join(new_lines)

def rewrite_label_in_place(file_path: Path, new_class_id: int):
    """(자기 자신용) 파일을 읽어서 ID만 바꿔서 덮어쓰기"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip(): return
    
    new_content = update_label_content(content, new_class_id)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

def merge_label_file(src_path: Path, dst_path: Path, new_class_id: int):
    """(외부 파일용) 내용을 읽어 ID 변경 후 대상 파일에 이어쓰기(Append)"""
    if not src_path.exists(): return
    with open(src_path, "r", encoding="utf-8") as f:
        src_content = f.read()
    
    modified_content = update_label_content(src_content, new_class_id)
    if not modified_content:
        src_path.unlink()
        return

    # 이미 파일이 있으면 줄바꿈 후 이어쓰기 (Append)
    mode = "a" if dst_path.exists() else "w"
    prefix = "\n" if mode == "a" else ""
    
    with open(dst_path, mode, encoding="utf-8") as f:
        f.write(prefix + modified_content)
    
    src_path.unlink() # 원본 삭제

def main():
    if not BASE_DIR.exists():
        print(f"❌ 작업 경로 없음: {BASE_DIR}")
        return

    name_to_id = load_class_mapping(YAML_PATH)
    if not name_to_id: return

    print(f"📂 작업 경로: {BASE_DIR}")
    
    for split in ["train", "valid"]:
        split_dir = BASE_DIR / split
        if not split_dir.exists(): continue
        
        print(f"\n--- Processing split: {split} ---")

        for target_name, source_list in MERGE_PLAN.items():
            if target_name not in name_to_id:
                print(f"⚠️ [Skip] '{target_name}'는 YAML에 없습니다.")
                continue

            new_id = name_to_id[target_name]
            target_dir = split_dir / target_name
            target_img_dir = target_dir / "images"
            target_lbl_dir = target_dir / "labels"

            # 타겟 폴더 생성
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"   Target: {target_name} (ID: {new_id}) <- {source_list}")

            for src_item in source_list:
                src_folder_name = get_source_folder_name(src_item)
                src_dir = split_dir / src_folder_name

                if not src_dir.exists(): continue

                src_img_dir = src_dir / "images"
                src_lbl_dir = src_dir / "labels"

                # 🔥 핵심 로직: 소스와 타겟이 같은 폴더인지 확인
                is_self = (src_dir.resolve() == target_dir.resolve())

                if is_self:
                    # 1. 자기 자신일 경우: 라벨 ID만 갱신 (이동 X, 삭제 X)
                    if src_lbl_dir.exists():
                        for lbl_file in src_lbl_dir.glob("*.txt"):
                            rewrite_label_in_place(lbl_file, new_id)
                    # print(f"      -> Self update complete ({src_folder_name})")

                else:
                    # 2. 다른 폴더일 경우: 파일 이동 및 병합 (Move & Merge)
                    
                    # 이미지 이동
                    if src_img_dir.exists():
                        for img_file in src_img_dir.glob("*"):
                            dst_file = target_img_dir / img_file.name
                            if not dst_file.exists():
                                shutil.move(str(img_file), str(dst_file))
                            else:
                                # 중복 시 원본 삭제 (Target 유지)
                                img_file.unlink()

                    # 라벨 병합 (이어쓰기)
                    if src_lbl_dir.exists():
                        for lbl_file in src_lbl_dir.glob("*.txt"):
                            dst_lbl = target_lbl_dir / lbl_file.name
                            merge_label_file(lbl_file, dst_lbl, new_id)

                    # 소스 폴더 삭제
                    shutil.rmtree(src_dir, ignore_errors=True)

    print("\n🎉 완료! 중복된 이미지/라벨도 안전하게 병합되었습니다.")

if __name__ == "__main__":
    main()