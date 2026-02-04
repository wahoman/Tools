#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2_crop_objects_v6.py (파일명 중간 삽입 + 카운팅)
 · 단일 객체: 원본 이름 유지 (예: Knife_3.png)
 · 다중 객체: 순번을 '맨 뒤 숫자' 앞에 삽입 (예: Knife_3 -> Knife_0_3.png)
   -> 이렇게 해야 합성기가 맨 뒤의 '3'을 보고 가방과 매칭할 수 있음.
 · 마지막에 다중 객체 파일이 몇 개였는지 리포트 출력.
"""

import os
import re
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ── ★ 사용자 설정 ───────────────────────────────────────────────
SRC_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image_png") 
DST_ROOT = Path(r"D:\hgyeo\BCAS_TIP\bare_image_crop2")

# 투명 배경(Alpha) 크롭이므로 여백 0 추천
MARGIN   = -1  
# ────────────────────────────────────────────────────────────────

def find_image_path(txt_path: Path):
    """이미지 파일 찾기"""
    class_dir = txt_path.parent.parent
    images_dir = class_dir / "images"
    candidates = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG"]
    stem = txt_path.stem
    for ext in candidates:
        img_path = images_dir / (stem + ext)
        if img_path.exists():
            return img_path
    return None

def crop_worker(txt_path: Path):
    """
    Returns:
        1 if multi-object file, 0 if single or skipped
    """
    try:
        # 1. 이미지 읽기
        img_path = find_image_path(txt_path)
        if img_path is None or os.path.getsize(img_path) == 0:
            return 0

        with open(str(img_path), "rb") as stream:
            bytes_data = bytearray(stream.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

        if img is None: return 0

        # 채널 처리
        if len(img.shape) == 2: 
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4: 
            img = img[:, :, :3]

        h_img, w_img = img.shape[:2]

        # 2. 라벨 읽기
        with txt_path.open("r", encoding='utf-8') as f:
            lines = f.readlines()

        # 유효한 라인만 필터링
        valid_lines = [line for line in lines if len(line.strip().split()) >= 5]
        object_count = len(valid_lines)
        
        if object_count == 0: return 0

        class_name = txt_path.parent.parent.name
        save_dir = DST_ROOT / class_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 3. 파일명 파싱 (맨 뒤의 숫자를 분리)
        # 예: Knife_Side_3 -> prefix="Knife_Side", suffix="3"
        stem = txt_path.stem
        match = re.search(r'^(.*)_(\d+)$', stem)
        
        has_suffix_num = False
        if match:
            prefix = match.group(1)
            suffix_num = match.group(2)
            has_suffix_num = True
        else:
            # 숫자로 안 끝나는 파일명일 경우 (그냥 뒤에 붙여야 함)
            prefix = stem
            suffix_num = ""

        # 4. 객체 크롭 루프
        for idx, line in enumerate(valid_lines):
            parts = line.strip().split()
            coords = np.array([float(p) for p in parts[1:]])
            if len(coords) % 2 != 0: continue

            xs = coords[0::2] * w_img
            ys = coords[1::2] * h_img

            # 마스킹
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            poly_points = np.column_stack((xs, ys)).astype(np.int32)
            cv2.fillPoly(mask, [poly_points], 255)

            b, g, r = cv2.split(img)
            bgra = cv2.merge((b, g, r, mask))

            x_min, x_max = int(np.min(xs)), int(np.max(xs))
            y_min, y_max = int(np.min(ys)), int(np.max(ys))

            x1 = max(0, x_min - MARGIN)
            y1 = max(0, y_min - MARGIN)
            x2 = min(w_img, x_max + MARGIN)
            y2 = min(h_img, y_max + MARGIN)

            if x2 <= x1 or y2 <= y1: continue
            crop = bgra[y1:y2, x1:x2]

            # ── [핵심] 저장명 생성 로직 ──
            if object_count == 1:
                # 단일 객체: 원본 이름 유지 (예: Knife_3.png)
                save_name = f"{stem}.png"
            else:
                # 다중 객체: 인덱스를 중간에 삽입 (예: Knife_0_3.png)
                if has_suffix_num:
                    save_name = f"{prefix}_{idx}_{suffix_num}.png"
                else:
                    # 숫자가 없던 파일이면 그냥 뒤에 붙임
                    save_name = f"{stem}_{idx}.png"

            save_path = save_dir / save_name

            ext = os.path.splitext(save_name)[1]
            result, encoded_img = cv2.imencode(ext, crop)
            if result:
                with open(str(save_path), "wb") as f:
                    f.write(encoded_img)

        # 다중 객체 파일이었다면 1 반환 (카운팅용)
        return 1 if object_count > 1 else 0

    except Exception:
        return 0

def main():
    print("✂️ 객체 크롭 작업 시작 (중간 삽입 + 카운팅)...")
    
    if not SRC_ROOT.exists():
        print(f"❌ 경로 없음: {SRC_ROOT}")
        return

    all_txt_files = glob(str(SRC_ROOT / "*" / "labels" / "*.txt"))
    
    if not all_txt_files:
        print(f"❌ 라벨 파일 없음. 경로 확인: {SRC_ROOT}")
        return

    print(f"총 {len(all_txt_files)}개의 라벨 파일 발견.")

    num_workers = max(cpu_count() - 1, 1)
    path_list = [Path(p) for p in all_txt_files]

    # imap을 사용하여 리턴값(0 or 1)을 수집
    multi_object_files_count = 0
    
    with Pool(num_workers) as pool:
        # 결과를 리스트로 받아서 합산
        results = list(tqdm(pool.imap(crop_worker, path_list), total=len(path_list), desc="Smart Cropping"))
        multi_object_files_count = sum(results)

    print("-" * 50)
    print(f"✅ 모든 작업이 완료되었습니다!")
    print(f"📂 저장 경로: {DST_ROOT}")
    print(f"🔢 다중 객체 포함 파일 수: {multi_object_files_count}개")
    print("-" * 50)

if __name__ == "__main__":
    main()