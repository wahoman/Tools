#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TIP 합성 파이프라인 - 최종 완성본
 · 배경(Raw): Z-eff 물리 기반 컬러매핑 적용
 · 대상(PNG): 투명 배경(Alpha)을 살려서 깔끔하게 합성
"""

import os
import random
import numpy as np
import cv2
from glob import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ── ★ 사용자 설정 ───────────────────────────────────────────────
TIP_ROOT        = Path("D:/hgyeo/BCAS_TIP/bare_image_crop")
BG_ROOT         = Path("D:/hgyeo/BCAS_TIP/APIDS Bare Bags/Bag5")
DST_ROOT        = Path("D:/hgyeo/BCAS_TIP/TIP_output")

# [핵심 스위치]
DO_PROCESS_BG_RAW  = True   # 배경은 Raw이므로 컬러 변환 수행
DO_PROCESS_TIP_RAW = False  # TIP은 이미 png이므로 변환 안 함

# 필수 LUT 파일 경로
PATH_LUT_CSV    = "LUT_ZeffU2.csv"
PATH_GRAY_CLUT  = "gray.clut"

RAW_WIDTH       = 896     # ★ 중요: 사용하시는 장비의 Raw 가로 해상도 확인 필요
TARGET_PER_CLS  = 10_000
NUM_WORKERS     = max(cpu_count() - 1, 1)
SEED            = 42
# ────────────────────────────────────────────────────────────────
random.seed(SEED)

# ╔═══════════════════════════════════════════════════════════════╗
# ║               1. 물리 기반 컬러매핑 엔진 (NumPy)                  ║
# ╚═══════════════════════════════════════════════════════════════╝

_LUT_Z = None
_LUT_K = None
_ZEFF_TABLE = None
_ZI0 = np.array([98000]*9, dtype=np.float32)

def load_luts():
    global _LUT_Z, _LUT_K, _ZEFF_TABLE
    
    # 1. Zeff LUT
    pseudoZ = np.zeros((300, 256, 3), dtype=np.uint8)
    try:
        with open(PATH_LUT_CSV, "rb") as f:
            lines = [r.strip() for r in f.read().decode("utf-16").splitlines() if r.strip()]
        for j, line in enumerate(lines[:300]):
            parts = line.split(",")
            if len(parts) >= 256*3:
                arr = np.array(parts[:256*3], dtype=np.int16).reshape(256, 3)
                pseudoZ[j] = np.clip(arr, 0, 255).astype(np.uint8)
        _LUT_Z = pseudoZ
        print(f"[OK] {PATH_LUT_CSV} 로드 완료.")
    except Exception as e:
        print(f"[Error] LUT CSV 로드 실패: {e}")
        _LUT_Z = None

    # 2. Gray CLUT
    try:
        with open(PATH_GRAY_CLUT, "rb") as f:
            buf = f.read()[8:]
        _LUT_K = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4)[:256, :3].copy()
        print(f"[OK] {PATH_GRAY_CLUT} 로드 완료.")
    except Exception as e:
        print(f"[Error] Gray CLUT 로드 실패: {e}")
        _LUT_K = np.zeros((256, 3), dtype=np.uint8)

    # 3. Build Zeff Table
    _Z_base = np.array([[1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,1.206871776,
                         1.215205132,1.226352526,1.240297472,1.257289302,1.27849552,1.304551537,
                         1.334194598,1.367066563,1.402681085,1.440472022,1.479718307,1.519644992,
                         1.559941821,1.599809092,1.638299752,1.675213149,1.709797614,1.742333096,
                         1.772344093,1.799326033,1.823810917,1.846055878,1.864917778,1.881765843]]*9, dtype=np.float32)
    
    luts = np.zeros((9, 2000), dtype=np.uint16)
    noffset = 1000
    for d in range(9):
        nZ = np.zeros(31, dtype=np.int32)
        vals = np.maximum(((_Z_base[d] + 0.0005) * 1000).astype(np.int32) - noffset, 0)
        nZ[:30], nZ[30] = vals, 2000
        i = 0
        for _ in range(int(nZ[0])):
            if i >= 2000: break
            luts[d, i], i = (1 << 8), i + 1
        for j in range(30):
            if i >= 2000: break
            nW = int(nZ[j+1] - nZ[j])
            if nW <= 0: continue
            k = np.arange(nW, dtype=np.int32)
            seg = ((j+1) << 8) + ((k << 8) // nW)
            end = min(i + nW, 2000)
            luts[d, i:end] = seg[:(end - i)].astype(np.uint16)
            i = end
        while i < 2000: luts[d, i], i = (30 << 8), i + 1
    _ZEFF_TABLE = luts

load_luts()

def process_raw_to_color(raw_path: Path):
    if _LUT_Z is None or _ZEFF_TABLE is None: return None
    try:
        arr = np.fromfile(str(raw_path), dtype=np.uint16)
    except: return None
        
    if arr.size == 0 or arr.size % RAW_WIDTH != 0: return None
    img = arr.reshape(-1, RAW_WIDTH)
    rows = img.shape[0]
    if rows < 2 or (rows % 2) != 0: return None

    # 배경 레벨링
    bin_center = 50000
    idx = np.floor((img.astype(np.int32) - bin_center) / 16 + 0.5 + 128).astype(np.int32)
    np.clip(idx, 0, 255, out=idx)
    hist = np.bincount(idx.ravel(), minlength=256)
    peak = int(np.argmax(hist))
    bkg_lvl = float(bin_center + 16 * (peak + 0.5 - 128))
    
    g = img.astype(np.float32)
    scale = 65535.0 / max(bkg_lvl, 1.0)
    g *= scale
    g = np.clip(g, 0, 65535).astype(np.uint16)

    # High / Low 분리
    vd = rows // 2
    top, bottom = g[:vd, :], g[vd:, :]
    if top.mean() >= bottom.mean():
        imgH, imgL = top, bottom
    else:
        imgH, imgL = bottom, top

    # Z-eff 계산
    det_idx = 0 
    Ilow = np.maximum(imgL.astype(np.float32), 1.0)
    Ihigh = np.maximum(imgH.astype(np.float32), 1.0)
    zI0 = _ZI0[det_idx]
    
    num = np.log(zI0 / Ilow)
    den = np.log(zI0 / Ihigh)
    den = np.where(den <= 1e-12, 1e-12, den)
    
    rate = np.clip(((num/den - 1.0 + 0.0005) * 1000.0).astype(np.int32), 0, 1999)
    ze_val = _ZEFF_TABLE[det_idx][rate]

    # 컬러 매핑
    pbSrc = (imgH >> 8).astype(np.uint16)
    nzeff = np.clip(((ze_val.astype(np.int32) * 10) // 256) - 1, 0, 299)
    mask_invalid = nzeff < 0
    
    color_K = _LUT_K[pbSrc] 
    color_Z = _LUT_Z[nzeff, pbSrc]
    rgb = np.where(mask_invalid[..., None], color_K, color_Z)
    bgr = np.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0]], axis=-1)
    
    return bgr.astype(np.uint8)

# ╔═══════════════════════════════════════════════════════════════╗
# ║                  2. 합성 파이프라인 (Overlay)                   ║
# ╚═══════════════════════════════════════════════════════════════╝

def random_scale(img, lo=0.4, hi=1.0):
    s = random.uniform(lo, hi)
    return cv2.resize(img, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

def overlay_images(bg, fg, x, y):
    """
    배경(BGR)에 전경(BGRA or BGR)을 합성
    전경의 Alpha 채널이 있으면 그것을 마스크로 사용
    """
    h, w = fg.shape[:2]
    roi = bg[y:y+h, x:x+w]
    
    # 1. 전경이 4채널(투명 배경)인 경우: Alpha 채널 사용
    if fg.shape[2] == 4:
        # Alpha 채널 분리 (0~255) -> 0~1.0 float으로 변환
        alpha = fg[:, :, 3] / 255.0
        fg_bgr = fg[:, :, :3]
        
        # 알파 블렌딩 공식: (전경 * alpha) + (배경 * (1-alpha))
        for c in range(3):
            roi[:, :, c] = (alpha * fg_bgr[:, :, c] + (1.0 - alpha) * roi[:, :, c])
            
    # 2. 전경이 3채널(검은 배경)인 경우: 임계값 마스킹
    else:
        fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(fg_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg_fg = cv2.bitwise_and(fg, fg, mask=mask)
        roi = cv2.add(bg_bg, fg_fg)

    # 합성 결과 적용
    bg[y:y+h, x:x+w] = roi.astype(np.uint8)
    return bg

def process_single_mix(bg_p: Path, tip_p: Path, out_stem: Path):
    # 1. 배경 이미지 (Raw -> Color)
    if DO_PROCESS_BG_RAW and bg_p.suffix.lower() == '.raw':
        bg = process_raw_to_color(bg_p)
    else:
        bg = cv2.imread(str(bg_p), cv2.IMREAD_COLOR)
        
    # 2. TIP 이미지 (PNG -> Read Alpha)
    # ★ 수정됨: IMREAD_UNCHANGED로 읽어야 투명도가 살아납니다.
    if DO_PROCESS_TIP_RAW and tip_p.suffix.lower() == '.raw':
        tip = process_raw_to_color(tip_p)
    else:
        tip = cv2.imread(str(tip_p), cv2.IMREAD_UNCHANGED)
        
    if bg is None or tip is None: return None

    # TIP 리사이즈
    tip = random_scale(tip)
    
    bh, bw = bg.shape[:2]
    th, tw = tip.shape[:2]
    
    if bw < tw or bh < th: return None 

    # 랜덤 좌표
    x = random.randint(0, bw - tw - 1)
    y = random.randint(0, bh - th - 1)

    # 합성
    overlay_images(bg, tip, x, y)
    
    # 결과 저장
    out_path = out_stem.with_suffix(".png")
    cv2.imwrite(str(out_path), bg)
    
    # 라벨 저장
    cx, cy = (x + tw/2)/bw, (y + th/2)/bh
    nw, nh = tw/bw, th/bh
    return cx, cy, nw, nh

def process_class(cls_name):
    tip_dir = TIP_ROOT / cls_name
    tip_list = []
    if DO_PROCESS_TIP_RAW:
        tip_list.extend(glob(str(tip_dir / "*.raw")))
    tip_list.extend(glob(str(tip_dir / "*.png")))
    
    if not tip_list: return

    bg_list = []
    if DO_PROCESS_BG_RAW:
        bg_list.extend(glob(str(BG_ROOT / "**/*.raw"), recursive=True))
    bg_list.extend(glob(str(BG_ROOT / "**/*.png"), recursive=True))
    
    if not bg_list: return

    out_img_dir = DST_ROOT / "images" / cls_name
    out_lbl_dir = DST_ROOT / "labels" / cls_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    while count < TARGET_PER_CLS:
        tip_p = Path(random.choice(tip_list))
        bg_p = Path(random.choice(bg_list))
        stem = out_img_dir / f"{cls_name}_{count:05d}"
        
        res = process_single_mix(bg_p, tip_p, stem)
        if res:
            cx, cy, w, h = res
            with (out_lbl_dir / f"{stem.name}.txt").open("w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            count += 1
            if count % 1000 == 0:
                print(f"[{cls_name}] {count}/{TARGET_PER_CLS}")

def main():
    if _LUT_Z is None or _LUT_K is None:
        print("CRITICAL ERROR: LUT 파일 없음")
        return

    classes = [d.name for d in TIP_ROOT.iterdir() if d.is_dir()]
    print(f"작업 시작: 클래스 {len(classes)}개")
    
    with Pool(NUM_WORKERS) as pool:
        pool.map(process_class, classes)

if __name__ == "__main__":
    main()