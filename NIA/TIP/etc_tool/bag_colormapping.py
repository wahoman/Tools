#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_batch_colormap_gpu.py
[GPU 가속] Raw 이미지 일괄 컬러 변환 도구
 · 역할: 배경(가방) Raw 폴더를 통째로 읽어 컬러 PNG 폴더로 변환
 · 기술: CuPy(GPU)를 사용하여 Z-eff 물리 연산을 고속 처리
"""

import os
import time
import numpy as np
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm

# CuPy 임포트 (GPU 가속 핵심)
try:
    import cupy as cp
    print(f"✅ GPU 가속 활성화: {cp.cuda.runtime.getDeviceCount()}개 장치 발견")
except ImportError:
    print("❌ 오류: CuPy가 설치되지 않았습니다. (pip install cupy-cuda1xx)")
    exit()

# ── ★ 사용자 설정 ───────────────────────────────────────────────
# 1. 변환할 Raw 파일들이 있는 최상위 폴더
SRC_ROOT = Path(r"D:\hgyeo\BCAS_TIP\APIDS Bare Bags")

# 2. 변환된 PNG가 저장될 폴더 (폴더 구조 그대로 복사됨)
DST_ROOT = Path(r"D:\hgyeo\BCAS_TIP\APIDS Bare Bags_ColorPNG")

# 3. LUT 파일 경로
PATH_LUT_CSV   = "LUT_ZeffU2.csv"
PATH_GRAY_CLUT = "gray.clut"
RAW_WIDTH      = 640
# ────────────────────────────────────────────────────────────────

class GpuColorMapper:
    def __init__(self):
        self._LUT_Z_GPU = None
        self._LUT_K_GPU = None
        self._ZEFF_TABLE_GPU = None
        # 보정 상수 (GPU 메모리로 업로드)
        self._ZI0_GPU = cp.array([98000]*9, dtype=cp.float32)

    def load_luts(self):
        """LUT 데이터를 CPU에서 읽어 GPU 메모리로 전송"""
        print(">>> LUT 데이터를 GPU 메모리로 로드 중...")
        
        # 1. Zeff LUT (CSV)
        pseudoZ = np.zeros((300, 256, 3), dtype=np.uint8)
        try:
            with open(PATH_LUT_CSV, "rb") as f:
                lines = [r.strip() for r in f.read().decode("utf-16").splitlines() if r.strip()]
            for j, line in enumerate(lines[:300]):
                parts = line.split(",")
                if len(parts) >= 256*3:
                    arr = np.array(parts[:256*3], dtype=np.int16).reshape(256, 3)
                    pseudoZ[j] = np.clip(arr, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"❌ LUT 로드 실패: {e}")
            return False

        # 2. Gray CLUT
        try:
            with open(PATH_GRAY_CLUT, "rb") as f:
                buf = f.read()[8:]
            lut_k = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4)[:256, :3].copy()
        except Exception:
            lut_k = np.zeros((256, 3), dtype=np.uint8)

        # 3. Zeff Table Calculation (CPU에서 계산 후 GPU로 전송)
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

        # ★ CPU -> GPU 메모리 전송 (핵심)
        self._LUT_Z_GPU = cp.asarray(pseudoZ)
        self._LUT_K_GPU = cp.asarray(lut_k)
        self._ZEFF_TABLE_GPU = cp.asarray(luts)
        print("✅ GPU 메모리 로드 완료.")
        return True

    def process_file(self, raw_path, save_path):
        """단일 파일을 GPU로 처리하여 저장"""
        # 1. 파일 읽기 (CPU IO)
        try:
            arr_cpu = np.fromfile(str(raw_path), dtype=np.uint16)
        except: return False
        
        if arr_cpu.size == 0 or arr_cpu.size % RAW_WIDTH != 0: return False
        
        # 2. 데이터 GPU로 업로드 (Host -> Device)
        img_gpu = cp.asarray(arr_cpu).reshape(-1, RAW_WIDTH)
        rows = img_gpu.shape[0]
        if rows < 2 or (rows % 2) != 0: return False

        # 3. 배경 레벨링 (GPU 연산)
        bin_center = 50000
        # bincount는 GPU에서 효율이 안나올 수 있어 histogram 사용 혹은 단순화
        # 여기서는 CPU 로직을 GPU로 그대로 번역
        idx = cp.floor((img_gpu.astype(cp.int32) - bin_center) / 16 + 0.5 + 128).astype(cp.int32)
        cp.clip(idx, 0, 255, out=idx)
        
        # 히스토그램 계산 (GPU)
        hist = cp.bincount(idx.ravel(), minlength=256)
        peak = int(cp.argmax(hist)) # 결과값 하나만 CPU로 가져옴
        bkg_lvl = float(bin_center + 16 * (peak + 0.5 - 128))

        g = img_gpu.astype(cp.float32)
        scale = 65535.0 / max(bkg_lvl, 1.0)
        g *= scale
        g = cp.clip(g, 0, 65535).astype(cp.uint16)

        # 4. High/Low 분리
        vd = rows // 2
        top, bottom = g[:vd, :], g[vd:, :]
        
        # 평균 계산 (GPU reduce)
        m_top = float(top.mean())
        m_bot = float(bottom.mean())
        
        if m_top >= m_bot:
            imgH, imgL = top, bottom
        else:
            imgH, imgL = bottom, top

        # 5. Z-eff 계산 (전체 픽셀 병렬 처리)
        det_idx = 0
        Ilow = cp.maximum(imgL.astype(cp.float32), 1.0)
        Ihigh = cp.maximum(imgH.astype(cp.float32), 1.0)
        zI0 = self._ZI0_GPU[det_idx]

        num = cp.log(zI0 / Ilow)
        den = cp.log(zI0 / Ihigh)
        # 0으로 나누기 방지
        den = cp.where(den <= 1e-12, 1e-12, den)

        rate = cp.clip(((num/den - 1.0 + 0.0005) * 1000.0).astype(cp.int32), 0, 1999)
        ze_val = self._ZEFF_TABLE_GPU[det_idx][rate]

        # 6. 컬러 매핑 (Advanced Indexing on GPU)
        pbSrc = (imgH >> 8).astype(cp.uint16)
        nzeff = cp.clip(((ze_val.astype(cp.int32) * 10) // 256) - 1, 0, 299)
        mask_invalid = nzeff < 0

        # LUT Lookup (GPU 메모리 내에서 조회)
        color_K = self._LUT_K_GPU[pbSrc]
        color_Z = self._LUT_Z_GPU[nzeff, pbSrc]

        rgb = cp.where(mask_invalid[..., None], color_K, color_Z)
        bgr = cp.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0]], axis=-1).astype(cp.uint8)

        # 7. 결과 다운로드 (Device -> Host) 및 저장
        bgr_cpu = cp.asnumpy(bgr)
        
        # 저장 경로 폴더 생성
        save_dir = save_path.parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            
        cv2.imwrite(str(save_path), bgr_cpu)
        return True

def main():
    if not SRC_ROOT.exists():
        print(f"❌ 원본 경로가 없습니다: {SRC_ROOT}")
        return

    # 1. 파일 목록 스캔
    print(">>> Raw 파일 스캔 중...")
    raw_files = sorted(list(SRC_ROOT.glob("**/*.raw")))
    print(f"총 {len(raw_files)}개의 Raw 파일을 발견했습니다.")

    if not raw_files: return

    # 2. 매퍼 초기화
    mapper = GpuColorMapper()
    if not mapper.load_luts():
        return

    # 3. 변환 루프 (tqdm으로 진행률 표시)
    print(">>> GPU 가속 변환 시작...")
    start_time = time.time()
    
    success_cnt = 0
    for raw_p in tqdm(raw_files):
        # 상대 경로 계산 (폴더 구조 유지)
        rel_path = raw_p.relative_to(SRC_ROOT)
        save_p = DST_ROOT / rel_path.with_suffix(".png")
        
        # 이미 변환된 파일이 있으면 스킵하려면 아래 주석 해제
        # if save_p.exists(): continue

        if mapper.process_file(raw_p, save_p):
            success_cnt += 1
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✅ 완료! {success_cnt}개 파일 변환됨.")
    print(f"⏱️ 소요 시간: {duration:.1f}초 (평균 {duration/len(raw_files):.3f}초/장)")
    print(f"📂 저장 위치: {DST_ROOT}")

if __name__ == "__main__":
    main()