# -*- coding: utf-8 -*-
import os
import sqlite3
import re
import cv2
import numpy as np

# ======================================================================
# ====== CuPy (GPU) Import ======
# ======================================================================
try:
    import cupy as cp
except ImportError as e:
    raise RuntimeError(
        "CuPy가 필요합니다. CUDA 버전에 맞게 설치하세요 (예: pip install cupy-cuda12x)."
    ) from e

# ======================================================================
# ====== 경로 설정 (사용자 환경에 맞게 이 부분을 수정하세요) ======
# ======================================================================
# [입력 1] 변환의 기준이 될 PNG 파일들이 있는 최상위 폴더
INPUT_PNG_ROOT = r"D:\hgyeo\test"

# [입력 2] 원본 Raw 파일의 전체 경로 정보가 담긴 데이터베이스 파일
DATABASE_FILE = r"D:\hgyeo\raw_files.db"

# [최종 출력] 컬러매핑을 마친 PNG 파일이 최종 저장될 폴더
FINAL_OUTPUT_PATH = r"D:\hgyeo\test_output2"

# [처리 설정] 이미지 너비
WIDTH = 896
# ======================================================================


# ======================================================================
# ====== Helper Functions ======
# ======================================================================

def get_raw_filepath_from_db(db_conn, raw_filename):
    """DB에서 raw 파일명을 이용해 전체 파일 경로를 조회"""
    cursor = db_conn.cursor()
    cursor.execute("SELECT filepath FROM files WHERE filename = ?", (raw_filename,))
    result = cursor.fetchone()
    return result[0] if result else None

def build_zeff_luts_strict_np(_Z_np: np.ndarray) -> np.ndarray:
    luts = np.zeros((9, 2000), dtype=np.uint16)
    noffset = 1000
    for d in range(9):
        nZ = np.zeros(31, dtype=np.int32)
        vals = np.maximum(((_Z_np[d] + 0.0005) * 1000).astype(np.int32) - noffset, 0)
        nZ[:30] = vals
        nZ[30] = 2000
        i = 0
        for _ in range(int(nZ[0])):
            if i >= 2000: break
            luts[d, i] = (1 << 8)
            i += 1
        for j in range(30):
            if i >= 2000: break
            nW = int(nZ[j+1] - nZ[j])
            if nW <= 0: continue
            k = np.arange(nW, dtype=np.int32)
            seg = ((j+1) << 8) + ((k << 8) // nW)
            end = min(i + nW, 2000)
            luts[d, i:end] = seg[:(end - i)].astype(np.uint16)
            i = end
        while i < 2000:
            luts[d, i] = (30 << 8)
            i += 1
    return luts

def load_palettes():
    pseudoZ = None
    pseudoK = np.zeros((256, 3), dtype=np.uint8)
    try:
        with open("LUT_ZeffU.csv", "rb") as f:
            raw = f.read().decode("utf-16")
        rows = [r.strip() for r in raw.splitlines() if r.strip()]
        if len(rows) >= 300:
            pseudoZ = np.zeros((300, 256, 3), dtype=np.uint8)
            for j in range(300):
                parts = rows[j].split(",")
                if len(parts) < 256*3: continue
                arr = np.array(parts[:256*3], dtype=np.int16).reshape(256, 3)
                pseudoZ[j] = np.clip(arr, 0, 255).astype(np.uint8)
    except FileNotFoundError:
        print("WARN: LUT_ZeffU.csv not found -> pseudoZ=None (회색 팔레트만 사용)")
    try:
        with open("gray.clut", "rb") as f:
            buf = f.read()
        pt = buf[8:]
        g = np.frombuffer(pt, dtype=np.uint8)
        pseudoK = g.reshape(-1, 4)[:256, :3].copy()
    except FileNotFoundError:
        print("WARN: gray.clut not found -> pseudoK zeros")
    return pseudoZ, pseudoK

def bkg_level_cpu_fast(arr: np.ndarray, bin_center=50000, bin_gap=16):
    if arr is None or arr.size == 0: return float(bin_center)
    idx = np.floor((arr.astype(np.int32) - bin_center) / bin_gap + 0.5 + 128).astype(np.int32)
    np.clip(idx, 0, 255, out=idx)
    hist = np.bincount(idx.ravel(), minlength=256)
    if hist.sum() == 0: return float(bin_center)
    peak = int(np.argmax(hist))
    peak_pos = float(bin_center + bin_gap * (peak + 0.5 - 128))
    return peak_pos if np.isfinite(peak_pos) else float(bin_center)

def zeff_exact_gpu_u16(imgLoE_gpu_u16: cp.ndarray, imgHiE_gpu_u16: cp.ndarray, d: int) -> cp.ndarray:
    if imgLoE_gpu_u16.size == 0 or imgHiE_gpu_u16.size == 0: return cp.empty((0, 0), dtype=cp.uint16)
    Ilow  = cp.maximum(imgLoE_gpu_u16.astype(cp.float32), 1.0)
    Ihigh = cp.maximum(imgHiE_gpu_u16.astype(cp.float32), 1.0)
    zI0   = _zI0_GPU[d]
    num   = cp.log(zI0 / Ilow)
    den   = cp.log(zI0 / Ihigh)
    den   = cp.where(den <= 1e-12, 1e-12, den)
    rate = ((num/den - 1.0 + 0.0005) * 1000.0).astype(cp.int32)
    rate = cp.clip(rate, 0, 1999)
    lut_d = _ZEFF_LUTS_GPU[d]
    return lut_d[rate].astype(cp.uint16)

def save_pseudo_bgr_from_gpu(imgH_gpu_u16: cp.ndarray, ze_gpu_u16: cp.ndarray, output_filepath: str) -> bool:
    """한글 경로 지원을 위해 cv2.imencode를 사용하여 안정적으로 저장"""
    H, W = imgH_gpu_u16.shape
    if H == 0 or W == 0 or ze_gpu_u16.size == 0:
        print(f"  [SKIP] empty image for {os.path.basename(output_filepath)}")
        return False
    
    pbSrc = (imgH_gpu_u16 >> 8).astype(cp.uint16)
    nzeff = ((ze_gpu_u16.astype(cp.uint32) * 10) // 256).astype(cp.int32) - 1
    mask_gray = nzeff < 0
    
    if _PSEUDO_Z_GPU is None:
        rgb = _PSEUDO_K_GPU[pbSrc]
        bgr = cp.stack([rgb[...,2], rgb[...,1], rgb[...,0]], axis=2)
    else:
        nzeff = cp.clip(nzeff, 0, _PSEUDO_Z_GPU.shape[0]-1)
        gray_rgb   = _PSEUDO_K_GPU[pbSrc]
        pseudo_rgb = _PSEUDO_Z_GPU[nzeff, pbSrc]
        rgb = cp.where(mask_gray[..., None], gray_rgb, pseudo_rgb)
        bgr = cp.stack([rgb[...,2], rgb[...,1], rgb[...,0]], axis=2)
    
    outBGR = cp.asnumpy(bgr)
    
    try:
        extension = os.path.splitext(output_filepath)[1]
        result, buffer = cv2.imencode(extension, outBGR)
        if result:
            with open(output_filepath, 'wb') as f:
                f.write(buffer)
            return True
        else:
            print(f"  [DEBUG] cv2.imencode failed for {os.path.basename(output_filepath)}")
            return False
    except Exception as e:
        print(f"  [DEBUG] File write error for {os.path.basename(output_filepath)}: {e}")
        return False

# ======================================================================
# ====== 전역 변수 및 GPU 메모리 초기화 ======
# ======================================================================
_zI0 = np.array([98000]*9, dtype=np.float32)
_Z = np.array([[1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,1.206871776,1.215205132,1.226352526,1.240297472,1.257289302,1.27849552,1.304551537,1.334194598,1.367066563,1.402681085,1.440472022,1.479718307,1.519644992,1.559941821,1.599809092,1.638299752,1.675213149,1.709797614,1.742333096,1.772344093,1.799326033,1.823810917,1.846055878,1.864917778,1.881765843]]*9, dtype=np.float32)
_ZEFF_LUTS_CPU = build_zeff_luts_strict_np(_Z)
_PSEUDO_Z_CPU, _PSEUDO_K_CPU = load_palettes()

print("GPU 메모리로 LUT 및 팔레트 데이터를 전송합니다...")
_ZEFF_LUTS_GPU = cp.asarray(_ZEFF_LUTS_CPU)
_zI0_GPU = cp.asarray(_zI0, dtype=cp.float32)
_PSEUDO_Z_GPU = None if _PSEUDO_Z_CPU is None else cp.asarray(_PSEUDO_Z_CPU)
_PSEUDO_K_GPU = cp.asarray(_PSEUDO_K_CPU)
print("초기화 완료.")

# ======================================================================
# ====== 메인 실행 로직 ======
# ======================================================================
# ======================================================================
# ====== 메인 실행 로직 (수정됨) ======
# ======================================================================
def main():
    """메인 처리 함수"""
    print("\n" + "="*50)
    print("====== Raw 파일 변환 및 컬러매핑을 시작합니다. ======")
    print("="*50)
    
    filename_pattern = re.compile(r'_\d{8}_(\d{8})_.*?_(\d+)\.png$', re.IGNORECASE)
    
    print("처리 대상 PNG 파일 목록을 생성 중입니다...")
    png_files_to_process = []
    for dirpath, _, filenames in os.walk(INPUT_PNG_ROOT):
        for png_filename in filenames:
            if png_filename.lower().endswith('.png') and filename_pattern.search(png_filename):
                full_path = os.path.join(dirpath, png_filename)
                png_files_to_process.append(full_path)
    
    total_files = len(png_files_to_process)
    if total_files == 0:
        print("처리할 PNG 파일을 찾지 못했습니다. 파일명 패턴이나 경로를 확인해주세요.")
        return
        
    print(f"총 {total_files}개의 PNG 파일을 기준으로 변환을 시도합니다.")
    
    db_conn = None
    try:
        if not os.path.exists(DATABASE_FILE):
            print(f"오류: 데이터베이스 파일 '{DATABASE_FILE}'을 찾을 수 없습니다.")
            return
        db_conn = sqlite3.connect(DATABASE_FILE)
        
        processed_count = 0
        next_progress_milestone = 10 # 첫 진행률 알림은 10%

        for png_full_path in png_files_to_process:
            processed_count += 1
            
            # --- 진행률 표시 로직 ---
            current_percent = (processed_count / total_files) * 100
            if current_percent >= next_progress_milestone:
                print(f"\n--- 진행 상황: {next_progress_milestone}% 완료 ({processed_count}/{total_files}) ---\n")
                next_progress_milestone += 10 # 다음 알림은 10% 뒤
            
            # --------------------------

            dirpath = os.path.dirname(png_full_path)
            png_filename = os.path.basename(png_full_path)

            match = filename_pattern.search(png_filename)
            sLID, d_str = match.groups()
            d_index = int(d_str)

            if d_index == 0:
                continue
            
            detector_index = d_index - 1
            
            target_raw_filename = f"{sLID}_{detector_index}.raw"
            source_raw_path = get_raw_filepath_from_db(db_conn, target_raw_filename)

            if not source_raw_path or not os.path.exists(source_raw_path):
                # 이 부분은 오류이므로 계속 표시해 주는 것이 좋습니다.
                print(f"  [오류] ({processed_count}/{total_files}) {png_filename} -> 원본 Raw({target_raw_filename}) 파일을 찾을 수 없습니다.")
                continue
            
            arr = np.fromfile(source_raw_path, dtype=np.uint16)
            if arr.size == 0 or arr.size % WIDTH != 0:
                continue
            
            img = arr.reshape(-int(arr.size / WIDTH), WIDTH)
            rows = img.shape[0]
            if rows < 2 or (rows % 2) != 0:
                continue

            bkg_lvl = bkg_level_cpu_fast(img)
            g = cp.asarray(img, dtype=cp.float32)
            scale = 65535.0 / max(bkg_lvl, 1.0)
            g *= scale
            g = cp.clip(g, 0, 65535).astype(cp.uint16)

            vd = rows // 2
            top, bottom = g[:vd, :], g[vd:, :]
            m_top, m_bot = float(top.mean().get()), float(bottom.mean().get())
            imgH_gpu = top if m_top >= m_bot else bottom
            imgL_gpu = bottom if m_top >= m_bot else top
            
            ze_gpu = zeff_exact_gpu_u16(imgL_gpu, imgH_gpu, detector_index)

            relative_dir = os.path.relpath(dirpath, INPUT_PNG_ROOT)
            output_dir_for_pngs = os.path.join(FINAL_OUTPUT_PATH, relative_dir)
            os.makedirs(output_dir_for_pngs, exist_ok=True)
            
            png_base_name = os.path.splitext(png_filename)[0]
            output_png_filename = f"{png_base_name}_PH.png"
            output_png_path = os.path.join(output_dir_for_pngs, output_png_filename)
            
            save_pseudo_bgr_from_gpu(imgH_gpu, ze_gpu, output_png_path)

    except sqlite3.Error as e:
        print(f"데이터베이스 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
    finally:
        if db_conn:
            db_conn.close()
    
    print("\n" + "="*50)
    print(f"모든 작업이 완료되었습니다. (총 {total_files}개 처리)")
    print("="*50)
    
if __name__ == "__main__":
    main()