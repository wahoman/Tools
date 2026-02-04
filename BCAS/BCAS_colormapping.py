import os
import cv2
import numpy as np
import cupy as cp
from glob import glob
import time
from concurrent.futures import ThreadPoolExecutor

# =========================================================
# 1. 경로 설정
# =========================================================
_szRootDir = r"\\SSTL_NAS\BCAS_Datacollection\1_Datacollection(at.ThreeD)\Day2"
_szPalettePath = r"D:\hgyeo\NIA_TIP\LUT_ZeffU2.csv"
_szGrayClutPath = r"D:\hgyeo\NIA_TIP\Gray.clut"

# =========================================================
# 2. 전역 변수
# =========================================================
_zeffTbl_gpu = None
_pseudoColorZ_gpu = None
_pseudoColorK_gpu = None
_zI0 = [ 98000.0 ] * 9

_Z = [
    [   1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,
        1.206871776,1.215205132,1.226352526,1.240297472,1.257289302,
        1.27849552,1.304551537,1.334194598,1.367066563,1.402681085,
        1.440472022,1.479718307,1.519644992,1.559941821,1.599809092,
        1.638299752,1.675213149,1.709797614,1.742333096,1.772344093,
        1.799326033,1.823810917,1.846055878,1.864917778,1.881765843
    ]
] * 9

def load_tables_to_gpu():
    global _zeffTbl_gpu, _pseudoColorZ_gpu, _pseudoColorK_gpu
    
    print("⏳ 테이블 데이터 로딩 중...")

    # 1. Zeff Table 생성
    zeffTbl_cpu = np.zeros((9, 2000), dtype=np.uint16)
    for d in range(9):
        nZ = [0] * 31
        noffset = 1000
        for i in range(30):
            fv = float(_Z[d][i])
            nZ[i] = max(int((fv + 0.0005) * 1000) - noffset, 0)
        idx = 0
        for k in range(nZ[0]): zeffTbl_cpu[d][idx] = 1 << 8; idx += 1
        for j in range(30):
             if idx >= 2000: break
             nW = nZ[j + 1] - nZ[j]
             for k in range(nW):
                 if idx >= 2000: break
                 zeffTbl_cpu[d][idx] = int(((j + 1) << 8) + (k << 8) / nW); idx += 1
        while idx < 2000: zeffTbl_cpu[d][idx] = 30 << 8; idx += 1
    _zeffTbl_gpu = cp.asarray(zeffTbl_cpu, dtype=cp.uint16)

    # 2. Color Table 로드 (LUT_ZeffU2.csv)
    pcZ_cpu = np.zeros((300, 256, 3), dtype=np.uint8)
    try:
        with open(_szPalettePath, 'rb') as file:
            pBuf = file.read().decode('utf-16') # utf-16 디코딩 에러 가능성 있음
            tokens = pBuf.split('\n')
            count = 0
            for j in range(min(300, len(tokens))):
                tokens[j] = tokens[j].replace('\r', '')
                vecRGB = tokens[j].split(',')
                if len(vecRGB) < 256 * 3: continue
                t = 0
                for k in range(256):
                    pcZ_cpu[j, k, 0] = int(vecRGB[t])     # R
                    pcZ_cpu[j, k, 1] = int(vecRGB[t + 1]) # G
                    pcZ_cpu[j, k, 2] = int(vecRGB[t + 2]) # B
                    t += 3
                count += 1
            if count == 0:
                print("⚠️ 경고: 컬러 테이블(LUT) 데이터가 비어있습니다. (형식 문제?)")
    except Exception as e:
        print(f"❌ 에러: 컬러 테이블 로드 실패 ({e})")
        
    _pseudoColorZ_gpu = cp.asarray(pcZ_cpu, dtype=cp.uint8)

    # 3. Gray Table 로드 (Gray.clut)
    pcK_cpu = np.zeros((256, 3), dtype=np.uint8)
    try:
        with open(_szGrayClutPath, 'rb') as file:
            file.seek(8)
            ptBuf = file.read()
            if len(ptBuf) < 256 * 4:
                print("⚠️ 경고: Gray 테이블 파일 크기가 너무 작습니다.")
            else:
                for k in range(256):
                    pcK_cpu[k, 0] = ptBuf[k * 4]     # R
                    pcK_cpu[k, 1] = ptBuf[k * 4 + 1] # G
                    pcK_cpu[k, 2] = ptBuf[k * 4 + 2] # B
                    # Alpha(3)은 무시하고 RGB만 사용
    except Exception as e:
         print(f"❌ 에러: Gray 테이블 로드 실패 ({e})")
         
    _pseudoColorK_gpu = cp.asarray(pcK_cpu, dtype=cp.uint8)
    print("✅ GPU 테이블 로드 완료.")

# =========================================================
# 3. 계산 함수
# =========================================================
def bkg_level_cpu(arr, bin_center=50000, bin_gap=16):
    bins = [bin_center + bin_gap*(n-128) for n in range(256)]
    hist, _ = np.histogram(arr, bins)
    peak_pos = bin_center + bin_gap*((np.argmax(hist)+0.5-128))
    return float(peak_pos)

def process_image_full_gpu(imgLoE, imgHiE, d=0):
    imgLoE_cp = cp.asarray(imgLoE, dtype=cp.float32)
    imgHiE_cp = cp.asarray(imgHiE, dtype=cp.float32)
    imgLoE_cp = cp.maximum(imgLoE_cp, 1.0)
    imgHiE_cp = cp.maximum(imgHiE_cp, 1.0)

    zI0_val = _zI0[d]
    numerator = cp.log(zI0_val / imgLoE_cp)
    denominator = cp.log(zI0_val / imgHiE_cp)
    denominator = cp.where(denominator == 0, 1e-6, denominator)

    ratio_cp = numerator / denominator - 1
    rate_cp = cp.clip((ratio_cp * 1000).astype(cp.int32), 0, 1999)
    pImgZeff_cp = _zeffTbl_gpu[d][rate_cp]

    pwSrc_cp = (cp.asarray(imgHiE, dtype=cp.uint16) >> 8).astype(cp.uint8)
    pbZeff_cp = cp.clip(((pImgZeff_cp.astype(cp.float32) * 10) / 256).astype(cp.int32) - 1, -1, 299)

    height, width = imgLoE.shape
    final_rgb_cp = cp.zeros((height, width, 3), dtype=cp.uint8)
    mask_dark = pbZeff_cp < 0
    mask_valid = ~mask_dark

    if cp.any(mask_dark):
        final_rgb_cp[mask_dark] = _pseudoColorK_gpu[pwSrc_cp[mask_dark]]
    if cp.any(mask_valid):
        final_rgb_cp[mask_valid] = _pseudoColorZ_gpu[pbZeff_cp[mask_valid], pwSrc_cp[mask_valid]]

    # RGB -> BGR 변환하여 CPU로 반환
    return cp.asnumpy(final_rgb_cp[..., ::-1])

def save_image_task(img_data, path):
    cv2.imwrite(path, img_data, [cv2.IMWRITE_PNG_COMPRESSION, 1])

def process_single_folder(src_path, dst_path, executor):
    os.makedirs(dst_path, exist_ok=True)
    raw_files = sorted(glob(os.path.join(src_path, "*.raw")))
    if not raw_files: return 0

    print(f"   📂 처리: {os.path.basename(src_path)} -> {len(raw_files)}장")
    
    for raw_file in raw_files:
        filename = os.path.basename(raw_file)
        save_path = os.path.join(dst_path, filename.replace('.raw', '.png'))
        
        try:
            raw_data = np.fromfile(raw_file, dtype='uint16')
            total_pixels = raw_data.size
            
            width = 0
            if total_pixels % 768 == 0: width = 768
            elif total_pixels % 1280 == 0: width = 1280
            elif total_pixels % 1024 == 0: width = 1024
            elif total_pixels % 896 == 0: width = 896
            elif total_pixels % 640 == 0: width = 640
            
            if width == 0: continue

            img = raw_data.reshape((-1, width))
            bkg_val = bkg_level_cpu(img)
            if bkg_val < 10000: bkg_val = 65535 
            img = np.clip(img.astype(np.float32) * (65535.0 / bkg_val), 0, 65535).astype(np.uint16)

            height = img.shape[0]
            mid = height // 2
            le_img = img[:mid, :]
            he_img = img[mid:, :]

            bgr_image = process_image_full_gpu(le_img, he_img)
            executor.submit(save_image_task, bgr_image, save_path)

        except Exception as e:
            print(f"❌ Error({filename}): {e}")
            
    return len(raw_files)

# =========================================================
# 4. 메인 실행
# =========================================================
def main():
    load_tables_to_gpu()
    
    # GPU 테이블이 제대로 로드됐는지 확인
    if cp.sum(_pseudoColorZ_gpu) == 0:
        print("\n⛔ [치명적 오류] 컬러 테이블(LUT)이 모두 0입니다! 파일 경로를 확인하세요.")
        print(f"   경로: {_szPalettePath}")
        return

    all_subdirs = [d for d in glob(os.path.join(_szRootDir, "*")) if os.path.isdir(d)]
    all_subdirs.sort()
    
    print(f"🚀 총 {len(all_subdirs)}개 폴더 변환 시작!")
    
    max_workers = os.cpu_count()
    start_time = time.time()
    total_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for src_dir in all_subdirs:
            folder_name = os.path.basename(src_dir)
            if folder_name.endswith("_png"): continue
            
            dst_dir = src_dir + "_png"
            total_count += process_single_folder(src_dir, dst_dir, executor)
            
    print(f"\n🎉 완료! 총 {total_count}장 / {time.time()-start_time:.2f}초")

if __name__ == "__main__":
    main()