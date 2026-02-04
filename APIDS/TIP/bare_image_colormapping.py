import os
import cv2
import numpy as np
import cupy as cp
from glob import glob

# =========================================================
# 1. 경로 설정
# =========================================================
_szSrcPath = r"D:\hgyeo\TIP\Origin_bare\Round1_Bare"
_szDstPath = r"D:\hgyeo\TIP\Origin_bare\Round1_Bare_Color"

# 테이블 파일 경로 (파일 있는지 확인!)
_szPalettePath = r"D:\hgyeo\LUT_ZeffU.csv"
_szGrayClutPath = r"D:\hgyeo\Gray.clut"

# GPU 확인
try:
    cp.cuda.Device(0).compute_capability
    print("✅ GPU Acceleration Enabled (CuPy)")
except:
    print("⚠️ GPU not detected. Using CPU only (might be slow).")

# =========================================================
# 2. 전역 변수 및 테이블
# =========================================================
_zeffTbl = [ [ 0 ] * 2000 ] * 9
_pseudoColorZ = None
_pseudoColorK = None

_zI0 = [ 98000 ] * 9
_Z = [
    [   1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,
        1.206871776,1.215205132,1.226352526,1.240297472,1.257289302,
        1.27849552,1.304551537,1.334194598,1.367066563,1.402681085,
        1.440472022,1.479718307,1.519644992,1.559941821,1.599809092,
        1.638299752,1.675213149,1.709797614,1.742333096,1.772344093,
        1.799326033,1.823810917,1.846055878,1.864917778,1.881765843
    ]
] * 9

def loadzefftable():
    for d in range(9):
        nZ = [0] * 31
        noffset = 1000
        for i in range(30):
            fv = float(_Z[d][i])
            nZ[i] = max(int((fv + 0.0005) * 1000) - noffset, 0)
        
        i = 0
        for k in range(nZ[0]):
            _zeffTbl[d][i] = 1 << 8
            i += 1
        for j in range(30):
             if i >= 2000: break
             nW = nZ[j + 1] - nZ[j]
             for k in range(nW):
                 if i >= 2000: break
                 _zeffTbl[d][i] = int(((j + 1) << 8) + (k << 8) / nW)
                 i += 1
        while i < 2000:
            _zeffTbl[d][i] = 30 << 8
            i += 1

def loadcolortable():
    global _pseudoColorZ, _pseudoColorK
    try:
        with open(_szPalettePath, 'rb') as file:
            lFileSize = file.seek(0, 2)
            file.seek(0)
            pBuf = file.read(lFileSize).decode('utf-16')
            tokens = pBuf.split('\n')
            _pseudoColorZ = np.zeros((300, 256), dtype=[('rgbRed', 'u1'), ('rgbGreen', 'u1'), ('rgbBlue', 'u1'), ('rgbReserved', 'u1')])
            for j in range(min(300, len(tokens))):
                tokens[j] = tokens[j].replace('\r', '')
                vecRGB = tokens[j].split(',')
                if len(vecRGB) < 256 * 3: continue
                t = 0
                for k in range(256):
                    _pseudoColorZ[j][k]['rgbRed'] = int(vecRGB[t])
                    _pseudoColorZ[j][k]['rgbGreen'] = int(vecRGB[t + 1])
                    _pseudoColorZ[j][k]['rgbBlue'] = int(vecRGB[t + 2])
                    _pseudoColorZ[j][k]['rgbReserved'] = 255
                    t += 3
    except Exception as e:
        print(f"Warning: Failed to load Zeff table ({e})")

    _pseudoColorK = np.zeros(256, dtype=[('rgbRed', 'u1'), ('rgbGreen', 'u1'), ('rgbBlue', 'u1'), ('rgbReserved', 'u1')])
    try:
        with open(_szGrayClutPath, 'rb') as file:
            file.seek(8)
            ptBuf = file.read()
            for k in range(256):
                _pseudoColorK[k]['rgbRed'] = ptBuf[k * 4]
                _pseudoColorK[k]['rgbGreen'] = ptBuf[k * 4 + 1]
                _pseudoColorK[k]['rgbBlue'] = ptBuf[k * 4 + 2]
                _pseudoColorK[k]['rgbReserved'] = ptBuf[k * 4 + 3]
    except Exception as e:
         print(f"Warning: Failed to load Gray table ({e})")

# =========================================================
# 3. GPU 처리 함수 (핵심)
# =========================================================

def bkg_level_cpu(arr, bin_center=50000, bin_gap=16):
    # 배경 레벨은 CPU(Numpy)가 빠르고 간편합니다
    bins = [bin_center + bin_gap*(n-128) for n in range(256)]
    hist, _ = np.histogram(arr, bins)
    peak_pos = bin_center + bin_gap*((np.argmax(hist)+0.5-128))
    return float(peak_pos)

def make_zeff_image_gpu(imgLoE, imgHiE, d=0):
    # GPU로 데이터 이동
    imgLoE_cp = cp.asarray(imgLoE, dtype=cp.float32)
    imgHiE_cp = cp.asarray(imgHiE, dtype=cp.float32)

    # 0 나누기 방지
    imgLoE_cp = cp.maximum(imgLoE_cp, 1.0)
    imgHiE_cp = cp.maximum(imgHiE_cp, 1.0)

    # 로그 계산 (GPU 병렬 처리)
    zI0_val = float(_zI0[d])
    numerator = cp.log(zI0_val / imgLoE_cp)
    denominator = cp.log(zI0_val / imgHiE_cp)
    
    # 분모 0 방지
    denominator = cp.where(denominator == 0, 1e-6, denominator)

    ratio_cp = numerator / denominator - 1
    rate_cp = cp.clip((ratio_cp * 1000).astype(cp.int32), 0, 1999)

    # LUT 적용 (Fancy Indexing on GPU)
    zeff_tbl_cp = cp.asarray(_zeffTbl[d], dtype=cp.uint16)
    pImgZeff_cp = zeff_tbl_cp[rate_cp]

    # CPU로 결과 반환
    return cp.asnumpy(pImgZeff_cp)

def make_pseudo_image_save(pImg, pImgzs, save_path):
    # 컬러 매핑 및 저장은 CPU(OpenCV)가 효율적입니다
    nHeight, nWidth = pImg.shape
    pImgDst = np.zeros((nHeight, nWidth, 4), dtype=np.uint8)

    global _pseudoColorK
    global _pseudoColorZ

    pwSrc = (pImg >> 8).astype(np.uint8)
    # Zeff 스케일링
    pbZeff = ((pImgzs.astype(np.float32) * 10) / 256).astype(np.int32) - 1
    pbZeff = np.clip(pbZeff, -1, 299)

    mask_dark = pbZeff < 0
    mask_valid = ~mask_dark

    # Vectorized indexing (빠름)
    if np.any(mask_dark):
        indices = pwSrc[mask_dark]
        pImgDst[mask_dark, 0] = _pseudoColorK[indices]['rgbRed']
        pImgDst[mask_dark, 1] = _pseudoColorK[indices]['rgbGreen']
        pImgDst[mask_dark, 2] = _pseudoColorK[indices]['rgbBlue']

    if np.any(mask_valid):
        z_indices = pbZeff[mask_valid]
        src_indices = pwSrc[mask_valid]
        pImgDst[mask_valid, 0] = _pseudoColorZ[z_indices, src_indices]['rgbRed']
        pImgDst[mask_valid, 1] = _pseudoColorZ[z_indices, src_indices]['rgbGreen']
        pImgDst[mask_valid, 2] = _pseudoColorZ[z_indices, src_indices]['rgbBlue']

    pImgDst[..., 3] = 255
    pImgDst2 = cv2.cvtColor(pImgDst, cv2.COLOR_RGBA2BGR)
    
    cv2.imwrite(save_path, pImgDst2)

# =========================================================
# 4. 메인 실행
# =========================================================
def main():
    loadcolortable()
    loadzefftable()
    
    os.makedirs(_szDstPath, exist_ok=True)
    
    raw_files = sorted(glob(os.path.join(_szSrcPath, "*.raw")))
    
    if not raw_files:
        print(f"❌ '{_szSrcPath}' 폴더에 .raw 파일이 없습니다.")
        return

    print(f"🚀 총 {len(raw_files)}개의 파일 변환 시작 (GPU Mode)...")

    for i, raw_path in enumerate(raw_files):
        filename = os.path.basename(raw_path)
        save_name = filename.replace('.raw', '.png')
        save_path = os.path.join(_szDstPath, save_name)

        try:
            # 1. 파일 읽기
            raw_data = np.fromfile(raw_path, dtype='uint16')
            
            # Width 감지
            total_pixels = raw_data.size
            width = 640
            if total_pixels % 640 == 0: width = 640
            elif total_pixels % 896 == 0: width = 896
            elif total_pixels % 1280 == 0: width = 1280
            
            img = raw_data.reshape((-1, width))
            
            # 2. 배경 정규화 (Normalization) - ★누렇게 뜨는 것 방지★
            bkg_val = bkg_level_cpu(img)
            if bkg_val < 10000: bkg_val = 65535 
            
            # CPU에서 정규화 계산 (단순 곱셈은 CPU도 빠름)
            img = np.clip(img.astype(np.float32) * (65535.0 / bkg_val), 0, 65535).astype(np.uint16)

            # 3. HE/LE 분리
            vd = img.shape[0] // 2
            
            # ★★★ [설정] LE = 위쪽 (Top), HE = 아래쪽 (Bottom) ★★★
            # (만약 색 반전되면 서로 바꾸세요)
            le_img = img[:vd, :]
            he_img = img[vd:, :]

            # 4. Zeff 계산 (GPU 가속)
            # GPU로 넘겨서 무거운 로그 연산 처리
            zeff_img = make_zeff_image_gpu(le_img, he_img, d=0)
            
            # 5. 컬러 입히기 및 저장 (CPU)
            make_pseudo_image_save(he_img, zeff_img, save_path)
            
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{len(raw_files)}] 처리 완료...")

        except Exception as e:
            print(f"❌ 에러({filename}): {e}")

    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()