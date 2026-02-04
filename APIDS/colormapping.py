import os
import cv2
import numpy as np
from glob import glob

# =========================================================
# 1. 경로 설정
# =========================================================
# 변환할 Raw 파일이 있는 폴더
_szSrcPath = r"C:\Users\hgy84\Desktop\colormapping\test_image\APIDS"

# 결과 저장할 폴더 (자동 생성됨)
_szDstPath = r"C:\Users\hgy84\Desktop\colormapping"

# 테이블 파일 경로 (파일이 있는지 꼭 확인하세요!)
_szPalettePath = r"C:\Users\hgy84\Desktop\APIDS\LUT_ZeffU.csv"
_szGrayClutPath = r"C:\Users\hgy84\Desktop\APIDS\Gray.clut"

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
# 3. 이미지 처리 함수 (Zeff + Pseudo) - CPU 버전
# =========================================================

def bkg_level(arr, bin_center=50000, bin_gap=16):
    bins = [bin_center + bin_gap*(n-128) for n in range(256)]
    hist, _ = np.histogram(arr, bins)
    peak_pos = bin_center + bin_gap*((np.argmax(hist)+0.5-128))
    return float(peak_pos)

def make_zeff_image(imgLoE, imgHiE, d=0):
    pImgZeff = np.zeros((imgLoE.shape[0], imgLoE.shape[1]), dtype=np.uint16)
    
    # 0으로 나누기 방지
    imgLoE = np.maximum(imgLoE, 1)
    imgHiE = np.maximum(imgHiE, 1)

    narea = imgLoE.size
    tpw1 = imgLoE.flatten()
    tpw2 = imgHiE.flatten()
    tpw5 = _zeffTbl[d]
    
    # Python Loop는 느리므로 Numpy Vectorization 적용 (속도 향상)
    Ilow = np.maximum(tpw1, 1)
    Ihigh = np.maximum(tpw2, 1)
    
    # 로그 계산
    val_low = np.log(_zI0[d] / Ilow)
    val_high = np.log(_zI0[d] / Ihigh)
    
    # 0 나누기 방지
    val_high[val_high == 0] = 1e-6
    
    rate = ((val_low / val_high - 1 + 0.0005) * 1000).astype(np.int32)
    rate = np.clip(rate, 0, 1999)
    
    # LUT 적용
    # _zeffTbl[d]는 리스트이므로 numpy array로 변환해서 인덱싱
    lut = np.array(tpw5, dtype=np.uint16)
    pImgZeff_flat = lut[rate]
    
    pImgZeff = pImgZeff_flat.reshape(imgLoE.shape)
    return pImgZeff

def make_pseudo_image_save(pImg, pImgzs, save_path):
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
    
    # 지정된 폴더 내의 모든 .raw 파일 찾기
    raw_files = sorted(glob(os.path.join(_szSrcPath, "*.raw")))
    
    if not raw_files:
        print(f"❌ '{_szSrcPath}' 폴더에 .raw 파일이 없습니다.")
        return

    print(f"🚀 총 {len(raw_files)}개의 파일 변환 시작...")

    for i, raw_path in enumerate(raw_files):
        filename = os.path.basename(raw_path)
        save_name = filename.replace('.raw', '.png')
        save_path = os.path.join(_szDstPath, save_name)

        try:
            # 1. 파일 읽기
            raw_data = np.fromfile(raw_path, dtype='uint16')
            
            # Width 감지 (640으로 가정, 안 맞으면 896 등 시도)
            # 파일 크기에 맞춰서 적절한 width를 찾습니다.
            total_pixels = raw_data.size
            width = 640 # 기본값
            
            if total_pixels % 640 == 0: width = 640
            elif total_pixels % 896 == 0: width = 896
            elif total_pixels % 1280 == 0: width = 1280
            
            img = raw_data.reshape((-1, width))
            
            # 2. 배경 정규화 (Normalization) - 누렇게 뜨는 것 방지
            bkg_val = bkg_level(img)
            # 배경이 너무 어두우면(값이 작으면) 왜곡되므로 최소값 보정
            if bkg_val < 10000: bkg_val = 65535 
            
            # 전체 밝기 스케일링 (배경을 65535로 맞춤)
            img = np.clip(img.astype(np.float32) * (65535.0 / bkg_val), 0, 65535).astype(np.uint16)

            # 3. HE/LE 분리
            vd = img.shape[0] // 2
            
            # ★★★ [설정] LE = 위쪽 (Top), HE = 아래쪽 (Bottom) ★★★
            le_img = img[:vd, :]
            he_img = img[vd:, :]

            # 4. Zeff 및 컬러 생성
            zeff_img = make_zeff_image(le_img, he_img, d=0)
            make_pseudo_image_save(he_img, zeff_img, save_path)
            
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{len(raw_files)}] 처리 완료...")

        except Exception as e:
            print(f"❌ 에러({filename}): {e}")

    print("\n🎉 모든 작업 완료!")

if __name__ == "__main__":
    main()