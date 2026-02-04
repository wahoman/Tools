import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

# =========================================================
# 1. 설정 (경로 및 파라미터)
# =========================================================
# 처리할 RAW 폴더
SRC_RAW_FOLDER = r"D:\hgyeo\BCAS_TIP\bare_image_raw\Adaptor\raws"

# 결과 저장할 폴더
DST_COLOR_FOLDER = r"D:\hgyeo\BCAS_TIP\bare_image_raw\Adaptor\colored_test"

# ★ LUT 파일 경로 ★
LUT_PATH = r"D:\hgyeo\BCAS_TIP\LUT_ZeffU2.csv"
CLUT_PATH = r"D:\hgyeo\BCAS_TIP\Gray.clut"

RAW_WIDTH = 896          
RAW_DTYPE = np.uint16    

# Adaptor는 물체이므로: 위가 HE, 아래가 LE
TOP_IS_HE = True         

# =========================================================
# 2. 컬러 매핑 상수 및 테이블
# =========================================================
_zI0 = 98000.0
_Z_CONST = [
    1.193710157, 1.194429011, 1.195497014, 1.197598952, 1.201261874,
    1.206871776, 1.215205132, 1.226352526, 1.240297472, 1.257289302,
    1.27849552, 1.304551537, 1.334194598, 1.367066563, 1.402681085,
    1.440472022, 1.479718307, 1.519644992, 1.559941821, 1.599809092,
    1.638299752, 1.675213149, 1.709797614, 1.742333096, 1.772344093,
    1.799326033, 1.823810917, 1.846055878, 1.864917778, 1.881765843
]

g_pseudoColorZ = None
g_pseudoColorK = None
g_zeffTbl = np.zeros(2000, dtype=np.uint16)

def load_luts():
    """ LUT 파일 로드 및 Zeff 테이블 생성 """
    global g_pseudoColorZ, g_pseudoColorK, g_zeffTbl
    
    # 1. Zeff Lookup Table 생성
    nZ = [0] * 31
    noffset = 1000
    for i in range(30):
        fv = float(_Z_CONST[i])
        nZ[i] = max(int((fv + 0.0005) * 1000) - noffset, 0)
    
    idx = 0
    for k in range(nZ[0]):
        g_zeffTbl[idx] = 1 << 8
        idx += 1
    for j in range(30):
        if idx >= 2000: break
        nW = nZ[j + 1] - nZ[j]
        for k in range(nW):
            if idx >= 2000: break
            g_zeffTbl[idx] = int(((j + 1) << 8) + (k << 8) / nW)
            idx += 1
    while idx < 2000:
        g_zeffTbl[idx] = 30 << 8
        idx += 1

    # 2. Color Palette 로드 (수정됨: 빈 문자열 처리)
    try:
        with open(LUT_PATH, 'rb') as f:
            content = f.read().decode('utf-16')
            lines = content.splitlines()
            g_pseudoColorZ = np.zeros((300, 256, 3), dtype=np.uint8) # BGR
            
            for j in range(min(300, len(lines))):
                line_str = lines[j].strip()
                if not line_str: continue 

                parts = [x.strip() for x in line_str.split(',') if x.strip()]
                vals = list(map(int, parts))
                if len(vals) < 256 * 3: continue
                
                for k in range(256):
                    r, g, b = vals[k*3], vals[k*3+1], vals[k*3+2]
                    g_pseudoColorZ[j, k] = [b, g, r] # BGR 순서로 저장
                    
    except Exception as e:
        print(f"❌ LUT 로드 실패: {e}")
        return False

    # 3. Gray Palette 로드
    try:
        with open(CLUT_PATH, 'rb') as f:
            f.seek(8)
            buf = f.read()
            g_pseudoColorK = np.zeros((256, 3), dtype=np.uint8)
            for k in range(256):
                r, g, b = buf[k*4], buf[k*4+1], buf[k*4+2]
                g_pseudoColorK[k] = [b, g, r]
    except Exception as e:
        print(f"❌ Gray CLUT 로드 실패: {e}")
        return False
    
    return True

# =========================================================
# 3. 밝기 보정 함수 (Legacy 코드 이식)
# =========================================================
def bkg_level_cpu(arr, bin_center=50000, bin_gap=16):
    """
    이미지의 배경 밝기 레벨(Background Level)을 추정하는 함수
    """
    bins = [bin_center + bin_gap*(n-128) for n in range(256)]
    hist, _ = np.histogram(arr, bins)
    
    if np.sum(hist) == 0: 
        return float(bin_center)
        
    peak_pos = bin_center + bin_gap*((np.argmax(hist)+0.5-128))
    return float(peak_pos)

# =========================================================
# 4. 이미지 처리 함수
# =========================================================
def process_single_image(raw_path, save_path):
    try:
        # 1. Load Raw
        arr = np.fromfile(raw_path, dtype=RAW_DTYPE)
        if arr.size % RAW_WIDTH != 0:
            return
        
        full_img = arr.reshape(-1, RAW_WIDTH)
        h = full_img.shape[0]
        mid = h // 2
        
        # 2. Split (Top/Bottom)
        top = full_img[:mid, :]
        bot = full_img[mid:, :]
        
        # 3. Assign HE/LE (객체 기준: Top=HE, Bot=LE)
        if TOP_IS_HE:
            img_he = top
            img_le = bot
        else:
            img_he = bot
            img_le = top

        # ★★★★★★★ [추가된 부분] 밝기 정규화 (Normalization) ★★★★★★★
        # 배경(img_he)의 밝기를 측정해서 65535로 맞춤
        # 이렇게 해야 Legacy 코드처럼 밝고 선명하게 나옴
        bkg_val = bkg_level_cpu(img_he)
        
        # 너무 어두운 값으로 측정되면 그냥 흰색(65535)으로 간주
        if bkg_val < 10000: 
            bkg_val = 65535
        
        scale_factor = 65535.0 / bkg_val
        
        # HE와 LE 모두 동일한 비율로 밝게 만듦 (float32 변환 후 곱셈 -> uint16 복귀)
        img_he = np.clip(img_he.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)
        img_le = np.clip(img_le.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # 4. Compute Zeff
        I_low = np.maximum(img_le.astype(np.float32), 1.0)
        I_high = np.maximum(img_he.astype(np.float32), 1.0)
        
        num = np.log(_zI0 / I_low)
        den = np.log(_zI0 / I_high)
        den = np.where(den < 1e-6, 1e-6, den)
        
        ratio = num / den - 1.0
        rate = np.clip((ratio * 1000).astype(np.int32), 0, 1999)
        z_idx_map = g_zeffTbl[rate]

        # 5. Color Mapping
        pwSrc = (img_he >> 8).astype(np.uint8)
        pbZeff = ((z_idx_map.astype(np.float32) * 10) / 256).astype(np.int32) - 1
        pbZeff = np.clip(pbZeff, -1, 299)
        
        h_out, w_out = img_he.shape
        color_img = np.zeros((h_out, w_out, 3), dtype=np.uint8)
        
        mask_dark = pbZeff < 0  
        mask_valid = ~mask_dark 
        
        if np.any(mask_dark):
            idx_src = pwSrc[mask_dark]
            color_img[mask_dark] = g_pseudoColorK[idx_src]
            
        if np.any(mask_valid):
            idx_z = pbZeff[mask_valid]
            idx_src = pwSrc[mask_valid]
            color_img[mask_valid] = g_pseudoColorZ[idx_z, idx_src]

        # 6. Save
        cv2.imwrite(save_path, color_img)

    except Exception as e:
        print(f"Error processing {os.path.basename(raw_path)}: {e}")

# =========================================================
# 5. 메인 실행
# =========================================================
if __name__ == "__main__":
    print("[*] 컬러 매핑 테스트 시작")
    
    if not load_luts():
        print("프로그램을 종료합니다.")
        exit()
        
    os.makedirs(DST_COLOR_FOLDER, exist_ok=True)
    raw_files = glob.glob(os.path.join(SRC_RAW_FOLDER, "*.raw"))
    print(f"[*] 대상 파일: {len(raw_files)}개")
    
    for raw_path in tqdm(raw_files):
        filename = os.path.basename(raw_path)
        save_name = filename.replace(".raw", ".png")
        save_path = os.path.join(DST_COLOR_FOLDER, save_name)
        process_single_image(raw_path, save_path)
        
    print("[*] 완료되었습니다.")