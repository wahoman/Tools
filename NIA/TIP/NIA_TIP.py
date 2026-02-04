import sys
import os
import cv2
import numpy as np
import random
from glob import glob
from collections import defaultdict
from multiprocessing import Pool, freeze_support
import cupy as cp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFileDialog, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QCheckBox, QSpinBox, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings

# =========================================================
# 1. 경로 및 설정
# =========================================================
# ★★★ 요청하신 새 경로로 수정됨 ★★★
DEFAULT_PATHS = {
    "_szTIPRootPath": "D:/hgyeo/BCAS_TIP/bare_image_raw_crop",
    "_szDstRootPath": "D:/hgyeo/BCAS_TIP/TIP_output",
    "_szSrcPath":     "D:/hgyeo/BCAS_TIP/APIDS Bare Bags",
    "_szPalettePath": "D:/hgyeo/BCAS_TIP/LUT_ZeffU2.csv",
    "_szGrayClutPath":"D:/hgyeo/BCAS_TIP/Gray.clut"
}

TARGET_H = 760          # 모든 이미지는 높이 760px 기준
AVG_PIXEL_ROWS = 20     # 배경 패딩용 하단 참조 라인 수

# =========================================================
# 2. 전역 변수
# =========================================================
_g_config = {}
_g_zeffTbl = [[0] * 2000 for _ in range(9)] # (올바른 방식)
_g_pseudoColorZ = None
_g_pseudoColorK = None
_g_rawdatas = []

_zI0 = [98000] * 9
_Z = [
    [1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,
     1.206871776,1.215205132,1.226352526,1.240297472,1.257289302,
     1.27849552,1.304551537,1.334194598,1.367066563,1.402681085,
     1.440472022,1.479718307,1.519644992,1.559941821,1.599809092,
     1.638299752,1.675213149,1.709797614,1.742333096,1.772344093,
     1.799326033,1.823810917,1.846055878,1.864917778,1.881765843]
] * 9

# =========================================================
# 3. 핵심 유틸리티 함수
# =========================================================

def normalize_height_uint16(img, target_h=760):
    """ 배경 이미지의 높이를 target_h(760)로 맞춤 (패딩 or 크롭) """
    h, w = img.shape
    if h == target_h:
        return img
    elif h < target_h:
        pad_h = target_h - h
        rows_to_avg = min(h, AVG_PIXEL_ROWS)
        avg_val = int(np.mean(img[-rows_to_avg:, :])) if rows_to_avg > 0 else 65535
        # 16비트 이미지 패딩
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=avg_val)
    else:
        return img[:target_h, :]

def get_bbox_4way_scan(img, threshold=45000):
    """ 가방 영역 BBox 추출 """
    mask = img < threshold
    if not np.any(mask): return 0, 0, 0, 0
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, (x_max - x_min), (y_max - y_min)

def transform_polygon_label(src_label_path, offset_x, offset_y, tip_w, tip_h, bg_w, bg_h):
    """ 라벨 좌표 변환: TIP 기준 -> 배경 기준 """
    new_lines = []
    if not os.path.exists(src_label_path):
        return []

    with open(src_label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue 
        
        cls_idx = parts[0]
        coords = list(map(float, parts[1:]))
        new_coords = []
        for i in range(0, len(coords), 2):
            # 1. 픽셀 좌표 복원 (TIP 이미지 내)
            px = coords[i] * tip_w
            py = coords[i+1] * tip_h
            
            # 2. 배경 위에서의 좌표 (Offset 추가)
            dst_px = px + offset_x
            dst_py = py + offset_y
            
            # 3. 배경 이미지 기준 정규화
            new_nx = dst_px / bg_w
            new_ny = dst_py / bg_h
            new_coords.extend([new_nx, new_ny])
            
        new_line = f"{cls_idx} " + " ".join([f"{val:.6f}" for val in new_coords]) + "\n"
        new_lines.append(new_line)
    return new_lines

def bkg_level_cpu(arr, bin_center=50000, bin_gap=16):
    """ 배경 밝기 측정 """
    bins = [bin_center + bin_gap*(n-128) for n in range(256)]
    hist, _ = np.histogram(arr, bins)
    if np.sum(hist) == 0: return float(bin_center)
    peak_pos = bin_center + bin_gap*((np.argmax(hist)+0.5-128))
    return float(peak_pos)

# =========================================================
# 4. GPU 처리 함수 (CuPy)
# =========================================================
def apply_polygon_mask(img_arr, label_path):
    """
    라벨 파일(txt)을 읽어서 폴리곤 영역 밖을 65535(White/Transparent)로 날려버림
    """
    if not os.path.exists(label_path):
        return img_arr

    h, w = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8) # 검은 배경 마스크 생성

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            # YOLO Polygon 포맷 가정: <class> <x1> <y1> <x2> <y2> ...
            # 0번 인덱스는 클래스이므로 제외하고 1번부터 좌표
            coords = list(map(float, parts[1:]))
            
            # 좌표가 짝수 개여야 함
            if len(coords) < 6: continue 
            
            # 정규화된 좌표(0~1)를 픽셀 좌표(w, h)로 변환
            pts = []
            for i in range(0, len(coords), 2):
                px = int(coords[i] * w)
                py = int(coords[i+1] * h)
                pts.append([px, py])
            
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 폴리곤 내부를 흰색(255)으로 칠함
            cv2.fillPoly(mask, [pts], 255)
            
        # 마스크가 그려졌다면(물체가 있다면) 적용
        if np.max(mask) > 0:
            # mask가 0인 부분(폴리곤 밖)을 65535로 변경
            img_arr = np.where(mask == 0, 65535, img_arr)
            
    except Exception as e:
        pass # 에러나면 원본 리턴

    return img_arr


def tip_merge_gpu(img, tipH, tipL, vd):
    """ GPU 합성: img는 [LE; HE] 스택 상태 """
    img_height, img_width = img.shape
    tip_height, tip_width = tipH.shape
    
    # vd(760) 기준 위쪽이 LE
    img_le = img[:vd, :] 
    
    bx, by, bw, bh = get_bbox_4way_scan(img_le, threshold=45000)
    if bw < tip_width or bh < tip_height: return img, 0, 0, 0, 0

    x, y = 0, 0
    found_spot = False
    
    for _ in range(100):
        margin_x = int(bw * 0.05)
        margin_y = int(bh * 0.05)
        min_x = bx + margin_x
        max_x = bx + bw - tip_width - margin_x
        min_y = by + margin_y
        max_y = by + bh - tip_height - margin_y
        
        if max_x <= min_x or max_y <= min_y: continue

        cand_x = random.randint(min_x, max_x)
        cand_y = random.randint(min_y, max_y)

        roi_check = img_le[cand_y:cand_y+tip_height, cand_x:cand_x+tip_width]
        valid_pixels = np.count_nonzero(roi_check < 48000)
        ratio = valid_pixels / (tip_width * tip_height)

        if ratio > 0.6:
            x = cand_x
            y = cand_y
            found_spot = True
            break

    if not found_spot: return img, 0, 0, 0, 0

    # CuPy Array 생성
    roi_L_cpu = img[y:y + tip_height, x:x + tip_width]
    roi_H_cpu = img[y + vd:y + vd + tip_height, x:x + tip_width]

    roi_L_gpu = cp.asarray(roi_L_cpu, dtype=cp.float32)
    roi_H_gpu = cp.asarray(roi_H_cpu, dtype=cp.float32)
    tipL_gpu = cp.asarray(tipL, dtype=cp.float32)
    tipH_gpu = cp.asarray(tipH, dtype=cp.float32)

    # 합성 (Multiply)
    blended_L_gpu = roi_L_gpu * (tipL_gpu / 65535.0)
    blended_H_gpu = roi_H_gpu * (tipH_gpu / 65535.0)

    blended_L_gpu = cp.clip(blended_L_gpu, 0, 65535)
    blended_H_gpu = cp.clip(blended_H_gpu, 0, 65535)

    # 결과 반영
    img[y:y + tip_height, x:x + tip_width] = cp.asnumpy(blended_L_gpu).astype(np.uint16)
    img[y + vd:y + vd + tip_height, x:x + tip_width] = cp.asnumpy(blended_H_gpu).astype(np.uint16)

    return img, x, y, tip_width, tip_height

def make_zeff_image_gpu(imgLoE, imgHiE, d=0):
    if imgLoE.size <= 0 or imgHiE.size <= 0: return np.zeros_like(imgLoE, dtype=np.uint16)
    imgLoE_cp = cp.asarray(imgLoE, dtype=cp.float32)
    imgHiE_cp = cp.asarray(imgHiE, dtype=cp.float32)
    imgLoE_cp = cp.maximum(imgLoE_cp, 1.0)
    imgHiE_cp = cp.maximum(imgHiE_cp, 1.0)
    zI0_val = float(_zI0[d])
    numerator = cp.log(zI0_val / imgLoE_cp)
    denominator = cp.log(zI0_val / imgHiE_cp)
    denominator = cp.where(denominator == 0, 1e-6, denominator)
    ratio_cp = numerator / denominator - 1
    rate_cp = cp.clip((ratio_cp * 1000).astype(cp.int32), 0, 1999)
    zeff_tbl_cp = cp.asarray(_g_zeffTbl[d], dtype=cp.uint16)
    return cp.asnumpy(zeff_tbl_cp[rate_cp])

def make_pseudo_image_save(pImg, pImgzs, save_path):
    nHeight, nWidth = pImg.shape
    pImgDst = np.zeros((nHeight, nWidth, 4), dtype=np.uint8)
    pwSrc = (pImg >> 8).astype(np.uint8)
    pbZeff = ((pImgzs.astype(np.float32) * 10) / 256).astype(np.int32) - 1
    pbZeff = np.clip(pbZeff, -1, 299)
    mask_dark = pbZeff < 0
    mask_valid = ~mask_dark
    
    if np.any(mask_dark):
        indices = pwSrc[mask_dark]
        pImgDst[mask_dark, 0] = _g_pseudoColorK[indices]['rgbRed']
        pImgDst[mask_dark, 1] = _g_pseudoColorK[indices]['rgbGreen']
        pImgDst[mask_dark, 2] = _g_pseudoColorK[indices]['rgbBlue']
    if np.any(mask_valid):
        z_indices = pbZeff[mask_valid]
        src_indices = pwSrc[mask_valid]
        pImgDst[mask_valid, 0] = _g_pseudoColorZ[z_indices, src_indices]['rgbRed']
        pImgDst[mask_valid, 1] = _g_pseudoColorZ[z_indices, src_indices]['rgbGreen']
        pImgDst[mask_valid, 2] = _g_pseudoColorZ[z_indices, src_indices]['rgbBlue']
    pImgDst[..., 3] = 255
    cv2.imwrite(save_path, cv2.cvtColor(pImgDst, cv2.COLOR_RGBA2BGR))
    return True

# =========================================================
# 5. 작업자 로직 (Worker Logic)
# =========================================================

def load_tables(palette_path, gray_path):
    global _g_pseudoColorZ, _g_pseudoColorK, _g_zeffTbl
    for d in range(9):
        nZ = [0] * 31
        noffset = 1000
        for i in range(30):
            fv = float(_Z[d][i])
            nZ[i] = max(int((fv + 0.0005) * 1000) - noffset, 0)
        i = 0
        for k in range(nZ[0]):
            _g_zeffTbl[d][i] = 1 << 8
            i += 1
        for j in range(30):
             if i >= 2000: break
             nW = nZ[j + 1] - nZ[j]
             for k in range(nW):
                 if i >= 2000: break
                 _g_zeffTbl[d][i] = int(((j + 1) << 8) + (k << 8) / nW)
                 i += 1
        while i < 2000:
            _g_zeffTbl[d][i] = 30 << 8
            i += 1
            
    try:
        with open(palette_path, 'rb') as file:
            content = file.read().decode('utf-16')
            lines = content.splitlines()
            _g_pseudoColorZ = np.zeros((300, 256), dtype=[('rgbRed', 'u1'), ('rgbGreen', 'u1'), ('rgbBlue', 'u1'), ('rgbReserved', 'u1')])
            for j in range(min(300, len(lines))):
                line = lines[j].strip()
                if not line: continue
                parts = [x.strip() for x in line.split(',') if x.strip()]
                vals = list(map(int, parts))
                if len(vals) < 256 * 3: continue
                t = 0
                for k in range(256):
                    _g_pseudoColorZ[j][k]['rgbRed'] = vals[t]
                    _g_pseudoColorZ[j][k]['rgbGreen'] = vals[t+1]
                    _g_pseudoColorZ[j][k]['rgbBlue'] = vals[t+2]
                    _g_pseudoColorZ[j][k]['rgbReserved'] = 255
                    t += 3
    except: pass
    
    _g_pseudoColorK = np.zeros(256, dtype=[('rgbRed', 'u1'), ('rgbGreen', 'u1'), ('rgbBlue', 'u1'), ('rgbReserved', 'u1')])
    try:
        with open(gray_path, 'rb') as file:
            file.seek(8)
            ptBuf = file.read()
            for k in range(256):
                _g_pseudoColorK[k]['rgbRed'] = ptBuf[k * 4]
                _g_pseudoColorK[k]['rgbGreen'] = ptBuf[k * 4 + 1]
                _g_pseudoColorK[k]['rgbBlue'] = ptBuf[k * 4 + 2]
                _g_pseudoColorK[k]['rgbReserved'] = ptBuf[k * 4 + 3]
    except: pass

def init_worker(config, raw_files):
    global _g_config, _g_rawdatas
    try: cv2.setNumThreads(0)
    except: pass
    _g_config = config
    _g_rawdatas = raw_files
    load_tables(_g_config['palette_path'], _g_config['gray_path'])

def process_class_task(task_data):
    class_folder, target_count = task_data
    tip_root = _g_config['tip_root']
    dst_root = _g_config['dst_root']
    
    current_tip_path = os.path.join(tip_root, class_folder)
    current_dst_path = os.path.join(dst_root, class_folder)
    
    dst_img_path = os.path.join(current_dst_path, "images")
    dst_lbl_path = os.path.join(current_dst_path, "labels")
    
    os.makedirs(dst_img_path, exist_ok=True)
    os.makedirs(dst_lbl_path, exist_ok=True)

    # TH 파일 기준 검색
    lst_tipdatas = sorted(glob(os.path.join(current_tip_path, '*_TH_*.raw')))
    if not lst_tipdatas: return f"[{class_folder}] ⚠️ 실패: 폴더 내에 raw 파일이 없음"

    success_count = 0
    fail_consecutive = 0

    while success_count < target_count:
        is_success = False
        
        for _retry in range(500):
            tip_file = random.choice(lst_tipdatas)
            tip_info = os.path.basename(tip_file).split('_')
            if len(tip_info) < 4: continue 
            
            tip_prefix = f"{tip_info[0]}_{tip_info[1]}"
            try:
                tip_dim = tip_info[-1].split('.')[0]
                tw, th = map(int, tip_dim.split('x'))
            except: continue

            tip_le_path = tip_file.replace('_TH_', '_TL_')
            src_label_path = tip_file.replace('.raw', '.txt')
            
            if not os.path.exists(tip_le_path) or not os.path.exists(src_label_path): continue

            try:
                # ---------------------------------------------------------
                # 1. 데이터 로드
                # ---------------------------------------------------------
                raw_he = np.fromfile(tip_file, dtype='uint16').reshape(-1, tw)
                raw_le = np.fromfile(tip_le_path, dtype='uint16').reshape(-1, tw)

                # ---------------------------------------------------------
                # 2. [핵심] 보여주신 코드의 "지능형 밝기 정규화" 이식
                # 배경 밝기를 측정해서 65535로 쫙 펴줌 -> 색감이 선명해짐
                # ---------------------------------------------------------
                # HE 이미지 기준으로 배경 레벨 측정
                bkg_val = bkg_level_cpu(raw_he) 
                if bkg_val < 10000: bkg_val = 65535 # 너무 어두우면 예외처리

                scale_factor = 65535.0 / bkg_val
                
                # 과도한 증폭 방지 (최대 5배까지만 허용)
                if scale_factor > 5.0: scale_factor = 1.0

                # HE/LE에 동일한 비율 적용 (비율이 깨지면 색이 변하므로 같이 적용)
                tip_he = np.clip(raw_he.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)
                tip_le = np.clip(raw_le.astype(np.float32) * scale_factor, 0, 65535).astype(np.uint16)

                # ---------------------------------------------------------
                # 3. [핵심] 폴리곤 마스킹 (Polygon Masking)
                # 정규화를 했어도 남은 자잘한 노이즈를 라벨 모양대로 오려냄
                # ---------------------------------------------------------
                tip_he = apply_polygon_mask(tip_he, src_label_path)
                tip_le = apply_polygon_mask(tip_le, src_label_path)

                # ---------------------------------------------------------
                # 4. [보험] 잔여 노이즈 제거 (White Clipping)
                # 마스킹 후에도 엣지에 남은 아주 밝은 회색들을 투명하게 처리
                # 정규화가 잘 됐다면 배경은 이미 65535 근처일 것임
                # ---------------------------------------------------------
                WHITE_TH = 64500 # 65535에 아주 가까운 값만 날림
                tip_he = np.where(tip_he > WHITE_TH, 65535, tip_he)
                tip_le = np.where(tip_le > WHITE_TH, 65535, tip_le)

            except Exception as e:
                continue

            # 배경 로드 (기존 유지)
            raw_file = random.choice(_g_rawdatas)
            raw_lid = os.path.basename(raw_file)[:-6]
            d = random.randint(0, 8)
            raw_path = f"{_g_config['src_path']}/{raw_lid}_{d}.raw"
            if not os.path.exists(raw_path): continue

            bg_width = 0
            try: 
                bg_raw = np.fromfile(raw_path, dtype='uint16')
                if bg_raw.size % 640 == 0: bg_width = 640
                elif bg_raw.size % 896 == 0: bg_width = 896
                else: continue 

                bg_full = bg_raw.reshape(-1, bg_width)
                mid = bg_full.shape[0] // 2
                
                bg_le = bg_full[:mid, :] 
                bg_he = bg_full[mid:, :] 

                bg_le_norm = normalize_height_uint16(bg_le, TARGET_H)
                bg_he_norm = normalize_height_uint16(bg_he, TARGET_H)

                img = np.vstack([bg_le_norm, bg_he_norm])
            except: continue

            if img.shape[0]//2 < th or img.shape[1] < tw: continue
            vd = TARGET_H 

            # 합성
            img, x, y, ftw, fth = tip_merge_gpu(img, tip_he, tip_le, vd)
            if ftw == 0: continue

            # 결과 평탄화 (선택 사항)
            bkg = bkg_level_cpu(img)
            if bkg < 10000: bkg = 65535
            img = np.clip(img.astype(np.float32) * (65535.0 / bkg), 0, 65535).astype(np.uint16)

            # 컬러 매핑
            le_img = img[:vd, :]
            he_img = img[vd:, :]
            imgZeff = make_zeff_image_gpu(le_img, he_img, d)
            
            success_count += 1
            sLID = f"{tip_prefix}_{success_count}_{random.randint(1000,9999)}"
            save_path = os.path.join(dst_img_path, f"{sLID}.png")
            
            make_pseudo_image_save(he_img, imgZeff, save_path)

            # 라벨 저장
            new_label_lines = transform_polygon_label(src_label_path, x, y, ftw, fth, bg_width, TARGET_H)
            
            with open(os.path.join(dst_lbl_path, f"{sLID}.txt"), 'w') as f:
                if new_label_lines:
                    f.writelines(new_label_lines)
                else:
                    f.write(f"0 {(x + ftw/2)/bg_width:.6f} {(y + fth/2)/TARGET_H:.6f} {ftw/bg_width:.6f} {fth/TARGET_H:.6f}\n")
            
            is_success = True
            fail_consecutive = 0
            break 
        
        if not is_success:
            fail_consecutive += 1
            if fail_consecutive > 10: 
                return f"[{class_folder}] ⚠️ 부분 완료"
    
    return f"[{class_folder}] ✅ 완료: {success_count}장"

class ProcessingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, config, task_list, raw_files):
        super().__init__()
        self.config = config
        self.task_list = task_list
        self.raw_files = raw_files
    def run(self):
        self.log_signal.emit("🚀 프로세스 시작... (Pool 초기화 중)")
        try:
            with Pool(processes=10, initializer=init_worker, initargs=(self.config, self.raw_files)) as pool:
                for result in pool.imap_unordered(process_class_task, self.task_list):
                    self.log_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"❌ 에러 발생: {str(e)}")
        self.log_signal.emit("✅ 전체 작업 완료!")
        self.finished_signal.emit()

class MergeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("MyCompany", "XRayMergeApp_Final")
        self.is_all_selected = True 
        self.initUI()
        self.load_last_paths()

    def initUI(self):
        self.setWindowTitle("X-ray Polygon Synthesis (Final Fixed)")
        self.setGeometry(100, 100, 800, 750)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.path_widgets = {}
        paths = [
            ("TIP Root Path", "_szTIPRootPath", DEFAULT_PATHS["_szTIPRootPath"]),
            ("Dst Root Path", "_szDstRootPath", DEFAULT_PATHS["_szDstRootPath"]),
            ("Src Path (Bg)", "_szSrcPath", DEFAULT_PATHS["_szSrcPath"]),
            ("Palette Path", "_szPalettePath", DEFAULT_PATHS["_szPalettePath"]),
            ("Gray Clut Path", "_szGrayClutPath", DEFAULT_PATHS["_szGrayClutPath"]),
        ]

        for label, key, default in paths:
            h_layout = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(100)
            line = QLineEdit(default)
            btn = QPushButton("...")
            btn.setFixedWidth(40)
            if "csv" in default or "clut" in default:
                btn.clicked.connect(lambda _, l=line: self.browse_file(l))
            else:
                btn.clicked.connect(lambda _, l=line: self.browse_folder(l))
                if key == "_szTIPRootPath":
                    line.textChanged.connect(self.scan_classes) 
            self.path_widgets[key] = line
            h_layout.addWidget(lbl)
            h_layout.addWidget(line)
            h_layout.addWidget(btn)
            layout.addLayout(h_layout)

        layout.addWidget(QLabel("📂 클래스 관리 및 설정"))
        
        control_layout = QHBoxLayout()
        self.btn_toggle = QPushButton("전체 해제")
        self.btn_toggle.clicked.connect(self.toggle_selection)
        control_layout.addWidget(self.btn_toggle)

        control_layout.addWidget(QLabel("일괄 수량:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 100000)
        self.spin_batch.setValue(50)
        control_layout.addWidget(self.spin_batch)

        self.btn_apply_batch = QPushButton("적용")
        self.btn_apply_batch.clicked.connect(self.apply_batch_count)
        control_layout.addWidget(self.btn_apply_batch)

        self.btn_refresh = QPushButton("새로고침")
        self.btn_refresh.clicked.connect(self.scan_classes)
        control_layout.addWidget(self.btn_refresh)
        layout.addLayout(control_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["선택", "클래스 이름", "합성 개수"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("📝 로그"))
        layout.addWidget(self.log_text)

        self.btn_start = QPushButton("합성 시작")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_start.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_start)

    def toggle_selection(self):
        self.is_all_selected = not self.is_all_selected
        target_state = self.is_all_selected
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk: chk.setChecked(target_state)
        if self.is_all_selected: self.btn_toggle.setText("전체 해제")
        else: self.btn_toggle.setText("전체 선택")

    def apply_batch_count(self):
        val = self.spin_batch.value()
        for i in range(self.table.rowCount()):
            widget = self.table.cellWidget(i, 2)
            if widget:
                spin = widget.findChild(QSpinBox)
                if spin: spin.setValue(val)
        QMessageBox.information(self, "알림", f"모든 클래스의 수량이 {val}개로 설정되었습니다.")

    def browse_folder(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택", line_edit.text())
        if path: line_edit.setText(path)

    def browse_file(self, line_edit):
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", line_edit.text())
        if path: line_edit.setText(path)

    def load_last_paths(self):
        for key, widget in self.path_widgets.items():
            val = self.settings.value(key)
            if val: widget.setText(str(val))
        self.scan_classes()

    def save_current_paths(self):
        for key, widget in self.path_widgets.items():
            self.settings.setValue(key, widget.text())

    def scan_classes(self):
        tip_path = self.path_widgets["_szTIPRootPath"].text()
        if not os.path.isdir(tip_path): return
        self.table.setRowCount(0)
        try:
            folders = sorted([f for f in os.listdir(tip_path) if os.path.isdir(os.path.join(tip_path, f))])
            self.table.setRowCount(len(folders))
            for i, folder in enumerate(folders):
                chk_widget = QWidget()
                chk_layout = QHBoxLayout(chk_widget)
                chk_layout.setContentsMargins(0,0,0,0)
                chk_layout.setAlignment(Qt.AlignCenter)
                chk = QCheckBox()
                chk.setChecked(True) 
                chk_layout.addWidget(chk)
                self.table.setCellWidget(i, 0, chk_widget)
                self.table.setItem(i, 1, QTableWidgetItem(folder))
                sp_widget = QWidget()
                sp_layout = QHBoxLayout(sp_widget)
                sp_layout.setContentsMargins(0,0,0,0)
                sp_layout.setAlignment(Qt.AlignCenter)
                spin = QSpinBox()
                spin.setRange(1, 10000)
                spin.setValue(50) 
                sp_layout.addWidget(spin)
                self.table.setCellWidget(i, 2, sp_widget)
            self.is_all_selected = True
            self.btn_toggle.setText("전체 해제")
        except Exception as e:
            self.log(f"클래스 로딩 실패: {e}")

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def start_processing(self):
        config = {
            'tip_root': self.path_widgets["_szTIPRootPath"].text(),
            'dst_root': self.path_widgets["_szDstRootPath"].text(),
            'src_path': self.path_widgets["_szSrcPath"].text(),
            'palette_path': self.path_widgets["_szPalettePath"].text(),
            'gray_path': self.path_widgets["_szGrayClutPath"].text(),
        }
        for k, v in config.items():
            if k == 'dst_root': continue
            if not os.path.exists(v):
                QMessageBox.critical(self, "경로 오류", f"경로가 존재하지 않습니다:\n{v}")
                return
        task_list = []
        for i in range(self.table.rowCount()):
            chk = self.table.cellWidget(i, 0).findChild(QCheckBox)
            if chk.isChecked():
                class_name = self.table.item(i, 1).text()
                count = self.table.cellWidget(i, 2).findChild(QSpinBox).value()
                task_list.append((class_name, count))
        if not task_list:
            QMessageBox.warning(self, "알림", "선택된 클래스가 없습니다.")
            return
        raw_files = sorted(glob(os.path.join(config['src_path'], '*_0.raw')))
        if not raw_files:
            QMessageBox.critical(self, "오류", "배경 이미지(*_0.raw)를 찾을 수 없습니다.")
            return
        self.save_current_paths()
        self.btn_start.setEnabled(False)
        self.log(f"총 {len(task_list)}개 클래스 작업을 시작합니다...")
        self.thread = ProcessingThread(config, task_list, raw_files)
        self.thread.log_signal.connect(self.log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        QMessageBox.information(self, "완료", "모든 작업이 완료되었습니다.")

if __name__ == '__main__':
    freeze_support()
    app = QApplication(sys.argv)
    window = MergeApp()
    window.show()
    sys.exit(app.exec_())