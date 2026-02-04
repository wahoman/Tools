import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    "RAW_ROOT_DIR": r"D:\hgyeo\BCAS_TIP\bare_image_raw",
    "LABEL_ROOT_DIR": r"D:\hgyeo\BCAS_TIP\bare_image_png",
    "SAVE_ROOT_DIR": r"D:\hgyeo\BCAS_TIP\bare_image_raw_crop",
    "RAW_WIDTH": 896,
    "RAW_DTYPE": np.uint16,
    "PADDING_PIXELS": 1,
    "TARGET_H": 760,
    "AVG_PIXEL_ROWS": 20,
    # 프로세스 개수 설정 (None=CPU코어개수 자동할당, HDD면 4~6 권장, SSD면 최대치)
    "NUM_WORKERS": max(1, multiprocessing.cpu_count() - 2) 
}

# ==========================================
# 2. 유틸리티 함수들 (독립적 실행 가능하도록 구성)
# ==========================================
def normalize_height_uint16(img, target_h=760):
    h, w = img.shape
    if h == target_h:
        return img
    elif h < target_h:
        pad_h = target_h - h
        rows_to_avg = min(h, CONFIG["AVG_PIXEL_ROWS"])
        if rows_to_avg > 0:
            avg_val = int(np.mean(img[-rows_to_avg:, :]))
        else:
            avg_val = 0
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=avg_val)
    else:
        return img[:target_h, :]

def get_polygon_mask_and_bbox(label_path, img_w, img_h, padding=0):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        if not lines: return None
        
        parts = lines[0].strip().split()
        cls_id = parts[0]
        coords = list(map(float, parts[1:]))
        
        pts = []
        for i in range(0, len(coords), 2):
            pts.append([int(coords[i] * img_w), int(coords[i+1] * img_h)])
        
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_np], 255)

        if padding > 0:
            kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        x, y, w, h = cv2.boundingRect(pts_np)
        x, y = max(0, x), max(0, y)
        w, h = min(img_w - x, w), min(img_h - y, h)
        return mask, (x, y, w, h), cls_id, coords
    except:
        return None

def crop_with_mask(image, mask, bbox):
    x, y, w, h = bbox
    masked_img = image.copy()
    masked_img[mask == 0] = 65535 
    return masked_img[y:y+h, x:x+w]

def save_renormalized_label(save_path, cls_id, coords, offset_x, offset_y, crop_w, crop_h, org_w, org_h):
    new_coords = []
    for i in range(0, len(coords), 2):
        org_px = coords[i] * org_w
        org_py = coords[i+1] * org_h
        
        new_nx = (org_px - offset_x) / crop_w
        new_ny = (org_py - offset_y) / crop_h
        
        new_coords.extend([max(0.0, min(1.0, new_nx)), max(0.0, min(1.0, new_ny))])
        
    with open(save_path, 'w') as f:
        f.write(f"{cls_id} " + " ".join([f"{val:.6f}" for val in new_coords]) + "\n")

# ==========================================
# 3. 워커 함수 (파일 1개 처리 로직)
# ==========================================
def process_single_file(args):
    """
    하나의 파일 세트를 처리하는 함수 (병렬로 실행됨)
    """
    idx, label_path, class_name, raw_dir, save_class_dir = args
    
    try:
        filename = os.path.basename(label_path)
        file_id = os.path.splitext(filename)[0]
        raw_path = os.path.join(raw_dir, f"{file_id}.raw")

        if not os.path.exists(raw_path): return False

        # 1. RAW 로드
        raw_data = np.fromfile(raw_path, dtype=CONFIG["RAW_DTYPE"])
        if raw_data.size % CONFIG["RAW_WIDTH"] != 0: return False
        
        full_img = raw_data.reshape(-1, CONFIG["RAW_WIDTH"])
        mid_h = full_img.shape[0] // 2
        
        # 2. HE/LE 분리
        img_top = full_img[:mid_h, :]
        img_bot = full_img[mid_h:, :]
        
        if img_top.mean() > img_bot.mean():
            img_th, img_tl = img_top, img_bot
        else:
            img_th, img_tl = img_bot, img_top

        # 3. 높이 정규화 (760px)
        img_th_760 = normalize_height_uint16(img_th, CONFIG["TARGET_H"])
        img_tl_760 = normalize_height_uint16(img_tl, CONFIG["TARGET_H"])

        # 4. 폴리곤 정보 계산
        poly_info = get_polygon_mask_and_bbox(
            label_path, CONFIG["RAW_WIDTH"], CONFIG["TARGET_H"], padding=CONFIG["PADDING_PIXELS"]
        )
        if poly_info is None: return False
        
        mask, bbox, cls_id, coords = poly_info
        x, y, w, h = bbox
        if w <= 0 or h <= 0: return False

        # 5. 크롭 & 저장
        crop_th = crop_with_mask(img_th_760, mask, bbox)
        crop_tl = crop_with_mask(img_tl_760, mask, bbox)
        
        final_h, final_w = crop_th.shape
        shape_str = f"{final_w}x{final_h}"
        
        th_name = f"{class_name}_{idx:04d}_TH_{shape_str}.raw"
        tl_name = f"{class_name}_{idx:04d}_TL_{shape_str}.raw"
        
        crop_th.tofile(os.path.join(save_class_dir, th_name))
        crop_tl.tofile(os.path.join(save_class_dir, tl_name))
        
        # 6. 라벨 저장
        txt_name = th_name.replace('.raw', '.txt')
        save_renormalized_label(
            os.path.join(save_class_dir, txt_name),
            cls_id, coords, x, y, final_w, final_h,
            CONFIG["RAW_WIDTH"], CONFIG["TARGET_H"]
        )
        
        return True

    except Exception as e:
        return False

# ==========================================
# 4. 메인 (멀티프로세싱 관리)
# ==========================================
def main():
    # 저장 폴더 초기화
    os.makedirs(CONFIG["SAVE_ROOT_DIR"], exist_ok=True)
    
    # 작업 목록 생성 (Task List)
    class_list = [d for d in os.listdir(CONFIG["LABEL_ROOT_DIR"]) 
                  if os.path.isdir(os.path.join(CONFIG["LABEL_ROOT_DIR"], d))]
    
    tasks = []
    
    print("[*] 작업 목록 생성 중...")
    for class_name in class_list:
        label_dir = os.path.join(CONFIG["LABEL_ROOT_DIR"], class_name, "labels")
        raw_dir = os.path.join(CONFIG["RAW_ROOT_DIR"], class_name, "raws")
        save_class_dir = os.path.join(CONFIG["SAVE_ROOT_DIR"], class_name)

        if not os.path.exists(label_dir) or not os.path.exists(raw_dir): continue
        os.makedirs(save_class_dir, exist_ok=True)
        
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        
        # 각 파일에 대해 (인덱스, 파일경로, 클래스명, RAW폴더, 저장폴더) 튜플 생성
        for idx, label_path in enumerate(label_files):
            tasks.append((idx, label_path, class_name, raw_dir, save_class_dir))

    print(f"[*] 총 {len(tasks)}개의 작업 발견. {CONFIG['NUM_WORKERS']}개의 프로세스로 처리 시작...")

    # 멀티프로세싱 실행
    success_count = 0
    with ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        # 진행률 표시 (tqdm)
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks), unit="file"))
        
        success_count = sum(results)

    print(f"\n[완료] 총 {len(tasks)}개 중 {success_count}개 처리 성공.")
    print(f"저장 경로: {CONFIG['SAVE_ROOT_DIR']}")

if __name__ == "__main__":
    # Windows에서 multiprocessing 사용 시 필수
    multiprocessing.freeze_support()
    main()