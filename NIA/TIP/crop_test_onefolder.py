import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

# ==========================================
# 1. 경로 및 설정
# ==========================================
CLASS_NAME = "Adaptor"

RAW_DIR = rf"D:\hgyeo\BCAS_TIP\bare_image_raw\{CLASS_NAME}\raws"
LABEL_DIR = rf"D:\hgyeo\BCAS_TIP\bare_image_png\{CLASS_NAME}\labels"
SAVE_DIR = rf"D:\hgyeo\BCAS_TIP\bare_image_raw_crop\{CLASS_NAME}"

RAW_WIDTH = 896
RAW_DTYPE = np.uint16
PADDING_PIXELS = 1

# ★ 높이 정규화 설정 (기존 코드 설정값)
TARGET_H = 760
AVG_PIXEL_ROWS = 20

# ==========================================
# 2. 이미지 높이 정규화 함수 (핵심 추가)
# ==========================================
def normalize_height_uint16(img, target_h=760):
    """
    이미지 높이를 target_h(760)로 맞춤.
    - H < 760: 하단 20줄 평균 색상으로 아래쪽 패딩
    - H > 760: 아래쪽 잘라냄 (Crop)
    - H == 760: 유지
    """
    h, w = img.shape
    
    if h == target_h:
        return img
    
    # 1. 높이가 작은 경우 (패딩)
    elif h < target_h:
        pad_h = target_h - h
        
        # 하단 n줄 평균 계산 (uint16이므로 int로 변환)
        rows_to_avg = min(h, AVG_PIXEL_ROWS)
        bottom_part = img[-rows_to_avg:, :]
        avg_val = int(np.mean(bottom_part)) # 단일 채널이므로 바로 mean
        
        # 패딩 적용 (BORDER_CONSTANT)
        # value 인자는 스칼라 혹은 [v, v, v, v] 형태
        padded_img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, 0, 
            borderType=cv2.BORDER_CONSTANT, 
            value=avg_val
        )
        return padded_img

    # 2. 높이가 큰 경우 (크롭)
    else: # h > target_h
        # 위에서부터 target_h 만큼만 남김
        cropped_img = img[:target_h, :]
        return cropped_img

# ==========================================
# 3. 폴리곤 마스크 및 BBox 계산
# ==========================================
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
            x_px = int(coords[i] * img_w)
            y_px = int(coords[i+1] * img_h)
            pts.append([x_px, y_px])
        
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_np], 255)

        if padding > 0:
            kernel = np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        x, y, w, h = cv2.boundingRect(pts_np)
        
        x = max(0, x)
        y = max(0, y)
        w = min(img_w - x, w)
        h = min(img_h - y, h)

        return mask, (x, y, w, h), cls_id, coords

    except Exception as e:
        # print(f"Label Parsing Error ({label_path}): {e}")
        return None

# ==========================================
# 4. 마스킹 크롭 및 라벨 재계산
# ==========================================
def crop_with_mask(image, mask, bbox):
    x, y, w, h = bbox
    masked_img = image.copy()
    masked_img[mask == 0] = 65535  # 배경 흰색 처리
    crop_img = masked_img[y:y+h, x:x+w]
    return crop_img

def save_renormalized_label(save_path, cls_id, coords, offset_x, offset_y, crop_w, crop_h, org_w, org_h):
    new_coords = []
    for i in range(0, len(coords), 2):
        org_px = coords[i] * org_w
        org_py = coords[i+1] * org_h
        
        curr_px = org_px - offset_x
        curr_py = org_py - offset_y
        
        new_nx = curr_px / crop_w
        new_ny = curr_py / crop_h
        
        new_nx = max(0.0, min(1.0, new_nx))
        new_ny = max(0.0, min(1.0, new_ny))
        
        new_coords.extend([new_nx, new_ny])
        
    with open(save_path, 'w') as f:
        coord_str = " ".join([f"{val:.6f}" for val in new_coords])
        f.write(f"{cls_id} {coord_str}\n")

# ==========================================
# 5. 메인 로직
# ==========================================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    
    if not label_files:
        print(f"❌ 라벨 없음: {LABEL_DIR}")
        return

    print(f"[*] '{CLASS_NAME}' 760px 정규화 후 폴리곤 크롭 시작... ({len(label_files)}개)")
    success_count = 0

    for i, label_path in enumerate(tqdm(label_files)):
        try:
            filename = os.path.basename(label_path)
            file_id = os.path.splitext(filename)[0]

            raw_path = os.path.join(RAW_DIR, f"{file_id}.raw")
            if not os.path.exists(raw_path): continue

            # 1. RAW 로드
            raw_data = np.fromfile(raw_path, dtype=RAW_DTYPE)
            if raw_data.size % RAW_WIDTH != 0: continue
            
            full_img = raw_data.reshape(-1, RAW_WIDTH)
            total_h = full_img.shape[0]
            mid_h = total_h // 2
            
            # 2. HE/LE 분리
            img_top = full_img[:mid_h, :]
            img_bot = full_img[mid_h:, :]

            if img_top.mean() > img_bot.mean():
                img_th, img_tl = img_top, img_bot
            else:
                img_th, img_tl = img_bot, img_top

            # ★★★ 3. [핵심] 760px로 높이 맞추기 (라벨과의 싱크를 위해) ★★★
            # 라벨 파일은 이미 760px로 변환된 PNG 기준으로 만들어졌으므로,
            # RAW 이미지도 760px로 변환한 뒤에 좌표를 적용해야 함.
            img_th_760 = normalize_height_uint16(img_th, TARGET_H)
            img_tl_760 = normalize_height_uint16(img_tl, TARGET_H)

            # 4. 마스크 및 BBox 계산
            # 이제 기준 높이는 mid_h가 아니라 TARGET_H(760) 입니다.
            poly_info = get_polygon_mask_and_bbox(label_path, RAW_WIDTH, TARGET_H, padding=PADDING_PIXELS)
            if poly_info is None: continue
            
            mask, bbox, cls_id, coords = poly_info
            x, y, w, h = bbox
            
            if w <= 0 or h <= 0: continue

            # 5. 마스킹 + 크롭 실행
            # 760px로 변환된 이미지에서 잘라냅니다.
            crop_th = crop_with_mask(img_th_760, mask, bbox)
            crop_tl = crop_with_mask(img_tl_760, mask, bbox)

            # 6. 저장
            final_h, final_w = crop_th.shape
            
            th_filename = f"{CLASS_NAME}_{i:04d}_TH_{final_w}x{final_h}.raw"
            tl_filename = f"{CLASS_NAME}_{i:04d}_TL_{final_w}x{final_h}.raw"
            
            crop_th.tofile(os.path.join(SAVE_DIR, th_filename))
            crop_tl.tofile(os.path.join(SAVE_DIR, tl_filename))

            # 7. 라벨 저장 (재계산)
            # 원본 크기 인자에 TARGET_H(760)를 넘겨야 정확하게 계산됩니다.
            txt_filename = th_filename.replace('.raw', '.txt')
            save_renormalized_label(
                os.path.join(SAVE_DIR, txt_filename),
                cls_id, coords,
                x, y, final_w, final_h,
                RAW_WIDTH, TARGET_H 
            )

            success_count += 1

        except Exception as e:
            # print(f"Error {file_id}: {e}")
            pass

    print(f"\n[완료] {success_count}개 생성 완료.")
    print(f"저장 경로: {SAVE_DIR}")

if __name__ == "__main__":
    main()