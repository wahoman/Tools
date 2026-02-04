import os
import numpy as np
import cv2
import yaml
import json
from glob import glob

# =========================================================
# 1. 경로 설정
# =========================================================
mapping_json_path = "D:/hgyeo/TIP/class_mapping.json" 
yaml_path = "D:/hgyeo/TIP/data.yaml"

raw_root_path = "D:/hgyeo/TIP/Origin_bare"
final_save_path = "D:/hgyeo/TIP/78Classified_polygon_raw"
# =========================================================

def load_yaml_mapping(yaml_file):
    print(f"🔄 YAML 파일 로드 중: {yaml_file}")
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    mapping = {}
    if isinstance(names, list):
        for idx, name in enumerate(names):
            mapping[name] = f"{idx}_{name}"
    elif isinstance(names, dict):
        for idx, name in names.items():
            mapping[name] = f"{idx}_{name}"
    return mapping

def get_class_mapping_from_json(json_path):
    print(f"🔄 매핑 파일(JSON) 로드 중: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ [오류] 매핑 파일을 찾을 수 없습니다: {json_path}")
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    print(f"✅ 매핑 로드 완료: 총 {len(mapping)}개의 정보를 불러왔습니다.")
    return mapping

def Raw_Loading(file_path):
    raw_array = np.fromfile(file_path, dtype='uint16')
    if raw_array.size == 0 or raw_array.size % 640 != 0:
        raise ValueError(f"Invalid raw size: {raw_array.size}")
    return raw_array.reshape((-1, 640))

# ★ 수정된 함수: 이미지만 자르는 게 아니라, 잘린 정보(offset, size)도 반환
def Polygon_Crop_Info(txt_label_path, raw_img, padding=0):
    height, width = raw_img.shape
    
    # 1. 라벨 읽기
    with open(txt_label_path, 'r') as f:
        lines = f.readlines()
    if not lines: raise ValueError("Empty label file")

    parts = lines[0].strip().split()
    cls_id = parts[0]
    coords = list(map(float, parts[1:]))
    
    # 2. 폴리곤 마스크 생성
    pts = []
    for i in range(0, len(coords), 2):
        x_px = int(coords[i] * width)
        y_px = int(coords[i+1] * height)
        pts.append([x_px, y_px])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    if padding != 0:
        kernel_size = abs(padding) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if padding < 0: mask = cv2.erode(mask, kernel, iterations=1)
        else: mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 3. Bounding Box 계산 (잘라낼 범위)
    x, y, w, h = cv2.boundingRect(pts)
    x, y = max(0, x), max(0, y)
    w, h = min(width - x, w), min(height - y, h)

    # 4. 이미지 마스킹 및 크롭
    masked_img = raw_img.copy()
    masked_img[mask == 0] = 65535 
    crop_img = masked_img[y:y+h, x:x+w]

    # ★ 반환값: 크롭된 이미지 + 좌표 정보(x, y, w, h) + 원본 좌표 리스트
    return crop_img, x, y, w, h, cls_id, coords

# ★ 신규 함수: 잘린 크기에 맞춰 라벨 좌표 재계산 (Renormalization)
def save_renormalized_label(save_path, cls_id, coords, offset_x, offset_y, crop_w, crop_h, org_w, org_h):
    new_coords = []
    
    # 좌표 순회 (x, y 쌍)
    for i in range(0, len(coords), 2):
        # 1. 원본 픽셀 좌표로 복구
        org_px = coords[i] * org_w
        org_py = coords[i+1] * org_h
        
        # 2. 크롭된 만큼 이동 (Translation)
        crop_px = org_px - offset_x
        crop_py = org_py - offset_y
        
        # 3. 크롭된 이미지 크기 기준으로 정규화 (0~1)
        new_nx = crop_px / crop_w
        new_ny = crop_py / crop_h
        
        # 범위 안전장치 (0~1 사이로 클램핑)
        new_nx = max(0.0, min(1.0, new_nx))
        new_ny = max(0.0, min(1.0, new_ny))
        
        new_coords.extend([new_nx, new_ny])
        
    # 파일 저장
    with open(save_path, 'w') as f:
        line = f"{cls_id} " + " ".join([f"{val:.6f}" for val in new_coords]) + "\n"
        f.write(line)

def main():
    yaml_id_map = load_yaml_mapping(yaml_path)
    current_padding = 1
    prefix_map = get_class_mapping_from_json(mapping_json_path)
    
    if len(prefix_map) == 0: return

    for i in range(1, 55): 
        round_str = f"Round{i}_Bare"
        raw_folder = os.path.join(raw_root_path, round_str)
        label_folder = os.path.join(raw_root_path, f"{round_str}_polygon")

        if not os.path.exists(raw_folder): continue
        if not os.path.exists(label_folder): continue

        print(f"🚀 처리 중: {round_str} ...")
        raw_files = [f for f in os.listdir(raw_folder) if f.endswith(".raw")]
        
        for filename in raw_files:
            base = filename[:-4]
            current_prefix = "_".join(base.split("_")[:2])

            if current_prefix in prefix_map:
                raw_class_name = prefix_map[current_prefix]
                if raw_class_name in yaml_id_map:
                    final_folder_name = yaml_id_map[raw_class_name]
                else:
                    final_folder_name = f"Unknown_{raw_class_name}"
                save_dir = os.path.join(final_save_path, final_folder_name)
            else:
                save_dir = os.path.join(final_save_path, "Unclassified")

            os.makedirs(save_dir, exist_ok=True)

            label_matches = [f for f in os.listdir(label_folder) if f.endswith('.txt') and base in f]
            if not label_matches: continue
            
            label_path = os.path.join(label_folder, label_matches[0])
            raw_path = os.path.join(raw_folder, filename)

            try:
                # 원본 로딩
                img = Raw_Loading(raw_path)
                org_h, org_w = img.shape
                vd = org_h // 2
                le_img = img[:vd, :]
                he_img = img[vd:, :]

                # 1. LE 이미지 크롭 및 정보 추출
                le_crop, lx, ly, lw, lh, cls_id, coords = Polygon_Crop_Info(label_path, le_img, padding=current_padding)
                
                # 2. HE 이미지 크롭 (LE와 같은 좌표 사용해야 함)
                # HE는 마스킹만 다시 하고 좌표(lx, ly, lw, lh)는 LE 것을 그대로 써야 정합이 맞음
                # 편의상 같은 함수 쓰되, 좌표는 이미 구했으니 이미지만 씀
                he_crop, _, _, _, _, _, _ = Polygon_Crop_Info(label_path, he_img, padding=current_padding)
                
                shape_str = f"{lw}x{lh}"

                # 3. 이미지 저장
                le_crop.tofile(os.path.join(save_dir, f"{base}_TL_{shape_str}.raw"))
                he_crop.tofile(os.path.join(save_dir, f"{base}_TH_{shape_str}.raw"))
                
                # ★★★ 4. [핵심] 재계산된 라벨 저장 ★★★
                # 원본 라벨을 그대로 복사하는 게 아니라, 크롭된 크기(lw, lh)에 맞춰 0~1로 늘려줍니다.
                new_label_path = os.path.join(save_dir, f"{base}.txt")
                save_renormalized_label(new_label_path, cls_id, coords, lx, ly, lw, lh, org_w, vd)
                
            except ValueError: pass 
            except Exception as e:
                print(f"❌ 에러({base}): {e}")

    print("\n🎉 모든 작업 완료! (라벨 재계산 적용됨)")

if __name__ == "__main__":
    main()