import os
import json
from pathlib import Path
import numpy as np
from skimage import measure
from pycocotools import mask as coco_mask

# ================= 사용자 경로 설정 =================
# 1. JSON 라벨이 있는 폴더
json_folder = Path(r"C:\Users\hgy84\Desktop\0520\label_Test\Scissors-A\json_labels")

# 2. 이미지가 있는 폴더 (참고용)
image_folder = Path(r"C:\Users\hgy84\Desktop\0520\label_Test\Scissors-A\images")

# 3. 결과(.txt)를 저장할 폴더
output_folder = Path(r"C:\Users\hgy84\Desktop\0520\label_Test\Scissors-A\labels_skimage")
output_folder.mkdir(parents=True, exist_ok=True)

# 클래스 ID 매핑 (필요시 수정)
CLASS_MAPPING = {} 
# ===================================================

def rle_to_polygon_skimage(rle, width, height):
    """skimage를 이용한 폴리곤 변환 (부드러운 곡선, 정밀함)"""
    binary_mask = coco_mask.decode(rle)
    
    # 0.5 레벨에서 등고선 추출 (Marching Squares)
    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []
    
    for contour in contours:
        # skimage는 (row, col) = (y, x) 순서이므로 (x, y)로 뒤집기
        contour = np.flip(contour, axis=1)
        
        normalized_poly = []
        for point in contour:
            x, y = point[0], point[1]
            x_norm = min(max(x / width, 0.0), 1.0)
            y_norm = min(max(y / height, 0.0), 1.0)
            normalized_poly.extend([round(x_norm, 6), round(y_norm, 6)])
            
        if len(normalized_poly) >= 6:
            polygons.append(normalized_poly)
            
    return polygons

print(f"🚀 [skimage] 변환 시작...")
count = 0

for json_file in json_folder.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        img_info_lookup = {img["id"]: img for img in data["images"]}
        label_lines = {}

        for ann in data["annotations"]:
            img_info = img_info_lookup.get(ann["image_id"])
            if not img_info: continue
            
            file_name = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]
            cid = CLASS_MAPPING.get(ann["category_id"], ann["category_id"])
            
            rle = ann.get("segmentation")
            if not rle: continue

            if isinstance(rle, dict) and 'counts' in rle:
                polygons = rle_to_polygon_skimage(rle, w, h)
                for poly in polygons:
                    line = f"{cid} " + " ".join(map(str, poly))
                    label_lines.setdefault(file_name, []).append(line)

        for fname, lines in label_lines.items():
            txt_path = output_folder / (Path(fname).stem + ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            count += 1

    except Exception as e:
        print(f"⚠️ 오류 ({json_file.name}): {e}")

print(f"✅ [skimage] 완료! 총 {count}개 파일 생성됨: {output_folder}")