import json
import os
import yaml
from glob import glob
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# ================= 설정 부분 =================
BASE_DIR = '/home/hgyeo/Desktop/1224_aug_merge'
YAML_PATH = '/home/hgyeo/Desktop/yaml/1208.yaml'

# 사용할 코어 개수 설정
NUM_CORES = 8

FIXED_SIZE_MODE = False 
# ===========================================

SETS = [
    {'name': 'train', 'img_dir': f'{BASE_DIR}/train/images', 'label_dir': f'{BASE_DIR}/train/labels', 'out_dir': f'{BASE_DIR}/train/json_labels'},
    {'name': 'valid', 'img_dir': f'{BASE_DIR}/valid/images', 'label_dir': f'{BASE_DIR}/valid/labels', 'out_dir': f'{BASE_DIR}/valid/json_labels'}
]

def load_classes_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if isinstance(names, dict):
        max_idx = max(names.keys())
        return [names.get(i, f"class_{i}") for i in range(max_idx + 1)]
    elif isinstance(names, list):
        cleaned = []
        for i, n in enumerate(names):
            if n is None or str(n).strip() == "":
                cleaned.append(f"class_{i}")
            else:
                cleaned.append(str(n).split('#')[0].strip())
        return cleaned
    return []

def process_single_image(args):
    """ 워커 함수: 이미지 1장에 대해 크기 확인 및 라벨 변환 수행 """
    img_path, label_dir, img_id_offset = args
    filename = os.path.basename(img_path)
    
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception as e:
        return None

    img_info = {
        "id": img_id_offset,
        "file_name": filename,
        "height": h,
        "width": w
    }

    anns = []
    label_file = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(label_dir, label_file)

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 5: continue
            
            class_id = int(parts[0])
            coords = parts[1:] 
            
            x_vals = [c * w for c in coords[0::2]]
            y_vals = [c * h for c in coords[1::2]]
            
            min_x = min(x_vals)
            max_x = max(x_vals)
            min_y = min(y_vals)
            max_y = max(y_vals)
            
            abs_w = max_x - min_x
            abs_h = max_y - min_y
            
            if abs_w < 1 or abs_h < 1: continue

            anns.append({
                "image_id": img_id_offset,
                "category_id": class_id, # [수정됨] +1 제거하여 0-base 유지
                "bbox": [min_x, min_y, abs_w, abs_h],
                "area": abs_w * abs_h,
                "iscrowd": 0,
                "segmentation": [] 
            })
            
    return (img_info, anns)

def convert_dataset_parallel(set_info, class_names):
    img_dir = set_info['img_dir']
    label_dir = set_info['label_dir']
    out_dir = set_info['out_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Scanning files in {set_info['name']}...")
    image_files = glob(os.path.join(img_dir, '*'))
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in valid_ext]
    
    tasks = []
    for i, f in enumerate(image_files):
        tasks.append((f, label_dir, i + 1))
    
    coco_images = []
    coco_annotations = []
    
    print(f"Processing {len(tasks)} images with {NUM_CORES} cores...")
    
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
    
    print("Aggregating results...")
    ann_id_counter = 1
    for res in results:
        if res is None: continue
        img_info, anns = res
        coco_images.append(img_info)
        
        for ann in anns:
            ann['id'] = ann_id_counter
            coco_annotations.append(ann)
            ann_id_counter += 1

    # [수정됨] categories id 생성 시 i+1 -> i 로 변경하여 0-base 유지
    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": n, "supercategory": "object"} for i, n in enumerate(class_names)]
    }
    
    json_path = os.path.join(out_dir, f"{set_info['name']}_annotations.json")
    with open(json_path, 'w') as f:
        json.dump(coco_data, f)
    print(f"Done! Saved to {json_path}")

if __name__ == "__main__":
    classes = load_classes_from_yaml(YAML_PATH)
    print(f"Loaded {len(classes)} classes.")
    
    for set_info in SETS:
        convert_dataset_parallel(set_info, classes)