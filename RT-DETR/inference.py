import torchvision.transforms as T
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import json
from tqdm import tqdm

# 경로 문제 해결용
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.core.yaml_config import YAMLConfig

# 1. 설정 파일 경로
CONFIG_FILE = "/home/hgyeo/Desktop/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/RT-DETRv2-X.yml" 

# 2. 학습된 가중치 파일 경로
CHECKPOINT_FILE = "/home/hgyeo/Desktop/RT-DETR/rtdetrv2_pytorch/output/train_r101_x_25epoch/best.pth"

# 3. [중요] 클래스 이름이 들어있는 JSON 파일 (학습때 쓴 거)
LABEL_JSON_FILE = "/home/hgyeo/Desktop/1224_aug_merge/train/json_labels/train_annotations.json"

# 4. 테스트할 이미지 폴더
TEST_IMAGE_DIR = "/home/hgyeo/Desktop/test_data/CUBOX/images"

# 4. 결과 이미지를 저장할 폴더 이름
OUTPUT_DIR = "/home/hgyeo/Desktop/RT-DETR/test_output1"

# 5. 검출 임계값 (0.4 이상인 것만 표시)
SCORE_THRESHOLD = 0.1
# =========================================================
def load_class_names(json_path):
    """JSON 파일에서 클래스 이름 목록을 가져옵니다."""
    print(f"📖 클래스 이름 로딩 중: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # categories 리스트 가져오기
        categories = data.get('categories', [])
        
        # ID 순서대로 정렬 (매우 중요)
        categories.sort(key=lambda x: x['id'])
        
        # { 0: "person", 1: "car", ... } 형태로 변환
        # 모델은 0부터 시작하는 인덱스를 뱉으므로 리스트 순서대로 매핑합니다.
        id_to_name = {i: cat['name'] for i, cat in enumerate(categories)}
        
        print(f"✅ 총 {len(id_to_name)}개의 클래스 이름을 찾았습니다.")
        return id_to_name
    except Exception as e:
        print(f"⚠️ 클래스 이름을 불러오는 데 실패했습니다: {e}")
        return None

def draw_bboxes(pil_img, boxes, labels, scores, class_names=None, thr=0.5):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size
    
    # 폰트 설정 (시스템에 따라 다를 수 있음, 없으면 기본)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < thr:
            continue
            
        # 좌표 변환
        cx, cy, bw, bh = box * torch.tensor([w, h, w, h])
        x1 = cx - 0.5 * bw
        y1 = cy - 0.5 * bh
        x2 = cx + 0.5 * bw
        y2 = cy + 0.5 * bh
        
        # 라벨 이름 가져오기
        label_id = int(label)
        if class_names and label_id in class_names:
            label_text = class_names[label_id]
        else:
            label_text = f"ID:{label_id}"
            
        # 그리기
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 텍스트 내용
        text = f"{label_text} ({score:.2f})"
        
        # 텍스트 배경 (가독성)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill='white', font=font)
        
    return pil_img

def main():
    # 0. 준비
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"❌ 폴더 없음: {TEST_IMAGE_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 클래스 이름 로드
    class_names = load_class_names(LABEL_JSON_FILE)

    # 2. 모델 로드
    cfg = YAMLConfig(CONFIG_FILE, resume=CHECKPOINT_FILE)
    model = cfg.model
    
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
        else:
            state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
    else:
        print("❌ 가중치 파일 없음")
        return

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 3. 이미지 전처리
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # 4. 이미지 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']:
        image_files.extend(glob.glob(os.path.join(TEST_IMAGE_DIR, ext)))
    image_files = sorted(list(set(image_files)))
    
    print(f"📸 {len(image_files)}장 처리 시작...")

    # 5. 추론
    for img_path in tqdm(image_files):
        try:
            filename = os.path.basename(img_path)
            
            # 이미지 열기
            original_image = Image.open(img_path).convert("RGB")
            img_tensor = transform(original_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)

            pred_logits = output['pred_logits'][0]
            pred_boxes = output['pred_boxes'][0]

            probas = pred_logits.sigmoid()
            scores, labels = probas.max(-1)
            
            keep = scores > SCORE_THRESHOLD
            boxes = pred_boxes[keep].cpu()
            labels = labels[keep].cpu()
            scores = scores[keep].cpu()

            # 그리기 (class_names 전달)
            result_img = draw_bboxes(original_image, boxes, labels, scores, 
                                   class_names=class_names, thr=SCORE_THRESHOLD)

            save_path = os.path.join(OUTPUT_DIR, filename)
            result_img.save(save_path)

        except Exception as e:
            print(f"Error: {e}")
            continue

    print(f"\n✨ 완료! 결과 폴더: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == '__main__':
    main()