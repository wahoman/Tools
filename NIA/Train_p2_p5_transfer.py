from ultralytics import YOLO
import torch

def train_model():
    # =========================================================
    # 1. [수정] 모델 초기화 단계 변경
    # =========================================================
    
    # 1단계: 커스텀 구조(P2-P5)로 모델 뼈대 생성
    # (이 단계에서는 가중치가 랜덤하게 초기화된 상태입니다)
    model = YOLO('/home/hgyeo/Desktop/yolo11-seg-p2-p5.yaml')

    # 2단계: 사전 학습된 가중치(yolo11x-seg.pt) 로드
    # (구조가 다른 부분은 제외하고, 맞는 부분만 가져와서 입힙니다)
    model.load('/home/hgyeo/Desktop/yolo11x-seg.pt')

    # =========================================================
    # 2. 학습 시작 (cfg 인자 삭제)
    # =========================================================
    model.train(
        # ❌ 여기 있던 cfg='/home/.../yolo11-seg-p2-p5.yaml' 삭제!
        # 대신 data, epochs 등 설정값은 그대로 둡니다.
        
        data='/home/hgyeo/Desktop/yaml/p2.yaml',
        
        # === [저장 경로] ===
        project='runs/segment',
        name='train_p2_p5',
        exist_ok=False,
        
        # === [시스템 설정] ===
        device=1,
        epochs=200,
        patience=30,
        
        # === [메모리 주의] ===
        # P2 레이어 + 896 해상도 + X모델은 메모리를 많이 먹습니다.
        # OOM 발생 시 4 -> 2 로 줄이세요.
        batch=2,    
        imgsz=640,
        retina_masks=True,
        
        # === [최적화 - P2 안정화] ===
        optimizer='auto',
        lr0=0.005,          # 초기 학습률 낮춤
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        
        # === [X-ray 전용 Augmentation] ===
        hsv_h=0.0,
        hsv_s=0.2,
        hsv_v=0.3,
        
        degrees=25.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        shear=2.0,
        
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        
        workers=8,
        plots=True,
        val=True
    )

if __name__ == '__main__':
    train_model()