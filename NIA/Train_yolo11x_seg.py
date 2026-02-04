from ultralytics import YOLO
import torch

def train_model():
    # 모델 로드
    model = YOLO('/home/hgyeo/Desktop/yolo11x-seg.pt')

    # 학습 시작
    model.train(
        data='/home/hgyeo/Desktop/yaml/1208.yaml',
        
        # === [기본 설정] ===
        device=0,
        epochs=100,             # X-ray는 길게 학습
        patience=15,   
        batch=16,               # OOM 나면 줄이세요
        imgsz=896,              
        
        # === [최적화] ===
        optimizer='auto',       
        lr0=0.01,               
        lrf=0.01,               
        cos_lr=True,            
        
        # === [X-ray 전용 Augmentation (필수)] ===
        # 1. 색상 (Color) - 재질 정보 보존
        hsv_h=0.0,              # 🚫 색조 변경 금지
        hsv_s=0.2,              
        hsv_v=0.3,              
        
        # 2. 기하학 (Geometry) - 다양한 배치 학습
        degrees=25.0,           
        flipud=0.5,             # 상하 반전
        fliplr=0.5,             # 좌우 반전
        scale=0.5,              
        shear=2.0,              
        
        # 3. 겹침 해결
        mosaic=1.0,             
        mixup=0.15,             
        copy_paste=0.3,         
        
        # === [시스템] ===
        workers=16,             
        plots=True,
        val=True                
    )

if __name__ == '__main__':
    train_model()