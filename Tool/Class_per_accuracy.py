from ultralytics import YOLO

# 1. 모델 로드
model = YOLO('/home/hgyeo/Desktop/runs/segment/train111/weights/best.pt')

# 2. 검증 실행 (표준 지표 계산)
metrics = model.val(
    data='/home/hgyeo/Desktop/yaml/1208.yaml',  # 데이터셋 설정 파일
    split='val',        # 검증 데이터셋(valid) 사용
    device='cuda:1',    # 사용하실 GPU 번호
    batch=8,           # 메모리 넉넉하면 32, 64로 늘려도 됨
    imgsz=640,          # 학습할 때 쓴 이미지 크기
    conf=0.001,         # mAP 계산용은 보통 conf를 낮게 잡습니다 (기본값 사용 추천)
    verbose=True,       # ★ 이게 True여야 표가 출력됩니다
    project='val_results', # 결과 저장할 메인 폴더명
    name='result_table'    # 결과 저장할 서브 폴더명
)

# 3. (선택) 지표값 따로 확인하고 싶으면 아래 변수에 들어있음
# print(f"Box mAP50: {metrics.box.map50}")
# print(f"Mask mAP50: {metrics.seg.map50}")