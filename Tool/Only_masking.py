from ultralytics import YOLO
import cv2
import os

# ────────── 설정 ──────────
model_path  = "/home/hgyeo/Desktop/runs/segment/train65/weights/best.pt"   # 사용할 .pt
input_root  = '/home/hgyeo/Desktop/sample 1'                                # 이미지가 바로 들어있는 폴더
output_root = "/home/hgyeo/Desktop"                                       # 결과 루트

model_name  = os.path.splitext(os.path.basename(model_path))[0]
save_dir    = os.path.join(output_root, model_name)
os.makedirs(save_dir, exist_ok=True)

# 모델 로드
model = YOLO(model_path)
print(f"\n🚀 모델 실행: {model_name}")
print(f"▶ 입력 폴더: {input_root}")
print(f"▶ 저장 폴더: {save_dir}")

# ────────── 이미지 순회 ──────────
valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
image_files = [f for f in os.listdir(input_root) if f.lower().endswith(valid_exts)]
image_files.sort()

if not image_files:
    print("⚠️ 처리할 이미지가 없습니다.")
else:
    for i, image_name in enumerate(image_files, 1):
        image_path = os.path.join(input_root, image_name)

        # 추론 (GPU 0번 사용, CPU 쓰려면 device='cpu')
        results = model.predict(image_path, verbose=False, device=0)
        result  = results[0]

        # 결과 이미지 저장
        save_path    = os.path.join(save_dir, image_name)
        # 결과 이미지 저장 (마스크만 표시)
        result_image = result.plot(
            boxes=False,   # 박스 끄기
            labels=False,  # 클래스명/ID 끄기
            conf=False     # 점수 끄기
            # masks=True   # 기본값이 True라 생략 가능 (세그모델이면 마스크만 남음)
        )
        cv2.imwrite(save_path, result_image)

        # 탐지 여부 출력
        has_boxes = (result.boxes is not None) and (len(result.boxes) > 0)
        status = "✅ 저장 완료" if has_boxes else "❌ 탐지 없음"
        print(f"[{i:04d}/{len(image_files):04d}] {status}: {save_path}")

print("\n🎯 단일 모델 예측 완료 (GPU)")
