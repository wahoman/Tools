#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
images → images_pred (폴더별 추론 + JSON)
- 세그 모델이더라도 '마스크는 전부 비활성'하고 'BBox만' 출력/저장
- 리콜(잘 찾기) 위주 기본값
- 결과 폴더 중복 시 images_pred1, images_pred2... 자동 생성
"""

from ultralytics import YOLO
import cv2, os, json, yaml, sys, torch
import numpy as np
from pathlib import Path

# ─── ① 사용자 설정 ───────────────────────────────────
MODEL_PATH = Path("/home/hgyeo/Desktop/runs/segment/train111/weights/best.pt")
YAML_PATH  = Path("/home/hgyeo/Desktop/yaml/1208.yaml")
IMAGES_DIR = Path("/home/hgyeo/Desktop/CUBOX/")
DEVICE     = 1

# ─── ② 하이퍼파라미터 ────────────────────────────────
IMGSZ    = 640
CONF_THR = 0.10
IOU_THR  = 0.70
MAX_DET  = 50
RETINA   = False

# ─── ③ 출력 경로 (자동 넘버링 추가) ───────────────────
# 기본 이름 설정
base_name = "images_pred"
PRED_ROOT = IMAGES_DIR.parent / base_name

# 폴더가 이미 존재하면 숫자를 붙여서 새 이름 찾기
if PRED_ROOT.exists():
    counter = 1
    while True:
        new_path = IMAGES_DIR.parent / f"{base_name}{counter}"
        if not new_path.exists():
            PRED_ROOT = new_path
            break
        counter += 1

# 최종 결정된 경로 생성
PRED_ROOT.mkdir(parents=True, exist_ok=True)
print(f"📂 결과 저장 경로: {PRED_ROOT}")

# ─── ④ 클래스 로드 ───────────────────────────────────
with open(YAML_PATH, encoding="utf-8") as f:
    names_raw = yaml.safe_load(f).get("names", [])
CLS_NAMES = {i: (n if (n and str(n).strip()) else f"cls_{i}") for i, n in enumerate(names_raw)}

# ─── ⑤ 모델 로드 ─────────────────────────────────────
model = YOLO(str(MODEL_PATH))
USE_FP16 = torch.cuda.is_available() and isinstance(DEVICE, int)

def to_list(arr):
    return arr.cpu().tolist() if hasattr(arr, "cpu") else arr.tolist()

# ─── ⑥ 하위 폴더 순회 ────────────────────────────────
if not IMAGES_DIR.exists():
    print(f"❌ 경로 없음: {IMAGES_DIR}")
    sys.exit(1)

id_dirs = sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir())
if not id_dirs:
    print("❌ images 하위에 ID 폴더가 없습니다.")
    sys.exit(1)

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for id_dir in id_dirs:
    vis_dir   = PRED_ROOT / id_dir.name
    label_dir = PRED_ROOT / f"{id_dir.name}_labels"
    vis_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in id_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    if not imgs:
        print(f"⚠ {id_dir.name}: 이미지 없음 (skip)")
        continue

    print(f"\n── {id_dir.name} ({len(imgs)}장) ──")
    for img_path in imgs:

        # YOLO 추론
        res = model.predict(
            str(img_path), device=DEVICE, imgsz=IMGSZ,
            conf=CONF_THR, iou=IOU_THR, max_det=MAX_DET,
            retina_masks=False, half=USE_FP16,
            verbose=False
        )[0]

        h, w = res.orig_shape

        # ───────────────────────────────────────────────
        # ⭐ ⑦ 시각화 이미지(BBox) 생성
        # ───────────────────────────────────────────────
        vis_img = res.plot(boxes=True, masks=False)

        # ⭐ ⑧ 흰색 패딩 100px 추가
        pad = 100
        vis_img = cv2.copyMakeBorder(
            vis_img,
            pad, pad, pad, pad,                # 상 하 좌 우
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)              # 흰색 패딩
        )
        # ───────────────────────────────────────────────

        cv2.imwrite(str(vis_dir / img_path.name), vis_img)

        # JSON 저장 (원본 그대로)
        meta = {
            "image": img_path.name,
            "size": {"h": int(h), "w": int(w)},
            "predictions": []
        }

        num_boxes = len(res.boxes) if res.boxes is not None else 0
        for i in range(num_boxes):
            cid = int(res.boxes.cls[i])
            pred = {
                "class_id": cid,
                "class_name": CLS_NAMES.get(cid, f"cls_{cid}"),
                "confidence": round(float(res.boxes.conf[i]), 4),
                "bbox": to_list(res.boxes.xyxy[i])
            }
            meta["predictions"].append(pred)

        with open(label_dir / f"{img_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"{'❌' if num_boxes == 0 else '✅'} {img_path.name}  (det={num_boxes})")

print(f"\n🎯 모든 폴더 처리 완료! ({PRED_ROOT})")