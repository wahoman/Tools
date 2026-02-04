#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validation_accuracy_v3_fast_final_exclude_electronic.py
- ✅ 배치 추론 & 멀티쓰레드 저장
- ✅ [Groupping 1] Gun 그룹핑 (Pistol, Revolvers, Plastic Pistol -> Gun_Group)
- ✅ [Groupping 2] Electronic 그룹핑 (Smart phone, Tablet pc, Laptop, Electronic device -> Electronic_Group)
- ✅ [Exclusion] Shovel 및 Hex key 분석 제외 적용
- ✅ Device: cuda:1 설정 완료
"""

from ultralytics import YOLO
from pathlib import Path
import cv2, csv, yaml
from typing import List
from concurrent.futures import ThreadPoolExecutor

# =============================
# CONFIG
# =============================
MODEL_PATH = "/home/hgyeo/Desktop/runs/segment/train111/weights/best.pt"
VALID_ROOT = "/home/hgyeo/Desktop/1114/valid"
GT_LABEL_ROOT = "/home/hgyeo/Desktop/1114_merge/valid/labels"
DATA_YAML    = "/home/hgyeo/Desktop/yaml/1208.yaml"
RESULT_DIR = "results"
OUT_CSV    = f"{RESULT_DIR}/classwise_result_grouped.csv"

# 🔥 Device 설정 (cuda:1)
DEVICE     = "cuda:1"

CONF_THRES = 0.20
IMGSZ      = 640
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 🔥 성능 튜닝 옵션
BATCH_SIZE = 32
NUM_WORKERS = 16

# =============================
# [NEW] 그룹핑 및 제외 헬퍼 함수
# =============================
def is_excluded(name: str) -> bool:
    """
    분석에서 아예 제외할 클래스인지 확인합니다.
    - Shovel 포함 시 제외
    - Hex key 포함 시 제외
    """
    if "Shovel" in name:
        return True
    if "Hex key" in name:
        return True
    return False

def get_group_name(cid: int, name: str) -> str:
    """
    클래스 이름을 받아 그룹명을 반환합니다.
    """
    # 1. [Gun 계열] Pistol, Revolvers 등도 Gun_Group으로 강제 통합
    if name in ["Pistol", "Plastic Pistol", "Revolvers"]:
        return "Gun_Group"

    # 2. [Electronic 계열] 요청사항 반영: 스마트폰, 노트북 등 통합
    if name in ["Smart phone", "Tablet pc", "Laptop", "Electronic device"]:
        return "Electronic_Group"

    # 3. 접두사(Prefix) 기준 통합
    target_prefixes = [
        "Battery", "Drill", "Grenade", "Gun", "Knife", 
        "LAGs", "Monkey wrench", "Scissors", "Spanner", "Vise plier"
    ]
    
    for prefix in target_prefixes:
        if name.startswith(prefix):
            return f"{prefix}_Group"
            
    # 4. 그 외에는 원래 이름(혹은 ID) 사용
    return str(cid)

# =============================
# 기존 헬퍼 함수들
# =============================
def load_names(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        id2name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id2name = {i: str(nm) for i, nm in enumerate(names)}
    else:
        raise ValueError("names must be dict or list")
    return id2name

def get_gt_label_for_image(img_path: Path) -> Path:
    candidate = Path(GT_LABEL_ROOT) / f"{img_path.stem}.txt"
    return candidate if candidate.exists() else None

def parse_gt_label(txt_path: Path) -> List[int]:
    cls_ids = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split()
            if not tok: continue
            try: cls_ids.append(int(float(tok[0])))
            except: continue
    return cls_ids

verdict_map = {"TP": "정탐", "FP": "오탐", "FN": "미탐"}

# =============================
# 시각화 및 저장 Worker
# =============================
def draw_and_save_worker(img_path_str, model_names, pred_boxes_by_cls, target_cid, verdict, pred_ids, gt_ids):
    img = cv2.imread(img_path_str)
    if img is None: return

    # 🟢 정답(GT) 그리기
    for cid in gt_ids:
        if cid in pred_boxes_by_cls:
            for (x1, y1, x2, y2) in pred_boxes_by_cls[cid]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(img, model_names.get(cid, str(cid)), (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    # 🔴 오탐(FP) 그리기
    non_gt_pred = pred_ids - gt_ids
    for cid in non_gt_pred:
        if cid in pred_boxes_by_cls:
            for (x1, y1, x2, y2) in pred_boxes_by_cls[cid]:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, model_names.get(cid, str(cid)), (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 📝 텍스트 출력
    gt_names = [model_names.get(cid, str(cid)) for cid in sorted(gt_ids)]
    text_y = 30
    for name in gt_names:
        ts, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(img, name, (img.shape[1] - ts[0] - 10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        text_y += ts[1] + 10

    # 💾 저장
    verdict_kor = verdict_map.get(verdict, verdict)
    save_dir = Path(RESULT_DIR) / model_names.get(target_cid, str(target_cid)) / verdict_kor
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / Path(img_path_str).name), img)


# =============================
# MAIN
# =============================
def main():
    id2name = load_names(DATA_YAML)
    print(f"[i] 모델 로딩: {MODEL_PATH} (device={DEVICE})")
    
    # 🚀 모델을 DEVICE(cuda:1)로 로드
    model = YOLO(MODEL_PATH).to(DEVICE)

    stats = {}
    root = Path(VALID_ROOT)
    all_images = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    
    if not all_images:
        print(f"[!] 이미지가 없습니다: {VALID_ROOT}")
        return

    print(f"[i] 총 이미지 수: {len(all_images)}장 (Batch Size: {BATCH_SIZE})")

    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    
    for i in range(0, len(all_images), BATCH_SIZE):
        batch_paths = all_images[i : i + BATCH_SIZE]
        
        # 🚀 배치 추론 (device=DEVICE)
        results = model.predict(
            source=batch_paths,
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF_THRES,
            retina_masks=False,
            verbose=False,
            stream=False
        )

        for j, res in enumerate(results):
            img_path = batch_paths[j]
            lbl_path = get_gt_label_for_image(img_path)
            
            if not lbl_path: continue
            
            # 1. GT 파싱 및 제외(Exclusion) 필터링
            raw_gt_ids = parse_gt_label(lbl_path)
            gt_ids = set()
            for gid in raw_gt_ids:
                gname = id2name.get(gid, "")
                if not is_excluded(gname): 
                    gt_ids.add(gid)
            
            if not gt_ids: continue 

            # 2. 예측 ID 추출 및 제외(Exclusion) 필터링
            pred_ids = set()
            pred_boxes_by_cls = {}
            if res.boxes is not None and len(res.boxes) > 0:
                clses = res.boxes.cls.cpu().numpy().astype(int)
                xys = res.boxes.xyxy.cpu().numpy().astype(int)
                for c, box in zip(clses, xys):
                    pname = id2name.get(c, "")
                    if not is_excluded(pname): 
                        pred_ids.add(c)
                        if c not in pred_boxes_by_cls: pred_boxes_by_cls[c] = []
                        pred_boxes_by_cls[c].append(box)

            # 🔥 그룹핑 로직 적용
            # 1. 예측된 클래스들을 그룹명으로 변환
            pred_groups = {get_group_name(pid, id2name.get(pid, "")) for pid in pred_ids}
            
            # 2. 정답 클래스들을 그룹명으로 변환
            gt_groups_all = {get_group_name(gid, id2name.get(gid, "")) for gid in gt_ids}
            
            # 3. 정답 그룹에 없는 엉뚱한 그룹이 탐지되었는지 확인 (오탐 여부용)
            extra_groups_detected = pred_groups - gt_groups_all

            for cid in sorted(gt_ids):
                if cid not in stats: stats[cid] = {"TP": 0, "FP": 0, "FN": 0}
                
                # 현재 검사할 타겟의 그룹명
                c_group = get_group_name(cid, id2name.get(cid, ""))
                
                # 해당 그룹이 예측 목록에 있는가? 
                is_detected = c_group in pred_groups

                if is_detected:
                    verdict = "TP" if len(extra_groups_detected) == 0 else "FP"
                else:
                    verdict = "FP" if len(extra_groups_detected) > 0 else "FN"

                stats[cid][verdict] += 1

                # 저장 (시각화)
                executor.submit(
                    draw_and_save_worker, 
                    str(img_path), id2name, pred_boxes_by_cls, cid, verdict, pred_ids, gt_ids
                )

        if (i + BATCH_SIZE) % 100 < BATCH_SIZE:
            print(f"[{i}/{len(all_images)}] 처리 중...")

    print("[i] 마무리 저장 작업 대기 중...")
    executor.shutdown(wait=True)

    # ── 결과 출력 ────────────────────────────────
    print("\n========== Result Summary (Grouped & Filtered) ==========")
    print(f"{'Class':<25}{'TP':>6}{'FP':>6}{'FN':>6}{'Total':>9}{'TP%':>9}{'FP%':>9}{'FN%':>9}")
    print("-" * 86)

    all_tp = all_fp = all_fn = all_total = 0
    rows = []

    for cid in sorted(stats.keys()):
        cname = id2name.get(cid, str(cid))
        d = stats[cid]
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        tot = tp + fp + fn
        all_tp += tp; all_fp += fp; all_fn += fn; all_total += tot

        tp_r = (tp/tot*100) if tot else 0
        fp_r = (fp/tot*100) if tot else 0
        fn_r = (fn/tot*100) if tot else 0

        print(f"{cname:<25}{tp:>6}{fp:>6}{fn:>6}{tot:>9}{tp_r:>8.1f}%{fp_r:>8.1f}%{fn_r:>8.1f}%")
        rows.append([cname, tp, fp, fn, tot, round(tp_r,2), round(fp_r,2), round(fn_r,2)])

    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Class", "TP", "FP", "FN", "Total", "TP%", "FP%", "FN%"])
        w.writerows(rows)
        if all_total > 0:
            w.writerow(["ALL", all_tp, all_fp, all_fn, all_total, 
                        round(all_tp/all_total*100, 2), 
                        round(all_fp/all_total*100, 2), 
                        round(all_fn/all_total*100, 2)])
    
    print(f"\n[Done] 결과 저장 완료: {OUT_CSV}")

if __name__ == "__main__":
    main()