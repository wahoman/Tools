#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
polygon YOLO → 증강 → 패딩 크롭 → (옵션) Gray 저장
멀티프로세스 + 파이프라인 캐시 + 빠른 PNG 저장
"""

import os, random, inspect
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import albumentations as A

# ─────────── 사용자 설정 ───────────
SRC_ROOT   = Path("/home/hgyeo/Desktop/1217")
DST_ROOT   = Path("/home/hgyeo/Desktop/1224_aug")
TARGET_CNT = {"train": 500, "valid": 100}
SAVE_GRAY  = False
PNG_ARGS   = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # 압축률 1
# ──────────────────────────────────

# ---- 1) keep_size 호환 처리 ----
HAS_KEEP_SIZE = 'keep_size' in inspect.signature(A.Rotate.__init__).parameters
def keep(t_cls, *a, **k):
    return t_cls(*a, keep_size=False, **k) if HAS_KEEP_SIZE else t_cls(*a, **k)

AUG_DEFS = {
    "rotate90": keep(A.Rotate, limit=(90, 90), p=1.0),
    "hflip":    A.HorizontalFlip(p=1.0),
    "vflip":    A.VerticalFlip(p=1.0),
    "distort":  keep(A.GridDistortion, num_steps=5, distort_limit=0.3, p=1.0),
    "elastic":  keep(A.ElasticTransform, alpha=1.0, sigma=50, p=1.0),
}

# ---- 2) 보조 함수 ----
def ensure(p: Path): p.mkdir(parents=True, exist_ok=True)

def to_gray(arr: np.ndarray) -> np.ndarray:
    if not SAVE_GRAY or arr.ndim == 2:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

def crop_black(img: np.ndarray, polys_pix):
    nz = cv2.findNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if nz is None:
        return img, polys_pix
    x, y, w, h = cv2.boundingRect(nz)
    cropped = img[y:y + h, x:x + w]
    shifted = [[c - x if i % 2 == 0 else c - y for i, c in enumerate(p)]
               for p in polys_pix]
    return cropped, shifted

def load_yolo(p: Path):
    cls, polys = [], []
    for ln in p.read_text().splitlines():
        sp = ln.split()
        if len(sp) >= 3:
            cls.append(int(sp[0]))
            polys.append(list(map(float, sp[1:])))
    return cls, polys

def save_yolo(p: Path, cls, polys):
    with open(p, "w") as f:
        for c, poly in zip(cls, polys):
            f.write(f"{c} " + " ".join(map(str, poly)) + "\n")

# ---- 3) 클래스 폴더 단위 증강 ----
def augment_class(args):
    split, cdir = args
    random.seed(os.getpid())          # 각 프로세스 시드 달리기
    outI = DST_ROOT/split/cdir.name/"images"
    outL = DST_ROOT/split/cdir.name/"labels"
    ensure(outI); ensure(outL)

    imgD, lblD = cdir/"images", cdir/"labels"
    imgs = sorted(glob(str(imgD/"*.png")))
    if not imgs:
        return f"{split}/{cdir.name}: 0→0"

    # ▸ 파이프라인 캐시 (프로세스마다 1회)
    kp_param = A.KeypointParams(format="xy", remove_invisible=False)
    PIPE     = {k: A.Compose([aug], keypoint_params=kp_param)
                for k, aug in AUG_DEFS.items()}

    # 1) 원본 복사
    for fp in imgs:
        name = Path(fp).name
        lbl  = lblD/name.replace(".png", ".txt")
        if not lbl.exists(): continue
        if not (outI/name).exists():
            cv2.imwrite(str(outI/name), to_gray(cv2.imread(fp)), PNG_ARGS)
        if not (outL/name.replace(".png", ".txt")).exists():
            save_yolo(outL/name.replace(".png", ".txt"), *load_yolo(lbl))

    cur, tgt = len(imgs), TARGET_CNT[split]
    if cur >= tgt:
        return f"{split}/{cdir.name}: {cur}→{cur}"
    need = tgt - cur
    done = loop_idx = 0

    while done < need:
        src  = random.choice(imgs)
        stem = Path(src).stem
        img  = cv2.imread(src)
        if img is None: continue
        H0, W0 = img.shape[:2]
        cls, polys = load_yolo(lblD/f"{stem}.txt")
        if not polys: continue

        # norm → pixel
        pix = [[c*(W0 if i%2==0 else H0) for i,c in enumerate(p)] for p in polys]
        kps = [(p[i],p[i+1]) for p in pix for i in range(0,len(p),2)]

        for aug_name, pipe in PIPE.items():
            if done >= need: break
            out = pipe(image=img, keypoints=kps)
            if len(out["keypoints"]) != len(kps):
                loop_idx += 1; continue

            # 복원
            pix_new, ptr = [], 0
            for p in polys:
                buf=[]
                for _ in range(len(p)//2):
                    x,y = out["keypoints"][ptr]; buf.extend([x,y]); ptr+=1
                pix_new.append(buf)

            img_crop, pix_new = crop_black(out["image"], pix_new)
            Hc, Wc = img_crop.shape[:2]
            norm = [[c/(Wc if i%2==0 else Hc) for i,c in enumerate(p)] for p in pix_new]

            out_name = f"{stem}_{aug_name}_{loop_idx}.png"
            cv2.imwrite(str(outI/out_name), to_gray(img_crop), PNG_ARGS)
            save_yolo(outL/out_name.replace(".png",".txt"), cls, norm)
            done  += 1; loop_idx += 1

    return f"{split}/{cdir.name}: {cur}→{cur+done}"

# ---- 4) 멀티프로세스 실행 ----
if __name__ == "__main__":
    cv2.setNumThreads(0)          # OpenCV 내부 스레드 끄기
    tasks=[]
    for sp in ("train", "valid"):
        for cdir in (SRC_ROOT/sp).iterdir():
            if cdir.is_dir():
                tasks.append((sp, cdir))

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for msg in ex.map(augment_class, tasks):
            print("✅", msg)

    print("🎉 모든 증강 작업 완료")
