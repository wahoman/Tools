# fast_pad_or_crop_to_760_mp.py
import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse

# ===== 설정 =====
TARGET_H = 760
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # 무손실 PNG, 빠른 압축
MIN_UNDERSCORES = 5
EXT = ".png"
# ===============

def list_targets(root_path: str):
    """root_path 하위의 모든 images 폴더에서 언더바 5개 이상인 PNG만 수집."""
    files = []
    for dirpath, _, filenames in os.walk(root_path):
        if os.path.basename(dirpath).lower() != "images":
            continue
        for fn in filenames:
            if fn.lower().endswith(EXT) and fn.count("_") >= MIN_UNDERSCORES:
                files.append(os.path.join(dirpath, fn))
    return files

def process_one(file_path: str):
    """
    단일 파일 처리:
      - H < 760: 아래쪽 replicate 패딩
      - H > 760: 아래쪽 크롭
      - H = 760: 건너뜀(재저장 안 함 → I/O 절감)
    반환: (result_key, file_path, folder_path)
    """
    folder_path = str(Path(file_path).parent)  # images 폴더 경로(로그용)
    try:
        np_arr = np.fromfile(file_path, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return ("read_fail", file_path, folder_path)

        h, w = img.shape[:2]
        if h < TARGET_H:
            pad = TARGET_H - h
            result = cv2.copyMakeBorder(img, 0, pad, 0, 0, borderType=cv2.BORDER_REPLICATE)
            kind = "padded"
        elif h > TARGET_H:
            result = img[:TARGET_H, :, :]  # 위에서부터 760만 유지
            kind = "cropped"
        else:
            return ("skipped", file_path, folder_path)  # 재저장 생략

        ok, enc = cv2.imencode(EXT, result, PNG_PARAMS)
        if not ok:
            return ("save_fail", file_path, folder_path)
        with open(file_path, "wb") as f:
            f.write(enc)
        return (kind, file_path, folder_path)

    except Exception as e:
        return ("error", f"{file_path} :: {e}", folder_path)

def pad_or_crop_to_760_mp(root_path: str, workers: int = 4, progress_every: int = 1000):
    files = list_targets(root_path)
    if not files:
        print("처리할 PNG 파일을 찾지 못했습니다.", flush=True)
        return

    print(f"대상 파일: {len(files):,}개 | 워커: {workers}", flush=True)
    stats = {"padded":0, "cropped":0, "skipped":0, "read_fail":0, "save_fail":0, "error":0}
    seen_folders = set()  # 처음 등장하는 images 폴더만 로그

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, fp) for fp in files]
        for i, fut in enumerate(as_completed(futures), 1):
            k, file_path, folder_path = fut.result()

            # 폴더 최초 등장 시 1회 출력
            if folder_path not in seen_folders:
                seen_folders.add(folder_path)
                try:
                    rel_folder = str(Path(folder_path).resolve().relative_to(Path(root_path).resolve()))
                except Exception:
                    rel_folder = folder_path
                print(f"\n▶ 처리 중: {rel_folder}", flush=True)

            stats[k] = stats.get(k, 0) + 1

            if progress_every and i % progress_every == 0:
                print(f"...진행 {i:,}/{len(files):,}", flush=True)

    print("\n✨ 작업 완료!", flush=True)
    print(f"- 패딩(H<760): {stats['padded']:,}", flush=True)
    print(f"- 크롭(H>760): {stats['cropped']:,}", flush=True)
    print(f"- 그대로 둠(H=760): {stats['skipped']:,}", flush=True)
    print(f"- 읽기 실패: {stats['read_fail']:,}", flush=True)
    print(f"- 저장 실패: {stats['save_fail']:,}", flush=True)
    print(f"- 기타 오류: {stats['error']:,}", flush=True)
    print(f"- 총 대상: {sum(stats.values()):,}", flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="Pad(<760) or crop(>760) PNGs to H=760 with multiprocessing (in-place).")
    p.add_argument("--root", type=str, default=r"\\Sstl_nas\ai\datasets\NIA_new_colormapping",
                   help="루트 경로 (train/valid 포함)")
    p.add_argument("--workers", type=int, default=4,   # HDD 기본값
                   help="프로세스 개수 (HDD: 2~4 권장, SSD/NVMe는 늘려도 OK)")
    p.add_argument("--progress-every", type=int, default=1000,
                   help="진행 로그 출력 간격(파일 개수)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pad_or_crop_to_760_mp(args.root, workers=args.workers, progress_every=args.progress_every)
