#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path

# ─── 사용자 설정 ───
DEFAULT_DIR = "/home/hgyeo/Desktop/BCAS/data_merge"
DEFAULT_RECURSIVE = True  # 하위 폴더까지 탐색 (train/valid 구조 대응)
# ──────────────────

def normalize_line(s: str) -> str:
    """
    공백 개수가 달라도 내용이 같으면 같은 줄로 인식하기 위해 정규화
    예: "0   1 2" 와 "0 1 2" 는 같은 것으로 취급
    """
    return " ".join(s.strip().split())

def process_file(path: Path) -> bool:
    try:
        orig = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
        
    lines = [ln for ln in orig.splitlines() if ln.strip()]
    if not lines:
        return False

    # ─────────────────────────────────────────────
    # [핵심 로직] 중복 라인 제거
    # ─────────────────────────────────────────────
    seen = set()
    result = []
    
    for ln in lines:
        # 공백을 정리해서 비교 키를 만듦 (띄어쓰기 달라도 내용 같으면 중복 처리)
        key = normalize_line(ln)
        
        # 이미 등록된 라인이면 건너뜀 (삭제)
        if key in seen:
            continue
            
        # 처음 보는 라인이면 등록하고 결과에 추가
        seen.add(key)
        result.append(ln)

    # 변경된 내용 조합
    new_text = "\n".join(result) + ("\n" if orig.endswith("\n") else "")
    
    # 내용이 달라졌을 때만 파일 덮어쓰기
    if new_text != orig:
        # 혹시 모르니 .bak 백업 파일 생성 (필요 없으면 주석 처리)
        # bak = path.with_suffix(path.suffix + ".bak")
        # if not bak.exists():
        #     bak.write_text(orig, encoding="utf-8", errors="ignore")
            
        path.write_text(new_text, encoding="utf-8")
        return True
    return False

def run(labels_dir: str, recursive: bool):
    root = Path(labels_dir)
    if not root.exists():
        print(f"[!] 경로 없음: {root}")
        sys.exit(1)

    pattern = "**/*.txt" if recursive else "*.txt"
    files = list(root.glob(pattern))
    
    if not files:
        print("[i] 처리할 txt 파일이 없습니다.")
        return

    print(f"🚀 '{root}' 내의 중복 라인 제거 시작 (총 {len(files)}개 파일)...")
    
    changed = 0
    for f in files:
        try:
            if process_file(f):
                changed += 1
                # print(f"  수정됨: {f.name}")  # 상세 로그 필요시 주석 해제
        except Exception as e:
            print(f"[!] 실패: {f} -> {e}")
            
    print(f"\n[✓] 완료! 총 {len(files)}개 중 {changed}개 파일에서 중복 라인을 제거했습니다.")

def main():
    ap = argparse.ArgumentParser(description="YOLO 라벨 파일 내 중복 라인 제거기")
    ap.add_argument("labels_dir", nargs="?", help="라벨 루트 폴더")
    ap.add_argument("--recursive", action="store_true", help="하위 폴더 포함")
    args = ap.parse_args()

    labels_dir = args.labels_dir or DEFAULT_DIR
    recursive = args.recursive or DEFAULT_RECURSIVE
    
    run(labels_dir, recursive)

if __name__ == "__main__":
    main()