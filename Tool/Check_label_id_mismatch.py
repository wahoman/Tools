from pathlib import Path
import yaml
import sys
from collections import defaultdict

# ───── 사용자 설정 ────────────────────────────────────────────────────────
SRC_ROOT = Path("/home/hgyeo/Desktop/1217")
YAML = Path("/home/hgyeo/Desktop/yaml/1208.yaml")
# ─────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path):
    try:
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = d.get("names")
        
        name2id = {}
        id2name = {}
        
        if isinstance(names, dict):
            for k, v in names.items():
                if v is not None:
                    name2id[str(v)] = int(k)
                    id2name[int(k)] = str(v)
        elif isinstance(names, list):
            for i, v in enumerate(names):
                if v is not None:
                    name2id[str(v)] = i
                    id2name[i] = str(v)
        
        # ⭐ 중요: 긴 이름부터 검사하도록 정렬 (Saw blade가 Saw보다 먼저 매칭되게 함)
        sorted_names = sorted(name2id.keys(), key=len, reverse=True)
        
        return name2id, id2name, sorted_names
    except Exception as e:
        sys.exit(f"❌ YAML 파일을 읽는 중 오류 발생: {e}")

def check_mismatch():
    name2id, id2name, sorted_names = load_yaml(YAML)
    
    print(f"🔍 검사 시작: {SRC_ROOT}")
    print("-" * 60)

    for split in ["train", "valid"]:
        split_dir = SRC_ROOT / split
        if not split_dir.exists():
            continue

        print(f"\n📂 [{split.upper()}] 세트 검사 중...")
        
        stats = defaultdict(lambda: {"total": 0, "mismatch": 0, "wrong_ids": set()})
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            expected_id = None
            
            # 긴 이름부터 순서대로 폴더명에 포함되어 있는지 확인
            for y_name in sorted_names:
                if y_name in class_name: 
                    expected_id = name2id[y_name]
                    break
            
            if expected_id is None:
                continue

            label_dir = class_dir / "labels"
            if not label_dir.exists():
                continue

            for lbl in label_dir.glob("*.txt"):
                stats[class_name]["total"] += 1
                try:
                    lines = lbl.read_text().strip().splitlines()
                    is_mismatch = False
                    for ln in lines:
                        parts = ln.split()
                        if not parts: continue
                        
                        current_id = int(parts[0])
                        if current_id != expected_id:
                            is_mismatch = True
                            stats[class_name]["wrong_ids"].add(current_id)
                    
                    if is_mismatch:
                        stats[class_name]["mismatch"] += 1
                except Exception:
                    continue

        # 결과 출력
        has_issue_in_split = False
        for c_name, s in stats.items():
            if s["mismatch"] > 0:
                has_issue_in_split = True
                wrong_names = [id2name.get(wid, f"ID:{wid}") for wid in s["wrong_ids"]]
                print(f"❌ 폴더 [{c_name}]: 불일치 {s['mismatch']}개 발견! (전체 {s['total']}개)")
                print(f"   └─ 원래 기대한 ID: {next((name2id[n] for n in sorted_names if n in c_name), 'Unknown')}")
                print(f"   └─ 실제 발견된 ID: {list(s['wrong_ids'])} ({', '.join(wrong_names)})")
        
        if not has_issue_in_split:
            print(f"✅ {split} 세트는 모두 정상입니다.")

if __name__ == "__main__":
    check_mismatch()