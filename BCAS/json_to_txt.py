import os
import json
import glob
from multiprocessing import Pool, cpu_count

# =========================================================
# [1] 설정 경로
# =========================================================
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-2\json_labels"
txt_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-2\labels"
# =========================================================

def process_single_file(txt_file):
    """
    파일 하나를 처리하는 함수 (병렬 처리를 위해 함수로 분리)
    """
    try:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        json_path = os.path.join(json_folder, base_name + ".json")\
        
        # JSON 파일이 없으면 패스
        if not os.path.exists(json_path):
            return 0 # 매칭 실패

        # 1. 파일 읽기
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        with open(txt_file, "r", encoding="utf-8") as f:
            txt_lines = f.readlines()

        img_w = json_data.get("imageWidth")
        img_h = json_data.get("imageHeight")

        if not img_w or not img_h:
            return 0 # 크기 정보 없음

        # 2. 좌표 변환
        new_shapes = []
        for line in txt_lines:
            parts = line.strip().split()
            if len(parts) < 3: continue

            label = parts[0]
            coords = [float(x) for x in parts[1:]]
            
            points = []
            for i in range(0, len(coords), 2):
                points.append([coords[i] * img_w, coords[i+1] * img_h])
            
            new_shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

        # 3. JSON 덮어쓰기
        json_data["shapes"] = new_shapes

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return 1 # 성공

    except Exception as e:
        print(f"❌ 에러({base_name}): {e}")
        return 0

def main():
    # TXT 파일 목록 로드
    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))
    
    if not txt_files:
        print("❌ TXT 파일이 없습니다.")
        return

    # CPU 코어 개수 확인
    num_cores = cpu_count()
    print(f"🚀 멀티프로세싱 시작! (CPU 코어 {num_cores}개 사용)")
    print(f"📂 총 {len(txt_files)}개 파일 처리 중...")

    # 병렬 처리 시작
    with Pool(num_cores) as pool:
        results = pool.map(process_single_file, txt_files)

    # 결과 집계
    success_count = sum(results)
    
    print(f"\n🎉 작업 완료!")
    print(f"✅ 성공적으로 업데이트됨: {success_count}개")

if __name__ == "__main__":
    # 윈도우에서 멀티프로세싱을 쓰려면 이 구문이 필수입니다.
    main()