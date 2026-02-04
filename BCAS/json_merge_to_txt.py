import os
import json
import glob
from multiprocessing import Pool, cpu_count, freeze_support

# =========================================================
# [1] 사용자 설정 구역
# =========================================================
# 작업할 JSON 폴더 경로 (병합된 폴더 경로를 넣으세요)
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\json_labels"

# 기능 스위치
DO_CONVERT_TXT = True       # 라벨 변환 (JSON -> TXT)
DO_REMOVE_IMAGEDATA = True  # 용량 최적화 (imageData -> null)

# [자동 설정] 출력 폴더 경로 (labels 폴더 자동 생성)
parent_dir = os.path.dirname(json_folder.rstrip(os.sep)) 
output_folder = os.path.join(parent_dir, "labels")
# =========================================================

def process_single_file(json_file):
    """
    하나의 파일에 대해 변환 및 최적화를 수행하는 작업자 함수
    """
    result_info = {
        "txt_created": False,
        "json_cleaned": False,
        "error": None,
        "filename": os.path.basename(json_file)
    }

    try:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        
        # JSON 로드
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # -----------------------------------------------------
        # [기능 1] JSON -> TXT 변환 (파일명 추론 로직 제거됨)
        # -----------------------------------------------------
        if DO_CONVERT_TXT:
            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")

            yolo_lines = []

            # shapes 리스트를 순회하며 라벨과 좌표를 직접 가져옴
            if img_w and img_h:
                for shape in data.get("shapes", []):
                    # 1. JSON 내부의 라벨을 그대로 사용
                    label = shape.get("label", "Unknown")
                    points = shape.get("points", [])

                    # 2. 좌표 정규화 (0~1 사이 값으로 변환)
                    normalized_points = []
                    for x, y in points:
                        nx = max(0, min(1, x / img_w))
                        ny = max(0, min(1, y / img_h))
                        normalized_points.append(f"{nx:.6f}")
                        normalized_points.append(f"{ny:.6f}")
                    
                    # 3. 한 줄 생성 (라벨명 + 좌표들)
                    # 주의: YOLO 학습 시에는 라벨명(문자열)을 숫자(ID)로 바꿔야 할 수 있습니다.
                    if normalized_points:
                        line = f"{label} " + " ".join(normalized_points)
                        yolo_lines.append(line)

                # 4. TXT 파일 저장 (파일명은 JSON과 동일하게 유지)
                if yolo_lines:
                    txt_path = os.path.join(output_folder, base_name + ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    result_info["txt_created"] = True

        # -----------------------------------------------------
        # [기능 2] imageData 제거
        # -----------------------------------------------------
        if DO_REMOVE_IMAGEDATA:
            if data.get('imageData') is not None:
                data['imageData'] = None
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                result_info["json_cleaned"] = True

    except Exception as e:
        result_info["error"] = str(e)

    return result_info

def main():
    # 윈도우 멀티프로세싱 필수 설정
    freeze_support()

    print(f"📍 입력 경로: {json_folder}")
    if DO_CONVERT_TXT:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"📍 출력 경로: {output_folder}")
    
    # 파일 리스트 확보
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("❌ 처리할 JSON 파일이 없습니다.")
        return

    # 사용 가능한 CPU 코어 수 확인
    num_cores = cpu_count()
    print(f"\n🚀 병렬 처리 시작! (사용 CPU 코어: {num_cores}개)")
    print(f"📂 총 파일 수: {total_files}개")
    print("=" * 60)

    # 병렬 처리 실행
    txt_cnt = 0
    clean_cnt = 0
    err_cnt = 0

    with Pool(processes=num_cores) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_file, json_files), 1):
            if res["txt_created"]: txt_cnt += 1
            if res["json_cleaned"]: clean_cnt += 1
            if res["error"]:
                print(f"❌ 오류 ({res['filename']}): {res['error']}")
                err_cnt += 1
            
            if i % 100 == 0 or i == total_files:
                print(f"   >>> 진행중: {i}/{total_files} ({(i/total_files)*100:.1f}%) 완료")

    print("=" * 60)
    print(f"🎉 모든 작업 완료!")
    print(f" - TXT 생성 완료 : {txt_cnt}개")
    print(f" - JSON 최적화   : {clean_cnt}개")
    print(f" - 에러 발생     : {err_cnt}개")

if __name__ == "__main__":
    main()