import os
import json
import cv2
import numpy as np
import glob
from multiprocessing import Pool, cpu_count, freeze_support

# =========================================================
# [1] 사용자 설정 구역
# =========================================================
# 작업할 JSON 폴더 경로
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-2\json_labels"

# 기능 스위치
DO_CONVERT_TXT = True       # 라벨 변환 (JSON -> TXT)
DO_REMOVE_IMAGEDATA = True  # 용량 최적화 (imageData -> null)

# [자동 설정] 출력 폴더 경로
parent_dir = os.path.dirname(json_folder.rstrip(os.sep)) 
output_folder = os.path.join(parent_dir, "labels")
# =========================================================

def get_class_name_from_filename(filename):
    """파일명 분석하여 클래스명 결정"""
    try:
        parts = filename.split('_')
        if len(parts) < 12: return "Unknown"

        threat_item = parts[4]          
        bg_item_type_str = parts[11]    
        
        try:
            bg_item_val = int(bg_item_type_str)
        except ValueError:
            bg_item_val = 0 

        if threat_item.lower() == 'x' and bg_item_val >= 3:
            return "electronics"
        
        return threat_item
    except:
        return "Error"

def process_single_file(json_file):
    """
    하나의 파일에 대해 변환 및 최적화를 수행하는 작업자 함수
    (이 함수가 병렬로 실행됩니다)
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
        # [기능 1] JSON -> TXT 변환
        # -----------------------------------------------------
        if DO_CONVERT_TXT:
            class_name = get_class_name_from_filename(base_name)
            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")

            if img_w and img_h:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for shape in data.get("shapes", []):
                    points = np.array(shape["points"], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                yolo_lines = []
                for contour in contours:
                    if len(contour) < 3: continue 
                    
                    normalized_points = []
                    for point in contour:
                        x, y = point[0]
                        nx = max(0, min(1, x / img_w))
                        ny = max(0, min(1, y / img_h))
                        normalized_points.append(f"{nx:.6f}")
                        normalized_points.append(f"{ny:.6f}")
                    
                    line = f"{class_name} " + " ".join(normalized_points)
                    yolo_lines.append(line)

                if yolo_lines:
                    # 병렬 처리 중 폴더 생성 경합 방지를 위해 미리 생성된 폴더 사용
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

    # 사용 가능한 CPU 코어 수 확인 (안전하게 1개 남겨두거나 전부 사용)
    num_cores = cpu_count()
    print(f"\n🚀 병렬 처리 시작! (사용 CPU 코어: {num_cores}개)")
    print(f"📂 총 파일 수: {total_files}개")
    print("=" * 60)

    # 병렬 처리 실행
    txt_cnt = 0
    clean_cnt = 0
    err_cnt = 0

    # Pool을 사용하여 병렬 작업 분배
    with Pool(processes=num_cores) as pool:
        # 진행 상황을 보기 위해 imap 사용 (순서 상관 없음)
        for i, res in enumerate(pool.imap_unordered(process_single_file, json_files), 1):
            if res["txt_created"]: txt_cnt += 1
            if res["json_cleaned"]: clean_cnt += 1
            if res["error"]:
                print(f"❌ 오류 ({res['filename']}): {res['error']}")
                err_cnt += 1
            
            # 100개 단위로 로그 출력 (너무 자주 찍으면 느려짐)
            if i % 100 == 0 or i == total_files:
                print(f"   >>> 진행중: {i}/{total_files} ({(i/total_files)*100:.1f}%) 완료")

    print("=" * 60)
    print(f"🎉 모든 작업 완료!")
    print(f" - TXT 생성 완료 : {txt_cnt}개")
    print(f" - JSON 최적화   : {clean_cnt}개")
    print(f" - 에러 발생     : {err_cnt}개")

if __name__ == "__main__":
    main()