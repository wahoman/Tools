import os
import json
import cv2
import numpy as np
import glob
from multiprocessing import Pool, cpu_count, freeze_support

# =========================================================
# [1] 사용자 설정 구역
# =========================================================
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY1\json_labels"

DO_CONVERT_TXT = True       # TXT 생성 (라벨을 문자열로 저장)
DO_REMOVE_IMAGEDATA = True  # imageData 제거
DO_UPDATE_JSON_LABEL = True # JSON 내부의 label 명칭도 파일명 기준으로 변경

# [자동 설정] 출력 폴더
parent_dir = os.path.dirname(json_folder.rstrip(os.sep)) 
output_folder = os.path.join(parent_dir, "labels")
# =========================================================

def get_class_name_from_filename(filename):
    """파일명의 4번째 인덱스(_)에서 클래스명 추출"""
    try:
        pure_name = os.path.basename(filename)
        parts = pure_name.split('_')
        
        # 예: E3S690G3(0)_00131251(1)_C(2)_6(3)_Printer-Cartridge(4)
        if len(parts) >= 5:
            threat_item = parts[4]
            
            # 기존 electronics 판별 로직 유지 (필요 없으면 제거 가능)
            if len(parts) >= 12:
                try:
                    bg_item_val = int(parts[11])
                    if threat_item.lower() == 'x' and bg_item_val >= 3:
                        return "electronics"
                except: pass
                
            return threat_item
        return "Unknown"
    except:
        return "Error"

def process_single_file(json_file):
    result_info = {"txt_created": False, "json_updated": False, "error": None, "filename": os.path.basename(json_file)}

    try:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        # 파일명에서 "Printer-Cartridge" 추출
        class_name = get_class_name_from_filename(base_name)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # -----------------------------------------------------
        # [기능 1] JSON 데이터 수정 (라벨 이름 변경 + 용량 최적화)
        # -----------------------------------------------------
        json_changed = False
        
        # JSON 안의 모든 shape 라벨을 추출한 이름으로 강제 통일
        if DO_UPDATE_JSON_LABEL:
            for shape in data.get("shapes", []):
                if shape["label"] != class_name:
                    shape["label"] = class_name
                    json_changed = True
        
        if DO_REMOVE_IMAGEDATA and data.get('imageData') is not None:
            data['imageData'] = None
            json_changed = True

        if json_changed:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            result_info["json_updated"] = True

        # -----------------------------------------------------
        # [기능 2] TXT 변환 (문자열 라벨 사용)
        # -----------------------------------------------------
        if DO_CONVERT_TXT:
            img_w, img_h = data.get("imageWidth"), data.get("imageHeight")
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
                        nx, ny = x / img_w, y / img_h
                        normalized_points.append(f"{max(0, min(1, nx)):.6f}")
                        normalized_points.append(f"{max(0, min(1, ny)):.6f}")
                    
                    # 숫자가 아닌 "Printer-Cartridge" 문자열이 바로 들어감
                    line = f"{class_name} " + " ".join(normalized_points)
                    yolo_lines.append(line)

                if yolo_lines:
                    txt_path = os.path.join(output_folder, base_name + ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    result_info["txt_created"] = True

    except Exception as e:
        result_info["error"] = str(e)

    return result_info

def main():
    freeze_support()
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("❌ JSON 파일을 찾을 수 없습니다.")
        return

    num_cores = cpu_count()
    print(f"🚀 작업 시작 (코어: {num_cores}개, 파일: {total_files}개)")

    txt_cnt, json_cnt, err_cnt = 0, 0, 0
    with Pool(processes=num_cores) as pool:
        for i, res in enumerate(pool.imap_unordered(process_single_file, json_files), 1):
            if res["txt_created"]: txt_cnt += 1
            if res["json_updated"]: json_cnt += 1
            if res["error"]: 
                print(f"❌ 에러: {res['filename']} - {res['error']}")
                err_cnt += 1
            if i % 100 == 0: print(f" 진행중: {i}/{total_files} 완료")

    print(f"\n✅ 완료! TXT 생성: {txt_cnt} / JSON 업데이트: {json_cnt} / 에러: {err_cnt}")

if __name__ == "__main__":
    main()