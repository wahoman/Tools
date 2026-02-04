import os
import json
import cv2
import numpy as np
import glob

# =========================================================
# 1. 사용자 설정 (경로 확인)
# =========================================================
# JSON 파일이 있는 폴더
json_folder = r"D:\hgyeo\testset_labeling\Central\PH\test_labels"

# 결과(.txt) 저장할 폴더
output_folder = r"D:\hgyeo\testset_labeling\Central\PH\labels"
# =========================================================

def convert_cv2_fill_holes_fixed():
    # 폴더 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    
    if not json_files:
        print(f"❌ 파일이 없습니다: {json_folder}")
        return

    print(f"📂 변환 시작 (OpenCV 모드 - 구멍 자동 삭제): {len(json_files)}개")

    count = 0
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_w = data.get("imageWidth")
            img_h = data.get("imageHeight")
            if not img_w or not img_h: continue

            yolo_lines = []
            
            # -----------------------------------------------------------
            # [1] 마스크 생성 (도화지 준비)
            # -----------------------------------------------------------
            mask = np.zeros((img_h, img_w), dtype=np.uint8)

            # JSON에 있는 모든 도형을 꺼내서
            for shape in data["shapes"]:
                points = np.array(shape["points"], dtype=np.int32)
                # 흰색(1)으로 꽉 채워서 그립니다. 
                # 이렇게 하면 겹치는 부분이나 안쪽 구멍들이 전부 흰색 덩어리가 됩니다.
                cv2.fillPoly(mask, [points], 1)

            # -----------------------------------------------------------
            # [2] 외곽선 재추출 (RETR_EXTERNAL 핵심)
            # -----------------------------------------------------------
            # 그려진 흰색 덩어리에서 '가장 바깥쪽 선'만 따옵니다.
            # 안쪽에 있던 구멍(검은색)이나 작은 도형들은 무시됩니다.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 너무 작은 노이즈(점)는 제외
                if len(contour) < 3: continue
                
                # 좌표 정규화
                normalized_points = []
                for point in contour:
                    x, y = point[0] # cv2는 [[x,y]] 형태라 [0] 필요
                    
                    # 0~1 사이로 맞춤
                    nx = max(0, min(1, x / img_w))
                    ny = max(0, min(1, y / img_h))
                    
                    normalized_points.append(f"{nx:.6f}")
                    normalized_points.append(f"{ny:.6f}")
                
                # -------------------------------------------------------
                # [3] 저장 (무조건 0번 클래스)
                # -------------------------------------------------------
                line = "0 " + " ".join(normalized_points)
                yolo_lines.append(line)

            # TXT 저장
            if yolo_lines:
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                txt_path = os.path.join(output_folder, base_name + ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                count += 1

        except Exception as e:
            print(f"❌ 에러 ({os.path.basename(json_file)}): {e}")

    print(f"🎉 완료! 총 {count}개 파일 변환됨.")
    print(f"📂 저장 경로: {output_folder}")

if __name__ == "__main__":
    convert_cv2_fill_holes_fixed()