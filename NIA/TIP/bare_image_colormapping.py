import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

# ==========================================
# 1. 경로 및 설정
# ==========================================
RAW_ROOT_DIR = r"D:\hgyeo\BCAS_TIP\bare_image_raw"
LABEL_ROOT_DIR = r"D:\hgyeo\BCAS_TIP\bare_image_png"
SAVE_ROOT_DIR = r"D:\hgyeo\BCAS_TIP\bare_image_raw_crop"

# ★ RAW 이미지 해상도 설정 (필수 수정) ★
# RAW 파일은 헤더가 없어서 해상도를 모르면 이미지가 깨집니다. 꼭 맞춰주세요.
RAW_WIDTH = 2448  
RAW_HEIGHT = 2048 
RAW_DTYPE = np.uint8 # 8bit면 uint8, 16bit면 uint16

def main():
    # 1. 클래스 목록 가져오기 (bare_image_png 폴더 안에 있는 폴더들을 클래스로 간주)
    class_list = [d for d in os.listdir(LABEL_ROOT_DIR) if os.path.isdir(os.path.join(LABEL_ROOT_DIR, d))]
    
    print(f"[*] 감지된 클래스 목록: {class_list}")

    for class_name in class_list:
        # 각 경로 설정
        label_dir = os.path.join(LABEL_ROOT_DIR, class_name, "labels")
        raw_dir = os.path.join(RAW_ROOT_DIR, class_name, "raws")
        save_class_dir = os.path.join(SAVE_ROOT_DIR, class_name)

        # 라벨 폴더나 RAW 폴더가 없으면 건너뜀
        if not os.path.exists(label_dir) or not os.path.exists(raw_dir):
            print(f"[-] {class_name}: 경로가 존재하지 않아 건너뜁니다.")
            continue

        # 저장 경로 생성
        os.makedirs(save_class_dir, exist_ok=True)

        # 라벨 파일 리스트 로드
        label_files = glob.glob(os.path.join(label_dir, "*.*"))
        
        # 진행상황 표시
        print(f"\n[-] Processing Class: {class_name} ({len(label_files)} files)")

        for label_path in tqdm(label_files):
            # 파일명(확장자 제외) 추출 -> RAW 파일 찾기용
            filename = os.path.basename(label_path)
            file_id = os.path.splitext(filename)[0]
            
            # 매칭되는 RAW 파일 경로 구성
            raw_path = os.path.join(raw_dir, f"{file_id}.raw")

            # RAW 파일이 실제로 있는지 확인
            if not os.path.exists(raw_path):
                # 라벨은 있는데 RAW가 없는 경우
                continue

            try:
                # ========================================================
                # [좌표 파싱 구간] : 사용하시는 라벨 포맷에 맞춰 수정 필요
                # ========================================================
                # 예: txt 파일에서 읽는 경우
                # with open(label_path, 'r') as f:
                #     line = f.readline()
                #     # YOLO 형식 등 파싱 로직
                
                # ★ 임시 좌표 (수정 필요) ★
                # 현재는 테스트를 위해 임의의 값을 넣었습니다. 
                # 실제 라벨 파일 내용을 읽어서 x, y, w, h에 할당해주세요.
                x, y, w, h = 500, 500, 300, 300 

                # --------------------------------------------------------

                # RAW 이미지 로드
                raw_data = np.fromfile(raw_path, dtype=RAW_DTYPE)
                image = raw_data.reshape((RAW_HEIGHT, RAW_WIDTH)) # 1채널 가정

                # 이미지 크롭 (범위 벗어남 방지 처리 포함)
                img_h, img_w = image.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)

                crop_img = image[y:y+h, x:x+w]

                # 저장 (파일명은 원본 유지 + .png)
                save_path = os.path.join(save_class_dir, f"{file_id}.png")
                cv2.imwrite(save_path, crop_img)

            except Exception as e:
                print(f"[Error] {file_id} 처리 중 오류: {e}")

if __name__ == "__main__":
    main()