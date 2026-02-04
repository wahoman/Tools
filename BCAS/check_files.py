import os

# 1. 경로 설정 (사용자 경로 그대로 유지)
images_dir = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\images"
labels_dir = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY9-1\Laptop_json_labels"

def check_bidirectional_files(img_path, lbl_path):
    # 폴더 존재 확인
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        print("경로를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # 2. 파일 목록 가져오기 (시스템 파일 제외 등 필터링이 필요하면 추가 가능)
    img_files = os.listdir(img_path)
    lbl_files = os.listdir(lbl_path)

    # 3. { '파일이름(확장자제외)': '원래파일이름' } 형태의 딕셔너리 생성
    # 이렇게 하면 나중에 결과를 출력할 때 확장자가 포함된 원래 이름을 보여줄 수 있습니다.
    img_map = {os.path.splitext(f)[0]: f for f in img_files}
    lbl_map = {os.path.splitext(f)[0]: f for f in lbl_files}

    # 파일 이름(key)만 추출하여 집합(Set)으로 변환
    img_keys = set(img_map.keys())
    lbl_keys = set(lbl_map.keys())

    # ---------------------------------------------------------
    # 4. 양방향 비교 (집합의 차집합 연산 이용)
    # ---------------------------------------------------------
    
    # Case A: 이미지 집합 - 라벨 집합 = 라벨이 없는 이미지들
    imgs_missing_labels = img_keys - lbl_keys
    
    # Case B: 라벨 집합 - 이미지 집합 = 이미지가 없는 라벨들
    labels_missing_imgs = lbl_keys - img_keys

    # ---------------------------------------------------------
    # 5. 결과 출력
    # ---------------------------------------------------------
    print(f"=== 검사 결과 ({img_path}) ===\n")

    # [결과 1] 라벨이 없는 이미지 출력
    if imgs_missing_labels:
        print(f"🔴 [라벨 없음] 이미지는 있는데 라벨 파일이 없는 경우 ({len(imgs_missing_labels)}개):")
        for key in sorted(imgs_missing_labels):
            print(f"  - {img_map[key]}")
    else:
        print("✅ 모든 이미지에 라벨이 존재합니다.")

    print("-" * 50)

    # [결과 2] 이미지가 없는 라벨 출력
    if labels_missing_imgs:
        print(f"🔵 [이미지 없음] 라벨은 있는데 이미지 파일이 없는 경우 ({len(labels_missing_imgs)}개):")
        for key in sorted(labels_missing_imgs):
            print(f"  - {lbl_map[key]}")
    else:
        print("✅ 모든 라벨에 이미지가 존재합니다.")
        
    print("\n=== 검사 종료 ===")

# 실행
check_bidirectional_files(images_dir, labels_dir)