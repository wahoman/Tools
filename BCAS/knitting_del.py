import os
import json
import glob
from multiprocessing import Pool, cpu_count
from functools import partial

# =========================================================
# [설정] 작업할 JSON 폴더 경로
# =========================================================
json_folder = r"C:\Users\hgy84\Desktop\BCAS\BCAS_Labeling\DAY13-2\object_json_labels"

# 삭제하고 싶은 라벨 명칭
TARGET_LABEL = "Matchbox-B"
# =========================================================

def process_single_file(file_path, target_label):
    """
    개별 JSON 파일을 처리하는 Worker 함수 (멀티프로세싱 용도)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        original_shape_count = len(data.get("shapes", []))
        
        # 리스트 컴프리헨션을 사용하여 필터링
        new_shapes = [shape for shape in data.get("shapes", []) if shape.get("label") != target_label]
        
        removed_in_this_file = original_shape_count - len(new_shapes)

        # 변경 사항이 있을 때만 파일 덮어쓰기
        if removed_in_this_file > 0:
            data["shapes"] = new_shapes
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # (수정됨 여부, 삭제된 개수, 파일명, 에러 메시지) 반환
            return True, removed_in_this_file, os.path.basename(file_path), None
        
        return False, 0, os.path.basename(file_path), None

    except Exception as e:
        return False, 0, os.path.basename(file_path), str(e)


def remove_specific_label_multiprocessing():
    print(f"📂 작업 경로: {json_folder}")
    
    # 1. 해당 폴더의 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    
    if not json_files:
        print("❌ 처리할 JSON 파일이 없습니다. 경로를 다시 확인해주세요.")
        return

    modified_count = 0
    total_removed_shapes = 0

    # 2. CPU 코어 수 확인 및 Pool 생성
    num_cores = cpu_count()
    print(f"🚀 멀티프로세싱 시작 (사용 코어 수: {num_cores}개, 대상 파일: {len(json_files)}개)")

    # process_single_file 함수에 target_label 인자를 고정
    worker_func = partial(process_single_file, target_label=TARGET_LABEL)

    # 3. 멀티프로세싱 실행
    # imap_unordered를 사용하여 처리되는 대로 즉시 결과를 반환받음
    with Pool(processes=num_cores) as pool:
        for is_modified, removed_count, file_name, error_msg in pool.imap_unordered(worker_func, json_files):
            if error_msg:
                print(f"❌ 에러 발생 ({file_name}): {error_msg}")
            elif is_modified:
                modified_count += 1
                total_removed_shapes += removed_count
                print(f"✅ 수정 완료: {file_name} ({removed_count}개 삭제됨)")

    print("\n" + "="*50)
    print(f"🎉 작업 완료!")
    print(f"📊 수정된 파일 수: {modified_count}개")
    print(f"🗑️ 삭제된 총 라벨(shape) 수: {total_removed_shapes}개")
    print("="*50)

if __name__ == "__main__":
    # Windows 환경에서 멀티프로세싱을 안전하게 실행하기 위해 필수적인 구문입니다.
    remove_specific_label_multiprocessing()