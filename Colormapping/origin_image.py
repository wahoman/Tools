import os
import shutil

# 1. 경로 설정
# 원본 실사 이미지가 있는 소스 경로
source_originals_root = r"\\SSTL_NAS\sstlabnas\1. Project\2. NIA\NIA\개별실사&물품DB"
# 복사할 대상의 루트 경로
dest_root = r"D:\hgyeo\image_data"

# 2. 작업 대상 카테고리 목록
categories = [
    "Bullet"
]

total_copied_files = 0

# 3. [최적화] 소스 폴더를 "한 번만" 스캔하여 모든 파일 목록을 메모리에 저장
print(f"🔍 원본 이미지 소스 폴더를 스캔합니다: {source_originals_root}")
print("(파일이 많으면 시간이 다소 걸릴 수 있습니다...)")
try:
    all_original_files = os.listdir(source_originals_root)
    print(f"✅ {len(all_original_files)}개의 원본 파일 목록을 확인했습니다.")
except FileNotFoundError:
    print(f"❌ 치명적 에러: 원본 이미지 소스 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
    all_original_files = [] # 에러 발생 시 빈 리스트로 초기화하여 프로그램 중단 방지
except Exception as e:
    print(f"❌ 치명적 에러: 소스 경로 접근 중 문제 발생 - {e}")
    all_original_files = []

# 스캔한 파일이 있을 경우에만 다음 단계 진행
if all_original_files:
    # 4. 각 카테고리별로 작업 시작
    for category in categories:
        try:
            # 기준이 될 x-ray 이미지 폴더 경로
            xray_folder = os.path.join(dest_root, category, "x-ray 이미지")
            # 원본 이미지를 복사할 최종 목적지 폴더 경로
            dest_original_folder = os.path.join(dest_root, category, "원본이미지")

            if not os.path.isdir(xray_folder):
                continue # x-ray 폴더가 없으면 건너뛰기
            
            os.makedirs(dest_original_folder, exist_ok=True)
            
            print(f"\n📂 [{category}] 폴더의 원본 이미지 검색 및 복사를 시작합니다...")
            
            # 5. x-ray 폴더 안의 각 파일을 기준으로 ID 추출
            for xray_filename in os.listdir(xray_folder):
                stem = os.path.splitext(xray_filename)[0]
                parts = stem.split('_')
                
                if len(parts) < 2:
                    continue # 파일 이름 형식이 맞지 않으면 건너뛰기
                
                # ID 추출 (예: '..._073-004_...')
                image_id = parts[-2]

                # 6. 미리 스캔해둔 전체 원본 파일 목록에서 ID가 포함된 파일 검색
                found_match = False
                for original_filename in all_original_files:
                    if image_id in original_filename:
                        
                        source_path = os.path.join(source_originals_root, original_filename)
                        dest_path = os.path.join(dest_original_folder, original_filename)

                        if not os.path.exists(dest_path):
                            shutil.copy2(source_path, dest_path)
                            print(f"  ✅ 복사: {original_filename}")
                            total_copied_files += 1
                        
                        found_match = True
                        break # 일치하는 파일을 찾았으면 더 이상 찾지 않고 다음 x-ray 파일로 넘어감

                if not found_match:
                    print(f"  🟡 경고: ID '{image_id}'에 해당하는 원본 이미지를 소스 폴더에서 찾지 못했습니다.")

        except Exception as e:
            print(f"❌ 에러 발생: [{category}] 폴더 처리 중 문제 발생 - {e}")

print(f"\n\n🎉 모든 작업 완료! 총 {total_copied_files}개의 원본 파일을 복사했습니다.")