import zipfile
import os

# 압축 파일 경로
zip_path = '/home/hgyeo/Desktop/APIDS.zip'
# 압축을 풀 경로
extract_dir = '/home/hgyeo/Desktop/APIDS'

# 압축 해제 실행
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"✅ 압축 해제 완료: {extract_dir}")
