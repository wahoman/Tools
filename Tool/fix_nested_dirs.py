import os
import shutil
from pathlib import Path

# 데이터 베이스 경로 설정
BASE_DIR = Path("/home/hgyeo/Desktop/NIA")

def fix_nested_folders():
    print(f"🚀 폴더 구조 정리 시작: {BASE_DIR}")
    
    # train, valid 순회
    for split in ["train", "valid"]:
        split_dir = BASE_DIR / split
        if not split_dir.exists(): continue
        
        # 각 클래스 폴더 순회
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir(): continue
            
            # 이중 폴더 확인 (예: A/A 가 있는지)
            nested_dir = class_dir / class_dir.name
            
            if nested_dir.exists() and nested_dir.is_dir():
                print(f"🔧 수정 중: {nested_dir} -> {class_dir}")
                
                # 내부의 images, labels 폴더를 상위로 이동
                for sub in nested_dir.iterdir():
                    src = sub
                    dst = class_dir / sub.name
                    
                    if dst.exists():
                        print(f"⚠️  이미 존재함 (병합): {dst}")
                        # 내용물 이동
                        for f in src.iterdir():
                            if not (dst / f.name).exists():
                                shutil.move(str(f), str(dst / f.name))
                    else:
                        shutil.move(str(src), str(dst))
                
                # 비어있는 내부 폴더 삭제
                try:
                    nested_dir.rmdir() 
                    print(f"🗑️  빈 폴더 삭제 완료: {nested_dir}")
                except:
                    print(f"⚠️  폴더가 비어있지 않아 삭제 실패: {nested_dir}")
            else:
                # 정상적인 경우 패스
                pass

    print("✅ 모든 폴더 정리가 완료되었습니다.")

if __name__ == "__main__":
    fix_nested_folders()