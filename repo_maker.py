import os
import shutil
import ast

# =========================================================
# [설정] 여기만 확인하세요
# =========================================================
# 1. 원본 소스 경로 (D 드라이브 전체)
source_root = "D:\\"  

# 2. 결과물이 저장될 경로 (바탕화면의 새 폴더)
dest_root = r"C:\Users\hgy84\Desktop\Git"

# 3. 제외할 폴더 목록 (Analyzer, TIPTool 등 포함)
IGNORE_DIRS = {
    'Analyzer', 'TIPTool', '$RECYCLE.BIN', 'System Volume Information',
    '__pycache__', '.git', '.idea', 'venv', 'env', 'node_modules',
    'print_images', 'images', 'dataset' # 데이터 폴더들도 제외
}
# =========================================================

def get_file_description(filepath):
    """파이썬 파일 맨 윗줄의 주석(Docstring)을 읽어오는 함수"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.split('\n')[0] # 첫 줄만 사용
    except:
        pass
    return "설명 없음"

def main():
    if not os.path.exists(dest_root):
        os.makedirs(dest_root)

    print(f"🚀 작업 시작: {source_root} -> {dest_root}")
    print("   (D드라이브를 스캔하여 파이썬 파일만 정리합니다...)")
    print("-" * 60)

    repo_structure = {}
    copy_count = 0

    # D드라이브 순회
    for root, dirs, files in os.walk(source_root):
        # 제외 폴더는 아예 진입하지 않음 (속도 향상)
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        # 파이썬 파일만 찾기
        py_files = [f for f in files if f.endswith('.py')]
        
        if not py_files:
            continue

        # 상대 경로 계산 (폴더 구조 유지)
        rel_path = os.path.relpath(root, source_root)
        target_dir = os.path.join(dest_root, rel_path)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # 파일 복사 및 정보 기록
        file_info_list = []
        for file in py_files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(target_dir, file)
            
            try:
                shutil.copy2(src_path, dst_path)
                copy_count += 1
                desc = get_file_description(src_path)
                file_info_list.append((file, desc))
            except Exception as e:
                print(f"⚠️ 복사 실패: {file} ({e})")
        
        if rel_path == ".": rel_path = "Root (최상위)"
        repo_structure[rel_path] = file_info_list

    # README.md 자동 생성
    readme_path = os.path.join(dest_root, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 📂 My Python Tools Collection\n\n")
        f.write("이 저장소는 로컬 툴들을 자동으로 정리하여 Git 관리용으로 만든 것입니다.\n")
        f.write(f"**자동 생성 시점:** {os.path.basename(dest_root)}\n\n")
        
        for folder, items in repo_structure.items():
            f.write(f"### 📁 {folder}\n")
            f.write("| 파일명 | 기능 설명 |\n")
            f.write("| :--- | :--- |\n")
            for filename, desc in items:
                f.write(f"| `{filename}` | {desc} |\n")
            f.write("\n")

    print("-" * 60)
    print(f"✨ 완료! 총 {copy_count}개의 파이썬 파일을 정리했습니다.")
    print(f"📂 결과 폴더: {dest_root}")

if __name__ == "__main__":
    main()