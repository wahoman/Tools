

import shutil
from pathlib import Path

src_dir  = Path('/home/hgyeo/Desktop/results')
zip_path = src_dir.with_suffix(".zip")   # best.zip

# 기존 zip 있으면 덮어쓰기cd
shutil.make_archive(base_name=zip_path.with_suffix(""),
                    format="zip",
                    root_dir=src_dir)

print(f"✅ 압축 완료: {zip_path}")
