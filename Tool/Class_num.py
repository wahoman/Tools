import os
import csv

# 기준 경로 설정
base_path = '/home/hgyeo/Desktop/1217'
train_path = os.path.join(base_path, 'train')
valid_path = os.path.join(base_path, 'valid')

# 이미지 확장자 정의
img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 클래스별 이미지 수 저장용 딕셔너리
class_stats = {}

# 함수: 'images' 하위 폴더에서 이미지 수 세기
def count_images_in_subfolder(class_root_path):
    images_folder = os.path.join(class_root_path, 'images')
    if not os.path.exists(images_folder):
        return 0
    return len([
        f for f in os.listdir(images_folder)
        if os.path.splitext(f)[1].lower() in img_exts
    ])

# 'train'과 'valid' 각각 순회
for split in ['train', 'valid']:
    split_path = os.path.join(base_path, split)
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue
        count = count_images_in_subfolder(class_path)
        if class_name not in class_stats:
            class_stats[class_name] = {'train': 0, 'valid': 0}
        class_stats[class_name][split] = count

# CSV 저장
output_csv = os.path.join(base_path, '0710_normal.csv')
with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['class_name', 'train', 'valid', 'total'])
    for class_name, counts in sorted(class_stats.items()):
        train_count = counts['train']
        valid_count = counts['valid']
        total = train_count + valid_count
        writer.writerow([class_name, train_count, valid_count, total])

print(f"✅ CSV 저장 완료: {output_csv}")
