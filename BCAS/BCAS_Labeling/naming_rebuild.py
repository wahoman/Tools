
import os
import pandas as pd

# =========================================================
# [설정] 파일들이 있는 폴더 경로
# =========================================================
WORK_DIR = r'D:\hgyeo\BCAS\X-ray_Data_png'
OUTPUT_FILE = r'D:\hgyeo\BCAS\converted_file_list.csv'  # 결과로 나올 CSV 파일 이름

# =========================================================
# [역변환 규칙] 코드를 다시 원래 텍스트로 바꾸는 사전
# =========================================================
def decode_value(category, code):
    code = str(code).strip()
    
    # 1. Background (0~4 -> 원래 텍스트)
    if category == 'background':
        bg_map = {
            '0': 'Without bags',
            '1': 'Bags only',
            '2': 'Bags + clothes',
            '3': 'Bags + electronics',
            '4': 'Bags + clothes + electronics'
        }
        return bg_map.get(code, code) # 매칭 안 되면 코드 그대로 반환

    # 2. Orientation 2 (0/1 -> Laid Flat/Upright)
    elif category == 'ori2':
        ori2_map = {'0': 'Laid Flat', '1': 'Upright'}
        return ori2_map.get(code, code)

    # 3. Position (L/C/R -> Left/Center/Right)
    elif category == 'position':
        pos_map = {
            'L': 'Left', 'C': 'Center', 'R': 'Right',
            '0': 'Left', '1': 'Center', '2': 'Right' # 혹시 숫자로 된 경우 대비
        }
        return pos_map.get(code, code)

    # 4. Folding (0/1 -> X/O)
    elif category == 'folding':
        fold_map = {'0': 'X', '1': 'O'}
        return fold_map.get(code, code)

    return code

def create_decoded_csv():
    print("파일명 분석 및 상세정보 복원 시작...")
    
    files = os.listdir(WORK_DIR)
    data_list = []
    
    for filename in files:
        name, ext = os.path.splitext(filename)
        if name.startswith('.'): continue
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']: continue

        # 파일명 분해 ( _ 기준)
        parts = name.split('_')
        
        # 파일명 구조가 우리가 만든 13자리 형식인지 확인
        # 구조: [0]장비_[1]UID_[2]시나리오_[3]위협ID_[4]아이템_[5]샘플_[6]자세1_[7]자세2_[8]위치_[9]폴딩_[10]가방_[11]배경_[12]뷰
        if len(parts) == 13:
            
            # --- [여기서 코드를 텍스트로 복원합니다] ---
            # 1. Threat Item: 하이픈(-)을 다시 공백( )으로
            threat_item_restored = parts[4].replace("-", " ")
            
            # 2. 나머지 코드값 복원
            ori2_restored = decode_value('ori2', parts[7])       # 0 -> Laid Flat
            pos_restored = decode_value('position', parts[8])    # L -> Left
            fold_restored = decode_value('folding', parts[9])    # 0 -> X
            bg_restored = decode_value('background', parts[11])  # 0 -> Without bags
            
            # 3. 데이터 저장
            row = {
                'File Name': filename,
                'Equipment': parts[0],
                'UID': parts[1],
                'Scenario': parts[2],
                'Threat ID': parts[3],
                'Threat Item': threat_item_restored, # 공백 복원됨
                'Sample ID': parts[5],
                'Orientation 1': parts[6],
                'Orientation 2': ori2_restored,      # Laid Flat 등으로 저장
                'Position': pos_restored,            # Left 등으로 저장
                'Folding': fold_restored,            # X, O 로 저장
                'Bag ID': parts[10],
                'Background Item Type': bg_restored, # Without bags 등으로 저장
                'View No': parts[12]
            }
            data_list.append(row)

    # CSV 저장
    if data_list:
        df = pd.DataFrame(data_list)
        # 보기 좋게 컬럼 순서 정렬 (UID 먼저 나오게 등 원하면 조정 가능)
        cols = ['No', 'Scenario', 'Threat ID', 'Threat Item', 'Sample ID', 
                'Orientation 1', 'Orientation 2', 'Position', 'Folding', 
                'Bag ID', 'Background Item Type', 'UID', 'View No', 'File Name']
        
        # No 컬럼은 임의로 생성 (1, 2, 3...)
        df.insert(0, 'No', range(1, 1 + len(df)))
        
        # 컬럼 순서 재배치 (없는 컬럼은 무시하도록 안전장치)
        exist_cols = [c for c in cols if c in df.columns]
        df = df[exist_cols]

        save_path = os.path.join(WORK_DIR, OUTPUT_FILE)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[성공] 변환된 정보가 저장되었습니다.")
        print(f"파일 경로: {save_path}")
        print(f"총 {len(df)}개의 파일 정보를 복원했습니다.")
    else:
        print("처리할 파일이 없거나 파일명 형식이 맞지 않습니다.")

if __name__ == "__main__":
    create_decoded_csv()