import os
import pandas as pd
import io

# =========================================================
# [1] 설정
# =========================================================
WORK_DIR = r'\\Sstl_nas\ai\5. BCAS_Labeling\BCAS_Labeling\DAY1\images'
EQUIPMENT_NAME = "E3S690G3"

# =========================================================
# [2] 데이터 입력
# =========================================================
# 엑셀 데이터를 여기에 복사해서 붙여넣으세요.
RAW_DATA = """
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	B	0	x	0	x	x	x	X	1	1		126313
2	B	0	x	0	x	x	x	X	2	1		126314
3	B	0	x	0	x	x	x	X	3	1		126323
4	B	0	x	0	x	x	x	X	4	1		126316
5	B	0	x	0	x	x	x	X	5	1		126317
6	B	0	x	0	x	x	x	X	6	1		126318
7	B	0	x	0	x	x	x	X	7	1		126319
8	B	0	x	0	x	x	x	X	8	1		126320
9	B	0	x	0	x	x	x	X	9	1		126321
10	B	0	x	0	x	x	x	X	10	1		126322
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	B	0	x	0	x	x	x	X	1	1		126324
2	B	0	x	0	x	x	x	X	2	1		126325
3	B	0	x	0	x	x	x	X	3	1		126326
4	B	0	x	0	x	x	x	X	4	1		126327
5	B	0	x	0	x	x	x	X	5	1		126328
6	B	0	x	0	x	x	x	X	6	1		126329
7	B	0	x	0	x	x	x	X	7	1		126330
8	B	0	x	0	x	x	x	X	8	1		126331
9	B	0	x	0	x	x	x	X	9	1		126332
10	B	0	x	0	x	x	x	X	10	1		126333
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	2		126348
2	C	0	x	0	x	x	x	X	2	2		126349
3	C	0	x	0	x	x	x	X	3	2		126350
4	C	0	x	0	x	x	x	X	4	2		126351
5	C	0	x	0	x	x	x	X	5	2		126352
6	C	0	x	0	x	x	x	X	6	2		126353
7	C	0	x	0	x	x	x	X	7	2		126354
8	C	0	x	0	x	x	x	X	8	2		126355
9	C	0	x	0	x	x	x	X	9	2		126356
10	C	0	x	0	x	x	x	X	10	2		126357
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	2		126358
2	C	0	x	0	x	x	x	X	2	2		126359
3	C	0	x	0	x	x	x	X	3	2		126360
4	C	0	x	0	x	x	x	X	4	2		126361
5	C	0	x	0	x	x	x	X	5	2		126362
6	C	0	x	0	x	x	x	X	6	2		126363
7	C	0	x	0	x	x	x	X	7	2		126364
8	C	0	x	0	x	x	x	X	8	2		126365
9	C	0	x	0	x	x	x	X	9	2		126366
10	C	0	x	0	x	x	x	X	10	2		126367
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	2		126368
2	C	0	x	0	x	x	x	X	2	2		126369
3	C	0	x	0	x	x	x	X	3	2		126370
4	C	0	x	0	x	x	x	X	4	2		126371
5	C	0	x	0	x	x	x	X	5	2		126372
6	C	0	x	0	x	x	x	X	6	2		126373
7	C	0	x	0	x	x	x	X	7	2		126374
8	C	0	x	0	x	x	x	X	8	2		126375
9	C	0	x	0	x	x	x	X	9	2		126376
10	C	0	x	0	x	x	x	X	10	2		126377
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	2		126389
2	C	0	x	0	x	x	x	X	2	2		126390
3	C	0	x	0	x	x	x	X	3	2		126391
4	C	0	x	0	x	x	x	X	4	2		126392
5	C	0	x	0	x	x	x	X	5	2		126393
6	C	0	x	0	x	x	x	X	6	2		126394
7	C	0	x	0	x	x	x	X	7	2		126395
8	C	0	x	0	x	x	x	X	8	2		126396
9	C	0	x	0	x	x	x	X	9	2		126399
10	C	0	x	0	x	x	x	X	10	2		126400
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	2		126401
2	C	0	x	0	x	x	x	X	2	2		126402
3	C	0	x	0	x	x	x	X	3	2		126403
4	C	0	x	0	x	x	x	X	4	2		126404
5	C	0	x	0	x	x	x	X	5	2		126405
6	C	0	x	0	x	x	x	X	6	2		126406
7	C	0	x	0	x	x	x	X	7	2		126407
8	C	0	x	0	x	x	x	X	8	2		126408
9	C	0	x	0	x	x	x	X	9	2		126409
10	C	0	x	0	x	x	x	X	10	2		126410
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	3		126411
2	C	0	x	0	x	x	x	X	2	3		126412
3	C	0	x	0	x	x	x	X	3	3		126413
4	C	0	x	0	x	x	x	X	4	3		126414
5	C	0	x	0	x	x	x	X	5	3		126415
6	C	0	x	0	x	x	x	X	6	3		126416
7	C	0	x	0	x	x	x	X	7	3		126417
8	C	0	x	0	x	x	x	X	8	3		126418
9	C	0	x	0	x	x	x	X	9	3		126419
10	C	0	x	0	x	x	x	X	10	3		126420
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	3		126421
2	C	0	x	0	x	x	x	X	2	3		126422
3	C	0	x	0	x	x	x	X	3	3		126423
4	C	0	x	0	x	x	x	X	4	3		126424
5	C	0	x	0	x	x	x	X	5	3		126425
6	C	0	x	0	x	x	x	X	6	3		126426
7	C	0	x	0	x	x	x	X	7	3		126427
8	C	0	x	0	x	x	x	X	8	3		126428
9	C	0	x	0	x	x	x	X	9	3		126429
10	C	0	x	0	x	x	x	X	10	3		126430
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	3		126431
2	C	0	x	0	x	x	x	X	2	3		126432
3	C	0	x	0	x	x	x	X	3	3		126433
4	C	0	x	0	x	x	x	X	4	3		126434
5	C	0	x	0	x	x	x	X	5	3		126435
6	C	0	x	0	x	x	x	X	6	3		126436
7	C	0	x	0	x	x	x	X	7	3		126437
8	C	0	x	0	x	x	x	X	8	3		126438
9	C	0	x	0	x	x	x	X	9	3		126439
10	C	0	x	0	x	x	x	X	10	3		126440
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	3		126441
2	C	0	x	0	x	x	x	X	2	3		126442
3	C	0	x	0	x	x	x	X	3	3		126443
4	C	0	x	0	x	x	x	X	4	3		126444
5	C	0	x	0	x	x	x	X	5	3		126445
6	C	0	x	0	x	x	x	X	6	3		126446
7	C	0	x	0	x	x	x	X	7	3		126447
8	C	0	x	0	x	x	x	X	8	3		126448
9	C	0	x	0	x	x	x	X	9	3		126449
10	C	0	x	0	x	x	x	X	10	3		126450
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	3		126451
2	C	0	x	0	x	x	x	X	2	3		126452
3	C	0	x	0	x	x	x	X	3	3		126453
4	C	0	x	0	x	x	x	X	4	3		126454
5	C	0	x	0	x	x	x	X	5	3		126455
6	C	0	x	0	x	x	x	X	6	3		126456
7	C	0	x	0	x	x	x	X	7	3		126457
8	C	0	x	0	x	x	x	X	8	3		126458
9	C	0	x	0	x	x	x	X	9	3		126459
10	C	0	x	0	x	x	x	X	10	3		126460
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	4		126461
2	C	0	x	0	x	x	x	X	2	4		126462
3	C	0	x	0	x	x	x	X	3	4		126463
4	C	0	x	0	x	x	x	X	4	4		126464
5	C	0	x	0	x	x	x	X	5	4		126465
6	C	0	x	0	x	x	x	X	6	4		126466
7	C	0	x	0	x	x	x	X	7	4		126467
8	C	0	x	0	x	x	x	X	8	4		126468
9	C	0	x	0	x	x	x	X	9	4		126469
10	C	0	x	0	x	x	x	X	10	4		126470
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	4		126471
2	C	0	x	0	x	x	x	X	2	4		126472
3	C	0	x	0	x	x	x	X	3	4		126473
4	C	0	x	0	x	x	x	X	4	4		126474
5	C	0	x	0	x	x	x	X	5	4		126475
6	C	0	x	0	x	x	x	X	6	4		126476
7	C	0	x	0	x	x	x	X	7	4		126477
8	C	0	x	0	x	x	x	X	8	4		126478
9	C	0	x	0	x	x	x	X	9	4		126479
10	C	0	x	0	x	x	x	X	10	4		126480
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	4		126481
2	C	0	x	0	x	x	x	X	2	4		126482
3	C	0	x	0	x	x	x	X	3	4		126483
4	C	0	x	0	x	x	x	X	4	4		126484
5	C	0	x	0	x	x	x	X	5	4		126485
6	C	0	x	0	x	x	x	X	6	4		126486
7	C	0	x	0	x	x	x	X	7	4		126487
8	C	0	x	0	x	x	x	X	8	4		126488
9	C	0	x	0	x	x	x	X	9	4		126489
10	C	0	x	0	x	x	x	X	10	4		126490
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	4		126491
2	C	0	x	0	x	x	x	X	2	4		126492
3	C	0	x	0	x	x	x	X	3	4		126493
4	C	0	x	0	x	x	x	X	4	4		126494
5	C	0	x	0	x	x	x	X	5	4		126495
6	C	0	x	0	x	x	x	X	6	4		126496
7	C	0	x	0	x	x	x	X	7	4		126497
8	C	0	x	0	x	x	x	X	8	4		126498
9	C	0	x	0	x	x	x	X	9	4		126499
10	C	0	x	0	x	x	x	X	10	4		126500
No	Scenario	Threat ID	Threat Item	Sample ID	Orientation 1	Orientation 2	Position	Folding	Bag ID	Background Item Type	PhotoID	UID
1	C	0	x	0	x	x	x	X	1	4		126501
2	C	0	x	0	x	x	x	X	2	4		126502
3	C	0	x	0	x	x	x	X	3	4		126503
4	C	0	x	0	x	x	x	X	4	4		126504
5	C	0	x	0	x	x	x	X	5	4		126505
6	C	0	x	0	x	x	x	X	6	4		126506
7	C	0	x	0	x	x	x	X	7	4		126507
8	C	0	x	0	x	x	x	X	8	4		126508
9	C	0	x	0	x	x	x	X	9	4		126509
10	C	0	x	0	x	x	x	X	10	4		126510
 """

# =========================================================
# [3] 내부 로직
# =========================================================
def get_code_from_value(column_name, value):
    val_str = str(value).strip()
    
    if column_name == 'Position': return val_str 
    elif column_name == 'Folding':
        if val_str == 'X': return "0"
        elif val_str == 'O': return "1"
        return "0"
    elif column_name == 'Orientation 2':
        if "Upright" in val_str: return "1"
        return "0" 
    elif column_name == 'Background Item Type': return val_str
    return val_str

def run_renaming():
    print("데이터 분석 중...")
    try:
        # RAW_DATA가 비어있으면 에러가 나므로 체크
        if not RAW_DATA.strip():
            print("[오류] RAW_DATA가 비어있습니다. [2] 데이터 입력란에 엑셀 내용을 붙여넣어주세요.")
            return

        df = pd.read_csv(io.StringIO(RAW_DATA), sep='\t')
        df = df[df['UID'] != 'UID'] 
        df = df.dropna(subset=['UID'])
        
        # UID 8자리 맞추기 (00 채우기)
        df['UID'] = df['UID'].astype(str).replace(r'\.0$', '', regex=True).str.strip()
        df['UID'] = df['UID'].apply(lambda x: x.zfill(8))
        
    except Exception as e:
        print(f"[오류] 데이터 로드 실패: {e}")
        return

    files = os.listdir(WORK_DIR)
    count = 0
    print(f"\n총 {len(df)}개 데이터 로드됨 (UID 00 채움 완료). 변경 시작...\n")

    for filename in files:
        name, ext = os.path.splitext(filename)
        if name.startswith('.'): continue

        # [★수정된 부분★] 파일명 처리를 시작하기 전에 '_PH' 제거
        # 예: "00126512_0_PH" -> "00126512_0" 으로 변경하여 아래 로직이 인식 가능하게 함
        if name.endswith('_PH'):
            name = name[:-3] # 뒤에서 3글자(_PH) 제거

        try:
            if '_' in name:
                # _PH를 제거했으므로 이제 "00126512_0" 형태가 되어 정상 분리됨
                file_uid = name.split('_')[0] 
                view_no = name.split('_')[-1]
            else:
                file_uid = name
                view_no = "1"
            
            # 엑셀 데이터 매칭
            row = df[df['UID'] == file_uid]
            
            if not row.empty:
                data = row.iloc[0]
                
                scenario = str(data['Scenario']).strip()
                threat_id = str(data['Threat ID']).strip()
                threat_item = str(data['Threat Item']).strip().replace(" ", "-")
                sample_id = str(data['Sample ID']).strip()
                ori1 = str(data['Orientation 1']).strip()
                ori2 = get_code_from_value('Orientation 2', data['Orientation 2'])
                pos = get_code_from_value('Position', data['Position'])
                folding = get_code_from_value('Folding', data['Folding'])
                bag_id = str(data['Bag ID']).strip()
                bg = get_code_from_value('Background Item Type', data['Background Item Type'])
                
                new_name = (
                    f"{EQUIPMENT_NAME}_{file_uid}_{scenario}_{threat_id}_{threat_item}_{sample_id}_"
                    f"{ori1}_{ori2}_{pos}_{folding}_{bag_id}_{bg}_{view_no}{ext}"
                )
                
                old_path = os.path.join(WORK_DIR, filename)
                new_path = os.path.join(WORK_DIR, new_name)
                
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"[변경] {filename} -> {new_name}")
                    count += 1
            else:
                pass
                
        except Exception as e:
            print(f"[에러] {filename}: {e}")

    print(f"\n작업 종료: 총 {count}개 변경됨")

if __name__ == "__main__":
    run_renaming()