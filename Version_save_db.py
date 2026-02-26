import sqlite3
import os
import time
from pathlib import Path

# ==========================================
# 1. 설정
# ==========================================
TARGET_TRAIN_FOLDER = "/home/hgyeo/Desktop/Origin_cluster_base_folder/Scissors_done/data_origin"
DB_FILE_NAME = "/home/hgyeo/Desktop/Origin_cluster_base_folder/Scissors_done/Scissors_Origin_dataset.db"

# 한 번에 DB에 밀어 넣을 데이터 묶음 크기 (메모리와 속도 조절)
# 텍스트가 엄청 길다면 1000~2000 정도가 적당, 짧다면 10000 추천
BATCH_SIZE = 2000 

class OptimizedDatasetSaver:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # [핵심 1] 대용량 처리를 위한 SQLite 속도 최적화 옵션
        self.conn.execute('PRAGMA journal_mode = WAL;')  # 쓰기 속도 대폭 향상
        self.conn.execute('PRAGMA synchronous = NORMAL;') # 안정성 vs 속도 타협
        self.conn.execute('PRAGMA cache_size = 10000;')   # 캐시 메모리 확보
        
        self._init_table()

    def _init_table(self):
        # 파일명이나 경로가 길어도 TEXT 타입은 10억 자까지 저장 가능하므로 문제없음
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relative_path TEXT,
                filename TEXT,
                label_content TEXT
            )
        ''')
        self.conn.commit()

    def save_snapshot(self, target_folder):
        target_path = Path(target_folder)
        
        if not target_path.exists():
            print(f"❌ 경로 없음: {target_path}")
            return

        print(f"🚀 [고성능 모드] 저장 시작...")
        print(f"   📂 대상: {target_folder}")
        
        buffer = [] # 데이터를 묶어두는 임시 창고
        total_count = 0
        start_time = time.time()

        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_abs_path = Path(root) / file
                    
                    try:
                        # 1. 상대 경로 (폴더 구조)
                        relative_path = str(img_abs_path.parent.relative_to(target_path))
                        
                        # 2. 라벨 읽기 (긴 텍스트 처리)
                        label_content = ""
                        parts = list(img_abs_path.parts)
                        parts_lower = [p.lower() for p in parts]
                        
                        if 'images' in parts_lower:
                            idx = len(parts) - 1 - parts_lower[::-1].index('images')
                            parts[idx] = 'labels'
                            label_path = Path(*parts).with_suffix('.txt')
                            
                            if label_path.exists():
                                # errors='ignore': 엄청 긴 파일 읽다가 특수문자 에러나면 무시하고 계속 진행
                                with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    label_content = f.read()

                        # 3. 버퍼에 추가 (DB에 바로 안 넣음)
                        # 튜플 형태로 (경로, 파일명, 내용) 저장
                        buffer.append((relative_path, file, label_content))
                        
                        # 4. [핵심 2] 버퍼가 꽉 차면 한방에 DB 투입 (Bulk Insert)
                        if len(buffer) >= BATCH_SIZE:
                            self._flush_buffer(buffer)
                            total_count += len(buffer)
                            buffer = [] # 버퍼 비우기 (메모리 해제)
                            
                            # 진행 상황 출력
                            elapsed = time.time() - start_time
                            speed = total_count / elapsed
                            print(f"▶ {total_count}개 저장 중... (속도: {speed:.1f}개/초)")

                    except Exception as e:
                        # 파일명이 너무 길어서 OS가 못 읽는 경우 등 예외 처리
                        print(f"⚠️ 스킵됨 ({file}): {e}")

        # 5. 남은 데이터 처리 (마지막 찌꺼기)
        if buffer:
            self._flush_buffer(buffer)
            total_count += len(buffer)

        self.conn.commit()
        print("-" * 50)
        print(f"✅ 저장 완료!")
        print(f"총 데이터: {total_count}개")
        print(f"소요 시간: {time.time() - start_time:.1f}초")

    def _flush_buffer(self, data_list):
        """executemany를 사용해 데이터를 한 번에 밀어 넣음"""
        try:
            self.cursor.executemany('''
                INSERT INTO dataset (relative_path, filename, label_content)
                VALUES (?, ?, ?)
            ''', data_list)
            self.conn.commit() # 중간 저장 (혹시 튕겨도 여기까지는 저장됨)
        except sqlite3.OperationalError as e:
            print(f"❌ DB 저장 중 에러: {e}")
            # 너무 큰 텍스트 때문에 에러나면 여기서 처리 가능

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    saver = OptimizedDatasetSaver(DB_FILE_NAME)
    saver.save_snapshot(TARGET_TRAIN_FOLDER)
    saver.close()