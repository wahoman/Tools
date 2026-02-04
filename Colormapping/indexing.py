import os
import sqlite3
import datetime

# --- 설정 (사용자 환경에 맞게 수정) ---
ROOT_DIRECTORY = r'\\SSTL_NAS\sstlabnas\1. Project\2. NIA\NIA'
DATABASE_FILE = 'raw_files.db'
TARGET_EXTENSIONS = ['.raw']

# ------------------------------------

def create_connection(db_file):
    """ SQLite 데이터베이스에 연결하고 connection 객체를 반환합니다. """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(f"데이터베이스 연결 오류: {e}")
    return conn

def create_table(conn):
    """ 'files' 테이블을 생성합니다. """
    sql = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL UNIQUE,
        filesize_bytes INTEGER,
        last_modified REAL,
        indexed_at TEXT NOT NULL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON files (filename);")
    except sqlite3.Error as e:
        print(f"테이블 생성 오류: {e}")

# <<< 변경점: count_target_files 함수가 필요 없으므로 삭제됨 >>>

# <<< 변경점: total_files 매개변수 제거 >>>
def index_files(conn, root_dir, extensions):
    """ 파일을 DB에 저장하고, 누적 처리 개수를 1000개 단위로 표시합니다. """
    cursor = conn.cursor()
    print(f"'{root_dir}' 에서 파일 인덱싱을 시작합니다...")
    
    newly_added_count = 0
    processed_count = 0
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(tuple(extensions)):
                continue

            processed_count += 1
            full_path = os.path.join(dirpath, filename)
            
            try:
                stat_info = os.stat(full_path)
                sql = '''
                INSERT OR IGNORE INTO files (filename, filepath, filesize_bytes, last_modified, indexed_at)
                VALUES (?, ?, ?, ?, ?)
                '''
                current_time = datetime.datetime.now().isoformat()
                cursor.execute(sql, (filename, full_path, stat_info.st_size, stat_info.st_mtime, current_time))
                
                if cursor.rowcount > 0:
                    newly_added_count += 1

            except Exception as e:
                print(f"  [오류] 파일 처리 중 문제 발생 ({full_path}): {e}")
            
            # <<< 변경점: 누적 개수만 표시 >>>
            if processed_count % 1000 == 0:
                print(f"  진행 상황: 현재까지 {processed_count}개 파일 처리됨...")

    conn.commit()
    print(f"\n인덱싱 완료! 총 {newly_added_count}개의 새 파일이 데이터베이스에 추가되었습니다.")

def search_file_path(db_file, filename_to_search):
    """ 데이터베이스에서 파일 이름으로 전체 경로를 검색합니다. """
    conn = create_connection(db_file)
    if conn is None: return []
    try:
        cursor = conn.cursor()
        sql = "SELECT filepath FROM files WHERE filename LIKE ?"
        cursor.execute(sql, (filename_to_search,))
        paths = [row[0] for row in cursor.fetchall()]
        return paths
    except sqlite3.Error as e:
        print(f"검색 오류: {e}")
        return []
    finally:
        if conn: conn.close()


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # <<< 변경점: 전체 개수 파악 단계 제거 >>>
    connection = create_connection(DATABASE_FILE)
    if connection:
        create_table(connection)
        
        # <<< 변경점: index_files 호출 방식 변경 >>>
        index_files(connection, ROOT_DIRECTORY, TARGET_EXTENSIONS)
        
        connection.close()
        
        print("\n--- 검색 예시 ---")
        search_term = 'DJI_0011.raw'
        found_paths = search_file_path(DATABASE_FILE, search_term)
        if found_paths:
            print(f"'{search_term}' 파일 검색 결과:")
            for path in found_paths: print(f" -> {path}")
        else:
            print(f"'{search_term}' 파일을 데이터베이스에서 찾을 수 없습니다.")