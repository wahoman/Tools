import sys
import os
import json
import shutil
import time
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLabel, QLineEdit, 
                             QFileDialog, QProgressBar, QTextEdit, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal

# =========================================================
# [핵심 로직] 개별 파일 처리 Worker (다중 폴더 지원)
# =========================================================
def process_single_file(filename, input_dirs, dir_out):
    """
    여러 입력 폴더를 확인하여 파일을 복사하거나 병합합니다.
    """
    # 이 파일이 존재하는 모든 입력 폴더의 경로 수집
    paths_with_file = [os.path.join(d, filename) for d in input_dirs if os.path.exists(os.path.join(d, filename))]
    
    if not paths_with_file:
        return "SKIP"
        
    path_out = os.path.join(dir_out, filename)
    
    try:
        # CASE 1: 1개의 폴더에만 파일이 존재할 때 (단순 복사)
        if len(paths_with_file) == 1:
            shutil.copy(paths_with_file[0], path_out)
            return "COPY"
            
        # CASE 2: 2개 이상의 폴더에 파일이 존재할 때 (병합)
        else:
            # 첫 번째 파일을 기준으로 데이터 로드
            with open(paths_with_file[0], 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
                
            # 'shapes' 키가 없을 경우를 대비해 초기화
            if 'shapes' not in merged_data:
                merged_data['shapes'] = []
                
            # 나머지 파일들의 shapes 데이터를 가져와서 병합(extend)
            for path in paths_with_file[1:]:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data['shapes'].extend(data.get('shapes', []))
                    
            # 결과 저장
            with open(path_out, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)
                
            return "MERGED"
            
    except Exception as e:
        return f"ERROR: {filename} - {str(e)}"


# =========================================================
# [멀티프로세싱 스레드] GUI 멈춤 방지를 위한 별도 스레드
# =========================================================
class MergeThread(QThread):
    progress_update = pyqtSignal(int, int, str)  # 현재, 전체, 메시지
    finished = pyqtSignal(dict, float)           # 결과 통계, 소요 시간

    def __init__(self, input_dirs, output_dir):
        super().__init__()
        self.input_dirs = input_dirs
        self.output_dir = output_dir

    def run(self):
        start_time = time.time()
        
        # 1. 모든 입력 폴더에서 고유한 JSON 파일명 수집
        all_files = set()
        for d in self.input_dirs:
            if os.path.exists(d):
                all_files.update([f for f in os.listdir(d) if f.endswith('.json')])
        
        all_files = list(all_files)
        total_files = len(all_files)

        if total_files == 0:
            self.finished.emit({"MERGED": 0, "COPY": 0, "ERROR": 0}, 0)
            return

        # 2. 출력 폴더 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        stats = {"MERGED": 0, "COPY": 0, "ERROR": 0}
        num_cores = cpu_count()
        
        # 3. partial로 인자 고정
        worker_func = partial(process_single_file, input_dirs=self.input_dirs, dir_out=self.output_dir)

        # 4. 멀티프로세싱 실행
        with Pool(processes=num_cores) as pool:
            for i, res in enumerate(pool.imap_unordered(worker_func, all_files), 1):
                msg = ""
                if res.startswith("ERROR"):
                    stats["ERROR"] += 1
                    msg = res
                else:
                    stats[res] += 1
                
                # 진행 상황 전송 (100번 단위 또는 에러 발생 시, 마지막에 전송)
                if i % max(1, total_files // 100) == 0 or i == total_files or msg:
                    self.progress_update.emit(i, total_files, msg)

        end_time = time.time()
        self.finished.emit(stats, end_time - start_time)


# =========================================================
# [GUI UI 설정] PyQt5 메인 윈도우
# =========================================================
class LabelMergerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('JSON 라벨 다중 폴더 병합기 🚀')
        self.resize(600, 500)
        
        layout = QVBoxLayout()

        # 1. 입력 폴더 리스트
        layout.addWidget(QLabel("<b>📂 입력 폴더 목록 (병합할 폴더들을 추가하세요)</b>"))
        self.list_inputs = QListWidget()
        layout.addWidget(self.list_inputs)

        btn_layout_in = QHBoxLayout()
        self.btn_add_input = QPushButton("➕ 입력 폴더 추가")
        self.btn_add_input.clicked.connect(self.add_input_folder)
        self.btn_remove_input = QPushButton("➖ 선택 항목 삭제")
        self.btn_remove_input.clicked.connect(self.remove_input_folder)
        
        btn_layout_in.addWidget(self.btn_add_input)
        btn_layout_in.addWidget(self.btn_remove_input)
        layout.addLayout(btn_layout_in)

        # 2. 출력 폴더 설정
        layout.addWidget(QLabel("<b>📁 출력 폴더 (결과물이 저장될 곳)</b>"))
        out_layout = QHBoxLayout()
        self.txt_output = QLineEdit()
        self.btn_set_output = QPushButton("경로 설정")
        self.btn_set_output.clicked.connect(self.set_output_folder)
        
        out_layout.addWidget(self.txt_output)
        out_layout.addWidget(self.btn_set_output)
        layout.addLayout(out_layout)

        # 3. 진행 상황 로그
        layout.addWidget(QLabel("<b>📝 작업 로그</b>"))
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        layout.addWidget(self.log_console)

        # 4. 프로그레스 바 & 시작 버튼
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.btn_start = QPushButton("🚀 병합 시작!")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.btn_start.clicked.connect(self.start_merge)
        layout.addWidget(self.btn_start)

        self.setLayout(layout)

    def log(self, message):
        self.log_console.append(message)
        # 스크롤 맨 아래로 이동
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def add_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "입력 폴더 선택")
        if folder:
            # 중복 방지
            items = [self.list_inputs.item(i).text() for i in range(self.list_inputs.count())]
            if folder not in items:
                self.list_inputs.addItem(folder)

    def remove_input_folder(self):
        selected = self.list_inputs.currentRow()
        if selected >= 0:
            self.list_inputs.takeItem(selected)

    def set_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "출력 폴더 선택")
        if folder:
            self.txt_output.setText(folder)

    def start_merge(self):
        input_dirs = [self.list_inputs.item(i).text() for i in range(self.list_inputs.count())]
        output_dir = self.txt_output.text().strip()

        if len(input_dirs) < 2:
            QMessageBox.warning(self, "경고", "병합하려면 최소 2개 이상의 입력 폴더가 필요합니다.")
            return
        if not output_dir:
            QMessageBox.warning(self, "경고", "출력 폴더를 지정해주세요.")
            return

        self.log("="*50)
        self.log(f"🚀 작업을 시작합니다. (CPU 코어 {cpu_count()}개 사용)")
        
        # UI 비활성화
        self.btn_start.setEnabled(False)
        self.btn_add_input.setEnabled(False)
        self.btn_remove_input.setEnabled(False)
        self.progress_bar.setValue(0)

        # 스레드 실행
        self.thread = MergeThread(input_dirs, output_dir)
        self.thread.progress_update.connect(self.update_progress)
        self.thread.finished.connect(self.merge_finished)
        self.thread.start()

    def update_progress(self, current, total, msg):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)
        if msg:
            self.log(msg)

    def merge_finished(self, stats, time_taken):
        self.log("="*50)
        self.log(f"🎉 작업 완료! (소요 시간: {time_taken:.2f}초)")
        self.log(f" - 🧩 병합됨 (여러 폴더 중복) : {stats.get('MERGED', 0)}개")
        self.log(f" - 📄 복사됨 (단일 폴더 존재) : {stats.get('COPY', 0)}개")
        self.log(f" - ⚠️ 에러 발생 : {stats.get('ERROR', 0)}개")
        
        QMessageBox.information(self, "완료", "파일 병합 작업이 완료되었습니다!")
        
        # UI 활성화
        self.btn_start.setEnabled(True)
        self.btn_add_input.setEnabled(True)
        self.btn_remove_input.setEnabled(True)


if __name__ == "__main__":
    # 윈도우 멀티프로세싱 필수
    freeze_support()
    
    app = QApplication(sys.argv)
    ex = LabelMergerApp()
    ex.show()
    sys.exit(app.exec_())