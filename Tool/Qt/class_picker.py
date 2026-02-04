import sys
import os
import zipfile
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QCheckBox, QLabel, QMessageBox, QProgressDialog,
    QLineEdit, QHBoxLayout, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt

class YoloCompressor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 클래스별 압축기 (Table View)")
        self.resize(600, 700) 

        self.base_dir = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 1. 폴더 선택 영역
        self.lbl_path = QLabel("선택된 경로: 없음")
        self.lbl_path.setStyleSheet("color: gray; font-size: 11px;")
        
        self.select_button = QPushButton("📂 YOLO 데이터셋 폴더 선택 (train/valid 상위)")
        self.select_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.select_button.clicked.connect(self.select_base_folder)
        
        main_layout.addWidget(self.select_button)
        main_layout.addWidget(self.lbl_path)

        # 2. 필터 및 전체 선택 영역
        control_layout = QHBoxLayout()
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 클래스명 검색...")
        self.search_bar.textChanged.connect(self.filter_classes)
        
        self.select_all_cb = QCheckBox("전체 선택")
        self.select_all_cb.stateChanged.connect(self.toggle_all_checkboxes)

        control_layout.addWidget(self.select_all_cb)
        control_layout.addWidget(self.search_bar)
        main_layout.addLayout(control_layout)

        # 3. 클래스 리스트 영역 (테이블 위젯 사용)
        group_box = QGroupBox("클래스 목록")
        group_layout = QVBoxLayout()
        
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["클래스 이름", "Train (장)", "Valid (장)"])
        
        # 테이블 스타일 설정
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers) # 수정 불가
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows) # 행 단위 선택
        self.table.verticalHeader().setVisible(False) # 행 번호 숨김
        
        # 컬럼 너비 조절
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch) # 이름 칸은 늘리기
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # 숫자는 내용만큼
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        # 체크박스 상태 변경 감지 (전체 선택 체크박스와 동기화용)
        self.table.itemChanged.connect(self.on_item_changed)

        group_layout.addWidget(self.table)
        group_box.setLayout(group_layout)
        main_layout.addWidget(group_box)

        # 4. 압축 버튼
        self.compress_button = QPushButton("🗜 선택한 클래스 압축하기 (.zip)")
        self.compress_button.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 10px;")
        self.compress_button.clicked.connect(self.compress_selected_classes)
        self.compress_button.setEnabled(False) 
        main_layout.addWidget(self.compress_button)

        self.setLayout(main_layout)

    def select_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "데이터셋 최상위 폴더 선택 (안에 train/valid가 있어야 함)")
        if folder:
            self.base_dir = Path(folder)
            train_dir = self.base_dir / "train"
            
            if not train_dir.exists():
                QMessageBox.critical(self, "오류", f"선택한 폴더 안에 'train' 폴더가 없습니다.\n경로: {self.base_dir}")
                self.lbl_path.setText("잘못된 경로")
                return
            
            self.lbl_path.setText(str(self.base_dir))
            self.load_classes(train_dir)
            self.compress_button.setEnabled(True)

    def load_classes(self, train_dir):
        """클래스 목록을 로드하고 테이블에 표시"""
        self.table.setRowCount(0) # 초기화
        
        class_dirs = [d for d in sorted(train_dir.iterdir()) if d.is_dir()]
        
        if not class_dirs:
            QMessageBox.warning(self, "주의", "'train' 폴더 안에 클래스 폴더가 없습니다.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor) # 로딩 중 커서 변경

        try:
            self.table.blockSignals(True) # 로딩 중 시그널 차단 (속도 향상)
            
            for row, class_folder in enumerate(class_dirs):
                cls_name = class_folder.name
                
                # 1) Train 개수
                train_img_dir = class_folder / "images"
                t_count = len(list(train_img_dir.glob("*.*"))) if train_img_dir.exists() else 0

                # 2) Valid 개수
                valid_img_dir = self.base_dir / "valid" / cls_name / "images"
                v_count = len(list(valid_img_dir.glob("*.*"))) if valid_img_dir.exists() else 0

                self.table.insertRow(row)

                # [컬럼 0] 클래스 이름 (체크박스 포함)
                item_name = QTableWidgetItem(cls_name)
                item_name.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item_name.setCheckState(Qt.Unchecked)
                self.table.setItem(row, 0, item_name)

                # [컬럼 1] Train 개수
                item_train = QTableWidgetItem(str(t_count))
                item_train.setTextAlignment(Qt.AlignCenter)
                item_train.setForeground(QBrush(QColor(0, 0, 255))) # 파란색
                self.table.setItem(row, 1, item_train)

                # [컬럼 2] Valid 개수
                item_valid = QTableWidgetItem(str(v_count))
                item_valid.setTextAlignment(Qt.AlignCenter)
                item_valid.setForeground(QBrush(QColor(0, 150, 0))) # 초록색
                self.table.setItem(row, 2, item_valid)

            self.table.blockSignals(False)
            
        finally:
            QApplication.restoreOverrideCursor()

        self.select_all_cb.setChecked(False)

    def filter_classes(self, text):
        """검색어에 따라 행 숨기기"""
        text = text.lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if text in item.text().lower():
                self.table.setRowHidden(row, False)
            else:
                self.table.setRowHidden(row, True)

    def toggle_all_checkboxes(self, state):
        """전체 선택/해제"""
        self.table.blockSignals(True) # 시그널 루프 방지
        for row in range(self.table.rowCount()):
            # 숨겨진 행은 제외할지 결정 (여기선 보이는 것만 선택하도록 설정)
            if not self.table.isRowHidden(row):
                item = self.table.item(row, 0)
                item.setCheckState(Qt.Checked if state == Qt.Checked else Qt.Unchecked)
        self.table.blockSignals(False)

    def on_item_changed(self, item):
        """개별 아이템 체크 시 전체선택 박스 상태 업데이트"""
        # 체크박스 컬럼(0번)이 아니면 무시
        if item.column() != 0: return
        
        # 전체 행 검사
        total_visible = 0
        checked_count = 0
        
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                total_visible += 1
                if self.table.item(row, 0).checkState() == Qt.Checked:
                    checked_count += 1
        
        self.select_all_cb.blockSignals(True)
        if total_visible > 0 and checked_count == total_visible:
            self.select_all_cb.setCheckState(Qt.Checked)
        elif checked_count == 0:
            self.select_all_cb.setCheckState(Qt.Unchecked)
        else:
            self.select_all_cb.setCheckState(Qt.PartiallyChecked)
        self.select_all_cb.blockSignals(False)

    def compress_selected_classes(self):
        if not self.base_dir: return

        selected_classes = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.checkState() == Qt.Checked:
                selected_classes.append(item.text())
        
        if not selected_classes:
            QMessageBox.warning(self, "경고", "압축할 클래스를 최소 하나 이상 선택해주세요.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "ZIP 파일 저장", "", "ZIP Files (*.zip)")
        if not save_path:
            return

        # 압축 로직 (기존과 동일)
        files_to_zip = []
        target_splits = ["train", "valid", "test"] 

        for split in target_splits:
            for cls in selected_classes:
                cls_dir = self.base_dir / split / cls
                if cls_dir.exists():
                    for root, dirs, files in os.walk(cls_dir):
                        for file in files:
                            file_path = Path(root) / file
                            files_to_zip.append(file_path)

        total_files = len(files_to_zip)
        if total_files == 0:
            QMessageBox.information(self, "알림", "선택한 클래스 경로에 파일이 존재하지 않습니다.")
            return

        progress = QProgressDialog("파일 압축 중...", "취소", 0, total_files, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for i, file_path in enumerate(files_to_zip):
                    if progress.wasCanceled():
                        zipf.close()
                        os.remove(save_path)
                        return

                    arcname = file_path.relative_to(self.base_dir)
                    zipf.write(file_path, arcname)
                    progress.setValue(i + 1)
            
            QMessageBox.information(self, "성공", f"압축이 완료되었습니다!\n파일 수: {total_files}개")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"압축 중 오류가 발생했습니다:\n{str(e)}")
        finally:
            progress.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = YoloCompressor()
    window.show()
    sys.exit(app.exec_())