import sys
import os
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QListWidget, QHBoxLayout, QMessageBox, QRadioButton, QButtonGroup, QGridLayout
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QSizePolicy


class ModelSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("모델 결과 비교 뷰어 (폴더 관리 기능 추가)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setAcceptDrops(True)
        self.model_folders = []
        self.initUI()

    def initUI(self):
        # 안내 라벨
        lbl_info = QLabel("폴더를 드래그하거나 버튼을 사용하여 리스트를 구성하세요.")
        lbl_info.setAlignment(Qt.AlignCenter)
        lbl_info.setStyleSheet("color: #DDD; font-size: 14px; margin-bottom: 5px;")

        # 폴더 리스트 위젯
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection) # 다중 선택 가능
        self.folder_list.setStyleSheet("font-size: 14px; padding: 5px;")

        # === [수정] 폴더 관리 버튼 그룹 (가로 배치) ===
        btn_layout = QHBoxLayout()
        
        btn_add = QPushButton("➕ 폴더 추가")
        btn_remove = QPushButton("➖ 선택 삭제") # New
        btn_clear = QPushButton("🗑 전체 초기화") # New

        # 버튼 스타일
        btn_add.setStyleSheet("background-color: #4CAF50; font-weight: bold;")
        btn_remove.setStyleSheet("background-color: #FF9800; font-weight: bold;")
        btn_clear.setStyleSheet("background-color: #F44336; font-weight: bold;")

        btn_add.clicked.connect(self.add_folder)
        btn_remove.clicked.connect(self.remove_selected_folders)
        btn_clear.clicked.connect(self.clear_all_folders)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_clear)
        # ==========================================

        # 시작 버튼
        btn_start = QPushButton("🚀 비교 시작")
        btn_start.setStyleSheet("font-weight: bold; font-size: 16px; height: 45px; background-color: #007ACC; margin-top: 10px;")
        btn_start.clicked.connect(self.start_viewer)

        # 전체 레이아웃 조합
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        layout.addWidget(lbl_info)
        layout.addLayout(btn_layout) # 버튼들 추가
        layout.addWidget(self.folder_list)
        layout.addWidget(btn_start)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # --- 기능 함수들 ---
    
    def remove_selected_folders(self):
        """선택된 폴더들을 리스트에서 제거"""
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "알림", "삭제할 폴더를 선택해주세요.")
            return

        for item in selected_items:
            folder = item.text()
            # 데이터 리스트에서 제거
            if folder in self.model_folders:
                self.model_folders.remove(folder)
            # UI 리스트에서 제거
            self.folder_list.takeItem(self.folder_list.row(item))

    def clear_all_folders(self):
        """리스트 전체 초기화"""
        if not self.model_folders:
            return
            
        reply = QMessageBox.question(self, '확인', '모든 폴더 목록을 삭제하시겠습니까?', 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.model_folders.clear()
            self.folder_list.clear()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            folder_path = url.toLocalFile()
            if os.path.isdir(folder_path):
                if folder_path not in self.model_folders:
                    self.model_folders.append(folder_path)
                    self.folder_list.addItem(folder_path)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "모델 결과 폴더 선택")
        if folder and folder not in self.model_folders:
            self.model_folders.append(folder)
            self.folder_list.addItem(folder)

    # Delete 키로도 삭제 가능하게 유지
    def keyPressEvent(self, event):
        if self.folder_list.hasFocus() and event.key() == Qt.Key_Delete:
            self.remove_selected_folders()
        else:
            super().keyPressEvent(event)

    def start_viewer(self):
        n = len(self.model_folders)
        if n < 2:
            QMessageBox.warning(self, "경고", "비교를 위해 최소 2개 이상의 폴더가 필요합니다.")
            return
        
        self.viewer = CompareViewer(self.model_folders)
        self.viewer.back_signal.connect(self.restore_ui)
        self.viewer.show()
        self.hide()

    def restore_ui(self):
        self.show()
        self.viewer.close()
        self.viewer = None


class CompareViewer(QMainWindow):
    back_signal = pyqtSignal()

    def __init__(self, model_folders):
        super().__init__()
        self.model_folders = model_folders
        self.current_image_idx = 0
        self.current_original_pixmaps = [None] * len(model_folders)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_image)
        self.interval_ms = 3000

        # 첫 번째 폴더 기준 이미지 리스트
        first_folder = self.model_folders[0]
        if os.path.exists(first_folder):
            self.image_files = sorted([
                f for f in os.listdir(first_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ])
        else:
            self.image_files = []

        if not self.image_files:
            QMessageBox.warning(self, "오류", "첫 번째 폴더에 이미지가 없습니다.")

        self.initUI()
        self.load_current_images_from_disk()
        self.update_display()
        
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def initUI(self):
        self.setWindowTitle("모델 비교 뷰어")
        self.setGeometry(100, 100, 1600, 900)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # 상단 컨트롤
        top_layout = QHBoxLayout()
        
        btn_back = QPushButton("🔙 폴더 재선택")
        btn_back.setStyleSheet("background-color: #D32F2F; font-weight: bold;")
        btn_back.clicked.connect(self.go_back)
        top_layout.addWidget(btn_back)

        self.speed_group = QButtonGroup(self)
        radio1 = QRadioButton("1초"); self.speed_group.addButton(radio1, 1000)
        radio3 = QRadioButton("3초"); self.speed_group.addButton(radio3, 3000)
        radio5 = QRadioButton("5초"); self.speed_group.addButton(radio5, 5000)
        radio3.setChecked(True)
        radio1.toggled.connect(self.update_speed)
        radio3.toggled.connect(self.update_speed)
        radio5.toggled.connect(self.update_speed)

        top_layout.addSpacing(20)
        top_layout.addWidget(QLabel("속도:"))
        top_layout.addWidget(radio1)
        top_layout.addWidget(radio3)
        top_layout.addWidget(radio5)
        top_layout.addStretch(1)

        self.info_label = QLabel("Ready")
        self.info_label.setFont(QFont("Arial", 12, QFont.Bold))
        top_layout.addWidget(self.info_label)
        top_layout.addStretch(1)

        btn_prev = QPushButton('◀ 이전 (A)')
        btn_play = QPushButton('▶ 재생 (W)')
        btn_stop = QPushButton('■ 멈춤 (S)')
        btn_next = QPushButton('다음 ▶ (D)')

        btn_prev.clicked.connect(self.prev_image)
        btn_play.clicked.connect(self.start_timer)
        btn_stop.clicked.connect(self.stop_timer)
        btn_next.clicked.connect(self.next_image)

        top_layout.addWidget(btn_prev)
        top_layout.addWidget(btn_play)
        top_layout.addWidget(btn_stop)
        top_layout.addWidget(btn_next)

        self.main_layout.addLayout(top_layout)

        # 이미지 그리드
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(5)
        self.main_layout.addLayout(self.grid_layout)

        self.image_labels = []
        self.model_labels = []

        n = len(self.model_folders)
        cols = math.ceil(math.sqrt(n))
        
        for i in range(n):
            row = i // cols
            col = i % cols

            vbox = QVBoxLayout()
            
            folder_name = os.path.basename(self.model_folders[i])
            lbl_model = QLabel(f"[{i+1}] {folder_name}")
            lbl_model.setAlignment(Qt.AlignCenter)
            lbl_model.setStyleSheet("font-weight: bold; background-color: #444; padding: 4px; border-radius: 4px;")
            lbl_model.setFixedHeight(30)
            
            lbl_img = QLabel()
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setStyleSheet("background-color: #222; border: 1px solid #555;")
            lbl_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

            vbox.addWidget(lbl_model)
            vbox.addWidget(lbl_img)
            
            container = QWidget()
            container.setLayout(vbox)
            self.grid_layout.addWidget(container, row, col)

            self.model_labels.append(lbl_model)
            self.image_labels.append(lbl_img)

        for c in range(cols):
            self.grid_layout.setColumnStretch(c, 1)
        rows = math.ceil(n / cols)
        for r in range(rows):
            self.grid_layout.setRowStretch(r, 1)

    def go_back(self):
        self.stop_timer()
        self.back_signal.emit()

    def load_current_images_from_disk(self):
        if not self.image_files: return
        filename = self.image_files[self.current_image_idx]
        self.info_label.setText(f"{filename} ({self.current_image_idx + 1}/{len(self.image_files)})")

        for i, folder in enumerate(self.model_folders):
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                self.current_original_pixmaps[i] = QPixmap(path)
            else:
                self.current_original_pixmaps[i] = None

    def update_display(self):
        for i, pixmap in enumerate(self.current_original_pixmaps):
            label = self.image_labels[i]
            if pixmap and not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled_pixmap)
            else:
                label.setText("이미지 없음")

    def next_image(self):
        if not self.image_files: return
        self.current_image_idx = (self.current_image_idx + 1) % len(self.image_files)
        self.load_current_images_from_disk()
        self.update_display()

    def prev_image(self):
        if not self.image_files: return
        self.current_image_idx = (self.current_image_idx - 1) % len(self.image_files)
        self.load_current_images_from_disk()
        self.update_display()

    def start_timer(self):
        self.timer.start(self.interval_ms)

    def stop_timer(self):
        self.timer.stop()

    def update_speed(self):
        self.interval_ms = self.speed_group.checkedId()
        if self.timer.isActive():
            self.start_timer()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W: self.start_timer()
        elif event.key() == Qt.Key_S: self.stop_timer()
        elif event.key() == Qt.Key_A: self.prev_image()
        elif event.key() == Qt.Key_D: self.next_image()
        elif event.key() == Qt.Key_Escape: self.go_back()

    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QMainWindow { background-color: #333; color: white; }
        QLabel { color: white; }
        QPushButton { background-color: #555; color: white; border: 1px solid #777; padding: 6px; border-radius: 4px; }
        QPushButton:hover { background-color: #666; border-color: #999; }
        QPushButton:pressed { background-color: #777; }
        QListWidget { background-color: #444; color: white; border: 1px solid #666; font-size: 13px; }
        QRadioButton { color: white; }
    """)
    selector = ModelSelector()
    selector.show()
    sys.exit(app.exec_())