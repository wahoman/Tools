import os
import csv
import sys
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QLabel, QMessageBox,
    QHeaderView, QAbstractItemView, QHBoxLayout
)
from PyQt5.QtCore import Qt

img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'} # webp 추가

# -----------------------------------------------------------------
# [개선 1] 정렬 로직을 별도 함수로 분리 (코드 중복 제거)
# -----------------------------------------------------------------
def get_natural_key(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

class NaturalSortItem(QTableWidgetItem):
    def __lt__(self, other):
        return get_natural_key(self.text()) < get_natural_key(other.text())

class ImageCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("클래스별 이미지 수 세기 (개선판)")
        self.resize(800, 600) # 창 크기 조금 더 키움

        self.layout = QVBoxLayout(self)

        # 상단 컨트롤 영역
        top_layout = QHBoxLayout()
        self.info_label = QLabel("📁 기준 폴더를 선택하세요 (train/valid 구조)")
        self.info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        top_layout.addWidget(self.info_label)
        
        self.select_btn = QPushButton("📂 기준 폴더 선택")
        self.select_btn.clicked.connect(self.select_base_folder)
        top_layout.addWidget(self.select_btn)
        self.layout.addLayout(top_layout)

        # 테이블 위젯
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['클래스 이름', 'Train', 'Valid', 'Total'])
        
        # [개선 2] 테이블 읽기 전용 설정 & 행 단위 선택
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        self.table.setSortingEnabled(True)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch) # 클래스명은 늘리기
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.layout.addWidget(self.table)

        # [개선 3] 전체 총합 표시 라벨 추가
        self.total_label = QLabel("총 이미지 수: 0장")
        self.total_label.setAlignment(Qt.AlignRight)
        self.total_label.setStyleSheet("color: blue; font-weight: bold; margin: 5px;")
        self.layout.addWidget(self.total_label)

        self.save_btn = QPushButton("💾 CSV로 저장")
        self.save_btn.clicked.connect(self.save_csv)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("height: 40px; font-size: 14px;")
        self.layout.addWidget(self.save_btn)

        self.base_path = ''
        self.class_stats = {}

    def select_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "기준 폴더 선택")
        if folder:
            self.base_path = folder
            self.info_label.setText(f"📁 {os.path.basename(folder)}")
            
            # [개선 4] 계산 중 모래시계 커서 표시
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                self.class_stats = self.count_all_classes()
                self.update_table()
                self.save_btn.setEnabled(True)
            finally:
                QApplication.restoreOverrideCursor()

    def count_all_classes(self):
        stats = {}
        for split in ['train', 'valid']:
            split_path = os.path.join(self.base_path, split)
            if not os.path.exists(split_path):
                continue
            
            # scandir이 listdir보다 대량의 파일 처리 시 빠름
            for entry in os.scandir(split_path):
                if entry.is_dir():
                    class_name = entry.name
                    class_path = entry.path
                    
                    # [개선 5] images 폴더가 없으면 클래스 폴더 자체를 카운트 (유연성)
                    images_folder = os.path.join(class_path, 'images')
                    target_dir = images_folder if os.path.exists(images_folder) else class_path
                    
                    count = 0
                    if os.path.exists(target_dir):
                        # 리스트 컴프리헨션 대신 제너레이터 사용하여 메모리 절약
                        count = sum(1 for f in os.listdir(target_dir) 
                                  if os.path.splitext(f)[1].lower() in img_exts)

                    if class_name not in stats:
                        stats[class_name] = {'train': 0, 'valid': 0}
                    stats[class_name][split] = count
        return stats

    def update_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        
        # [개선 1 활용] 공통된 정렬 키 함수 사용
        sorted_items = sorted(self.class_stats.items(), key=lambda x: get_natural_key(x[0]))

        grand_total = 0 # 전체 총합 계산용

        for row_idx, (class_name, counts) in enumerate(sorted_items):
            train = counts['train']
            valid = counts['valid']
            total = train + valid
            grand_total += total

            self.table.insertRow(row_idx)

            self.table.setItem(row_idx, 0, NaturalSortItem(class_name))

            # 숫자 데이터 설정 함수 (반복 줄이기)
            def set_num_item(col, val):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, val)
                item.setTextAlignment(Qt.AlignCenter) # 가운데 정렬
                self.table.setItem(row_idx, col, item)

            set_num_item(1, train)
            set_num_item(2, valid)
            set_num_item(3, total)

        # [개선 3 활용] 총합 라벨 업데이트
        self.total_label.setText(f"총 클래스: {len(sorted_items)}개 / 총 이미지: {grand_total:,}장")
        
        self.table.setSortingEnabled(True)

    def save_csv(self):
        output_csv = os.path.join(self.base_path, 'class_image_counts.csv')
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['class_name', 'train', 'valid', 'total'])
                
                sorted_items = sorted(self.class_stats.items(), key=lambda x: get_natural_key(x[0]))
                
                for class_name, counts in sorted_items:
                    train = counts['train']
                    valid = counts['valid']
                    total = train + valid
                    writer.writerow([class_name, train, valid, total])
                    
            QMessageBox.information(self, "저장 완료", f"CSV 저장 완료:\n{output_csv}")
        except PermissionError:
             QMessageBox.critical(self, "오류", "파일이 열려있습니다. 엑셀을 닫고 다시 시도해주세요.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"CSV 저장 실패: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # [옵션] 폰트 가독성 향상
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = ImageCounterApp()
    window.show()
    sys.exit(app.exec_())