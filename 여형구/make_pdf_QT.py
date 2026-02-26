import sys
import os
import re  # [추가] 정규표현식 모듈 (숫자 추출용)
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PIL import Image

class PngToPdfConverter(QWidget):
    def __init__(self):
        super().__init__()
        self.source_folder = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('PNG to PDF 변환기 (오름차순 정렬)')
        self.setGeometry(300, 300, 400, 200)

        layout = QVBoxLayout()

        self.lbl_info = QLabel('이미지가 있는 폴더를 선택해주세요.', self)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        self.btn_select = QPushButton('📂 이미지 폴더 선택', self)
        self.btn_select.clicked.connect(self.select_folder)
        layout.addWidget(self.btn_select)

        self.btn_convert = QPushButton('🔄 PDF로 변환하기', self)
        self.btn_convert.clicked.connect(self.convert_files)
        self.btn_convert.setEnabled(False)
        layout.addWidget(self.btn_convert)

        self.setLayout(layout)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "이미지 폴더 선택")
        if folder:
            self.source_folder = folder
            self.lbl_info.setText(f"선택된 폴더:\n{folder}")
            self.btn_convert.setEnabled(True)

    # [중요] 자연스러운 정렬을 위한 헬퍼 함수
    def natural_sort_key(self, text):
        # 파일명에서 숫자와 숫자가 아닌 부분을 분리하여 리스트로 만듭니다.
        # 예: 'file10.png' -> ['file', 10, '.png']
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

    def convert_files(self):
        if not self.source_folder:
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "PDF 저장", "", "PDF Files (*.pdf)")
        if not save_path:
            return

        try:
            # 1. 파일 목록 가져오기
            files = [f for f in os.listdir(self.source_folder) if f.lower().endswith('.png')]

            # 2. [수정됨] 숫자 기준 오름차순 정렬 (1 -> 2 -> ... -> 10)
            files.sort(key=self.natural_sort_key)

            if not files:
                QMessageBox.warning(self, "오류", "선택한 폴더에 PNG 파일이 없습니다.")
                return

            image_list = []
            
            # 3. 이미지 변환 작업
            for file in files:
                img_path = os.path.join(self.source_folder, file)
                img = Image.open(img_path)
                img = img.convert('RGB')
                image_list.append(img)

            if image_list:
                image_list[0].save(
                    save_path,
                    save_all=True,
                    append_images=image_list[1:]
                )
                
                QMessageBox.information(self, "성공", f"변환 완료!\n총 {len(image_list)}장의 이미지가 합쳐졌습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "에러 발생", f"변환 중 오류가 발생했습니다:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PngToPdfConverter()
    ex.show()
    sys.exit(app.exec_())