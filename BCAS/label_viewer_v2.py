from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from pathlib import Path
from shutil import copy2

# ──────────────────────────────────────────────────────────────
# 메인 뷰어
# ──────────────────────────────────────────────────────────────
class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, image_dir: Path, label_dir: Path, parent=None):
        super().__init__(parent)

        # 데이터 설정
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        
        # 이미지 파일 리스트 로드 (이름순 정렬)
        self.image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )

        self.current_index = 0          # 현재 이미지 인덱스
        self.label_index = 0            # 현재 라벨(줄) 인덱스
        self.label_list: list[str] = [] # 현재 txt 라벨 전체 내용
        self.undo_stack = []            # 삭제 → 되돌리기용 스택
        self.show_overlay = True        # 폴리곤 표시 ON/OFF

        # QGraphicsView 기본 세팅
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        # 텍스트 오버레이 (좌상단 정보창)
        self.text_overlay = QtWidgets.QLabel(self)
        self.text_overlay.setStyleSheet(
            "color:black;font-weight:bold;background-color:rgba(255,255,255,200);padding:4px;"
        )
        self.text_overlay.setFont(QtGui.QFont("Arial", 12))
        self.text_overlay.move(10, 10)
        self.text_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        self.update_image()

        # 자동 재생용 타이머
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_label)
        self.auto_playing = False 

    # ───────────────────  기본 기능  ───────────────────
    def goto_image_by_index(self, idx: int):
        if 0 <= idx < len(self.image_files):
            self.current_index = idx
            self.label_index = 0
            self.update_image()
        else:
            QtWidgets.QMessageBox.warning(self, "오류", "해당 번호의 이미지가 없습니다.")

    # [수정됨] 파일명 검색 (리스트 처음부터 찾아 가장 빠른 것 하나만 이동)
    def goto_image_by_name(self, query: str) -> bool:
        """
        리스트의 처음(0번)부터 끝까지 순회하며, 
        검색어가 포함된 '가장 빠른' 파일을 찾아 이동합니다.
        """
        query = query.lower().strip()
        
        # 0번부터 순서대로 검사
        for idx, path in enumerate(self.image_files):
            # 파일명에 검색어가 포함되어 있으면
            if query in path.name.lower():
                self.current_index = idx
                self.label_index = 0
                self.update_image()
                return True # 찾자마자 함수 종료 (가장 빠른 파일에서 멈춤)
        
        return False

    def load_new_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "images/labels 포함 폴더 선택")
        if not folder:
            return
        image_dir = Path(folder) / "images"
        label_dir = Path(folder) / "labels"
        if not image_dir.exists() or not label_dir.exists():
            QtWidgets.QMessageBox.critical(self, "오류", "선택한 폴더 안에 images/, labels/ 폴더가 있어야 합니다.")
            return
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(
            [f for f in self.image_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        self.current_index, self.label_index = 0, 0
        self.update_image()

    # ───────────────────  화면 갱신  ───────────────────
    def update_image(self):
        if not self.image_files or self.current_index >= len(self.image_files):
            self.setSceneRect(QtCore.QRectF(0, 0, 1000, 800))
            self.pixmap_item.setPixmap(QtGui.QPixmap())
            self.text_overlay.setText("No images.")
            self.text_overlay.adjustSize()
            return

        image_path = self.image_files[self.current_index]
        pixmap = QtGui.QPixmap(str(image_path))
        self.pixmap_item.setPixmap(pixmap)
        self.resetTransform()
        self.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.scene.update()
        self.setFocus(QtCore.Qt.OtherFocusReason) 

        # 라벨 파일 읽기
        label_path = self.label_dir / image_path.with_suffix(".txt").name
        self.label_list = []
        if label_path.exists():
            with open(label_path, "r", encoding='utf-8') as f:
                self.label_list = [ln.strip() for ln in f if ln.strip()]
        
        if self.label_index >= len(self.label_list):
            self.label_index = 0

        # 오버레이 글자
        total = len(self.image_files)
        class_name = "N/A"
        
        if self.label_list:
            parts = self.label_list[self.label_index].split()
            if parts:
                class_name = parts[0] 

        self.text_overlay.setText(
            f"{self.current_index+1}/{total} | {image_path.name}\n🟢 {class_name}"
        )
        self.text_overlay.adjustSize()

        # 기존 폴리곤 삭제 후 다시 그림
        for it in self.scene.items():
            if isinstance(it, QtWidgets.QGraphicsPolygonItem):
                self.scene.removeItem(it)

        if self.show_overlay and self.label_list:
            parts = self.label_list[self.label_index].split()
            try:
                coords = list(map(float, parts[1:]))
                w, h = pixmap.width(), pixmap.height()
                pts = [QtCore.QPointF(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)]
                poly = QtGui.QPolygonF(pts)
                poly_item = QtWidgets.QGraphicsPolygonItem(poly)
                poly_item.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0, 200), 2))
                self.scene.addItem(poly_item)
            except ValueError:
                pass 

    # ───────────────────  키보드 이벤트  ───────────────────
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_D:
            self.next_label()
        elif event.key() == QtCore.Qt.Key_A:
            self.prev_label()
        elif event.key() == QtCore.Qt.Key_W:
            self.delete_current_label()
        elif event.key() == QtCore.Qt.Key_Q:
            self.change_current_label()
        elif event.key() == QtCore.Qt.Key_R:
            self.undo_last_deletion()
        elif event.key() == QtCore.Qt.Key_S:
            self.show_overlay = not self.show_overlay
            self.update_image()
        elif event.key() == QtCore.Qt.Key_F:
            self.toggle_auto_play()
        elif event.key() == QtCore.Qt.Key_F1:
            self.toggle_auto_play(interval_ms=1000)

    def wheelEvent(self, event):
        self.scale(1.1 if event.angleDelta().y() > 0 else 0.9, 1.1 if event.angleDelta().y() > 0 else 0.9)

    # ───────────────────  라벨 탐색  ───────────────────
    def next_label(self):
        if self.label_index < len(self.label_list) - 1:
            self.label_index += 1
        else:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                self.label_index = 0
        self.update_image()

    def prev_label(self):
        if self.label_index > 0:
            self.label_index -= 1
        else:
            if self.current_index > 0:
                self.current_index -= 1
                lbl = self.label_dir / self.image_files[self.current_index].with_suffix(".txt").name
                if lbl.exists():
                    with open(lbl, "r", encoding='utf-8') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                        self.label_index = max(0, len(lines) - 1)
        self.update_image()

    # -----------------------------------------------------------
    def _get_error_dirs(self, img_path: Path):
        images_dir = img_path.parent
        class_dir  = images_dir.parent
        split_dir  = class_dir.parent
        root_dir   = split_dir.parent

        error_root = root_dir.with_name(root_dir.name + "_error")
        error_dir  = error_root / split_dir.name / class_dir.name

        err_img_dir = error_dir / "images"
        err_lbl_dir = error_dir / "labels"
        err_img_dir.mkdir(parents=True, exist_ok=True)
        err_lbl_dir.mkdir(parents=True, exist_ok=True)
        return err_img_dir, err_lbl_dir
    
    # ───────────────────  라벨 삭제 (+undo push)  ───────────────────
    def delete_current_label(self):
        if not self.label_list:
            return

        line     = self.label_list[self.label_index]
        cls_name = line.split()[0] 

        if QtWidgets.QMessageBox.question(
            self, "라벨 삭제 확인",
            f"🟢 [{cls_name}] 라벨을 삭제하시겠습니까?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        ) != QtWidgets.QMessageBox.Yes:
            return

        image_path = self.image_files[self.current_index]
        label_path = self.label_dir / image_path.with_suffix(".txt").name
        err_img_dir, err_lbl_dir = self._get_error_dirs(image_path)

        will_delete_img = (len(self.label_list) == 1)
        self.undo_stack.append({
            "image_index":      self.current_index,
            "label_index":      self.label_index,
            "label_line":       line,
            "image_path":       image_path,
            "label_path":       label_path,
            "all_label_lines":  self.label_list.copy(),
            "image_was_deleted":will_delete_img
        })

        if not (err_img_dir / image_path.name).exists():
            copy2(str(image_path), err_img_dir / image_path.name)
        with open(err_lbl_dir / label_path.name, "a", encoding='utf-8') as ef:
            ef.write(line + "\n")

        del self.label_list[self.label_index]

        if self.label_list:
            with open(label_path, "w", encoding='utf-8') as f:
                f.write("\n".join(self.label_list) + "\n")
            self.label_index = min(self.label_index, len(self.label_list) - 1)
        else:
            if label_path.exists():
                label_path.unlink()
            if image_path.exists():
                image_path.unlink()
            del self.image_files[self.current_index]
            self.current_index = max(0, self.current_index - 1)
            self.label_index   = 0

        self.update_image()


    # ───────────────────  삭제 되돌리기 (R)  ───────────────────
    def undo_last_deletion(self):
        if not self.undo_stack:
            QtWidgets.QMessageBox.information(self, "알림", "되돌릴 라벨 삭제 기록이 없습니다.")
            return

        last = self.undo_stack.pop()
        img_idx  = last["image_index"]
        lbl_idx  = last["label_index"]
        img_path = last["image_path"]
        lbl_path = last["label_path"]
        line     = last["label_line"]
        all_lbls = last["all_label_lines"]
        img_del  = last["image_was_deleted"]

        cls_name = line.split()[0] 
        
        if QtWidgets.QMessageBox.question(
            self, "라벨 복구 확인",
            f"🟢 [{cls_name}] 라벨을 복구하시겠습니까?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes
        ) != QtWidgets.QMessageBox.Yes:
            return

        err_img_dir, err_lbl_dir = self._get_error_dirs(img_path)
        err_img = err_img_dir / img_path.name
        err_lbl = err_lbl_dir / lbl_path.name

        if img_del and err_img.exists():
            copy2(str(err_img), str(img_path))

        with open(lbl_path, "w", encoding='utf-8') as f:
            f.write("\n".join(all_lbls) + "\n")

        if err_lbl.exists():
            with open(err_lbl, "r", encoding='utf-8') as f:
                lines = f.readlines()
            tgt = line if line.endswith("\n") else line + "\n"
            lines = [l for l in lines if l != tgt]

            if lines:
                with open(err_lbl, "w", encoding='utf-8') as f:
                    f.writelines(lines)
            else:
                err_lbl.unlink()
                if err_img.exists():
                    err_img.unlink()

        if img_del and img_path not in self.image_files:
            self.image_files.insert(img_idx, img_path)

        self.current_index = img_idx
        self.label_list    = all_lbls
        self.label_index   = lbl_idx
        self.update_image()

    # ───────────────────  클래스 변경 (Q)  ───────────────────
    def change_current_label(self):
        if not self.label_list:
            return

        cur_line = self.label_list[self.label_index]
        cur_parts = cur_line.split()
        cur_name = cur_parts[0] 

        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "클래스 이름 변경", 
            f"현재 클래스: {cur_name}\n새로운 클래스 이름을 입력하세요:",
            QtWidgets.QLineEdit.Normal, cur_name
        )

        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        
        if new_name == cur_name:
            return

        cur_parts[0] = new_name
        self.label_list[self.label_index] = " ".join(cur_parts)

        lbl_path = self.label_dir / self.image_files[self.current_index].with_suffix(".txt").name
        with open(lbl_path, "w", encoding='utf-8') as f:
            f.write("\n".join(self.label_list) + "\n")

        self.update_image()

    def toggle_auto_play(self, interval_ms=500):
        if self.auto_playing:
            self.timer.stop()
            self.auto_playing = False
        else:
            self.timer.start(interval_ms)
            self.auto_playing = True

# ──────────────────────────────────────────────────────────────
# 메인 윈도
# ──────────────────────────────────────────────────────────────
class MainApp(QtWidgets.QMainWindow):
    def __init__(self, image_dir: Path, label_dir: Path):
        super().__init__()
        self.setWindowTitle("YOLO 객체 단위 리뷰어 (No-YAML) | W:삭제 R:취소 Q:라벨변경")
        self.viewer = ImageViewer(image_dir, label_dir)
        self.setCentralWidget(self.viewer)
        self.resize(1200, 900)

        # 메뉴
        menubar = self.menuBar()
        file_menu = menubar.addMenu("파일")
        open_act = QtWidgets.QAction("폴더 열기", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.viewer.load_new_folder)
        file_menu.addAction(open_act)

        # 툴바 생성
        tb = QtWidgets.QToolBar("탐색 도구")
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)

        # 1. 번호로 이동
        self.page_input = QtWidgets.QLineEdit()
        self.page_input.setFixedWidth(50)
        self.page_input.setPlaceholderText("번호")
        self.page_input.returnPressed.connect(self.goto_page)
        
        go_btn = QtWidgets.QPushButton("이동")
        go_btn.clicked.connect(self.goto_page)

        tb.addWidget(QtWidgets.QLabel("번호: "))
        tb.addWidget(self.page_input)
        tb.addWidget(go_btn)

        # 구분선
        tb.addSeparator()

        # 2. 파일명으로 검색
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setFixedWidth(150)
        self.name_input.setPlaceholderText("파일명(부분일치)")
        self.name_input.returnPressed.connect(self.search_by_name)
        
        search_btn = QtWidgets.QPushButton("검색")
        search_btn.clicked.connect(self.search_by_name)

        tb.addWidget(QtWidgets.QLabel("  파일명: "))
        tb.addWidget(self.name_input)
        tb.addWidget(search_btn)

    def goto_page(self):
        try:
            idx = int(self.page_input.text())
            self.viewer.goto_image_by_index(idx - 1)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "오류", "유효한 번호를 입력하세요.")

    def search_by_name(self):
        text = self.name_input.text().strip()
        if not text:
            return
        
        # viewer의 검색 기능 호출 (True면 성공, False면 실패)
        if not self.viewer.goto_image_by_name(text):
            QtWidgets.QMessageBox.warning(self, "검색 실패", f"'{text}'가 포함된 파일명을 찾을 수 없습니다.")


# ──────────────────────────────────────────────────────────────
# 앱 실행
# ──────────────────────────────────────────────────────────────
def run_app():
    app = QtWidgets.QApplication(sys.argv)

    root = QtWidgets.QFileDialog.getExistingDirectory(None, "images/labels 포함 폴더 선택")
    if not root:
        return
    img_dir = Path(root) / "images"
    lbl_dir = Path(root) / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        QtWidgets.QMessageBox.critical(None, "오류", "images/, labels/ 폴더가 필요합니다.")
        return

    win = MainApp(img_dir, lbl_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()