from PyQt5 import QtWidgets, QtGui, QtCore
import sys
from pathlib import Path
import shutil
import yaml

# ──────────────────────────────────────────────────────────────
# YAML 로드
# ──────────────────────────────────────────────────────────────
def load_class_id_to_name_from_yaml(yaml_path: str) -> dict[int, str]:
    if not Path(yaml_path).exists():
        raise FileNotFoundError("YAML 파일을 찾을 수 없습니다.")
        
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    raise ValueError("YAML 파일 형식이 올바르지 않습니다 (names 항목 누락).")


# ──────────────────────────────────────────────────────────────
# 커스텀 뷰어
# ──────────────────────────────────────────────────────────────
class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, image_dir: Path, label_dir: Path,
                 class_id_to_name: dict[int, str], parent=None):
        super().__init__(parent)

        # 데이터 초기화
        self.class_id_to_name = class_id_to_name
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_files = []
        self.current_index = 0
        self.label_index = 0
        self.label_list = []
        self.undo_stack = []
        self.show_overlay = True
        
        # [추가] 다크모드 상태 변수
        self.is_dark_mode = True 

        # 이미지 파일 로드
        self.load_images_from_dir()

        # Scene 설정
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # 뷰 설정
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        # 초기 테마 설정
        self.apply_theme()

        # 텍스트 오버레이
        self.text_overlay = QtWidgets.QLabel(self)
        self.text_overlay.setFont(QtGui.QFont("Segoe UI", 11))
        self.text_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.text_overlay.move(15, 15)
        self.update_overlay_style() # 스타일 적용

        # 자동 재생 타이머
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_label)
        self.auto_playing = False

        self.update_image()

    def load_images_from_dir(self):
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        if self.image_dir.exists():
            self.image_files = sorted(
                [f for f in self.image_dir.iterdir() if f.suffix.lower() in exts]
            )
        else:
            self.image_files = []

    # ─────────────────── 테마 토글 기능 [추가됨] ───────────────────
    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.apply_theme()
        self.update_overlay_style()
        
        # 안내 메시지 잠깐 표시
        mode_str = "Dark Mode" if self.is_dark_mode else "Light Mode"
        self.text_overlay.setText(f"Switched to {mode_str}")
        self.text_overlay.adjustSize()
        QtCore.QTimer.singleShot(1000, self.update_image_text) # 1초 뒤 원래 텍스트로

    def apply_theme(self):
        if self.is_dark_mode:
            self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        else:
            self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))

    def update_overlay_style(self):
        if self.is_dark_mode:
            # 다크모드: 어두운 배경 + 흰 글씨
            self.text_overlay.setStyleSheet(
                "color: #EEE; font-weight: bold; background-color: rgba(0, 0, 0, 150); padding: 6px; border-radius: 4px;"
            )
        else:
            # 라이트모드: 밝은 배경 + 검은 글씨
            self.text_overlay.setStyleSheet(
                "color: #333; font-weight: bold; background-color: rgba(255, 255, 255, 180); padding: 6px; border-radius: 4px; border: 1px solid #CCC;"
            )

    # ─────────────────── 외부 호출 기능 ───────────────────
    def load_new_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "데이터셋 폴더 선택")
        if not folder: return

        path = Path(folder)
        img_dir = path / "images"
        lbl_dir = path / "labels"
        
        if not img_dir.exists():
            if path.name == "images":
                img_dir = path
                lbl_dir = path.parent / "labels"
            else:
                if (path / "train" / "images").exists():
                    res = QtWidgets.QMessageBox.question(self, "폴더 선택", 
                        "'train' 폴더를 여시겠습니까? (No를 누르면 'valid' 폴더)", 
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
                    if res == QtWidgets.QMessageBox.Yes:
                        img_dir = path / "train" / "images"
                        lbl_dir = path / "train" / "labels"
                    elif res == QtWidgets.QMessageBox.No:
                        img_dir = path / "valid" / "images"
                        lbl_dir = path / "valid" / "labels"
                    else:
                        return
                else:
                    QtWidgets.QMessageBox.warning(self, "오류", "폴더 내에 'images'와 'labels' 폴더가 있어야 합니다.")
                    return

        if not lbl_dir.exists():
             QtWidgets.QMessageBox.warning(self, "오류", "'labels' 폴더를 찾을 수 없습니다.")
             return

        self.image_dir = img_dir
        self.label_dir = lbl_dir
        self.load_images_from_dir()
        self.current_index = 0
        self.label_index = 0
        self.undo_stack.clear()
        self.update_image()

    def goto_image_by_index(self, idx):
        if 0 <= idx < len(self.image_files):
            self.current_index = idx
            self.label_index = 0
            self.update_image()
        else:
            QtWidgets.QMessageBox.warning(self, "오류", f"유효하지 않은 번호입니다. (1 ~ {len(self.image_files)})")

    # ─────────────────── 화면 갱신 ───────────────────
    def update_image(self):
        if not self.image_files:
            self.scene.clear()
            self.text_overlay.setText("No images found.")
            self.text_overlay.adjustSize()
            return

        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        image_path = self.image_files[self.current_index]
        pixmap = QtGui.QPixmap(str(image_path))
        
        if pixmap.isNull():
            self.text_overlay.setText(f"Error loading: {image_path.name}")
            return

        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))

        # 라벨 파일 읽기
        label_path = self.label_dir / image_path.with_suffix(".txt").name
        self.label_list = []
        if label_path.exists():
            with open(label_path, "r", encoding='utf-8') as f:
                self.label_list = [ln.strip() for ln in f if ln.strip()]
        
        if self.label_list:
            self.label_index = self.label_index % len(self.label_list)
        else:
            self.label_index = 0

        self.update_image_text() # 텍스트 업데이트 분리
        self._draw_polygons(pixmap.width(), pixmap.height())

    def update_image_text(self):
        image_path = self.image_files[self.current_index]
        cls_info = "No Label"
        if self.label_list:
            try:
                cls_id = int(self.label_list[self.label_index].split()[0])
                cls_name = self.class_id_to_name.get(cls_id, f"Unknown ID: {cls_id}")
                cls_info = f"{cls_id}: {cls_name}"
            except:
                cls_info = "Label Error"

        toggle_stat = "ON" if self.show_overlay else "OFF"
        txt = (f"[{self.current_index + 1} / {len(self.image_files)}]  {image_path.name}\n"
               f"Label: {self.label_index + 1} / {len(self.label_list)}  ▶  {cls_info}\n"
               f"Box View (S): {toggle_stat}")
        self.text_overlay.setText(txt)
        self.text_overlay.adjustSize()

    def _draw_polygons(self, w, h):
        for item in self.scene.items():
            if isinstance(item, (QtWidgets.QGraphicsPolygonItem, QtWidgets.QGraphicsRectItem)):
                self.scene.removeItem(item)

        if not self.show_overlay or not self.label_list:
            return

        current_line = self.label_list[self.label_index]
        self._draw_single_poly(current_line, w, h, is_selected=True)

        for i, line in enumerate(self.label_list):
            if i == self.label_index: continue
            self._draw_single_poly(line, w, h, is_selected=False)

    def _draw_single_poly(self, line, w, h, is_selected):
        try:
            parts = list(map(float, line.split()[1:]))
            if len(parts) == 4: # Box
                cx, cy, bw, bh = parts
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                rect = QtCore.QRectF(x1, y1, bw*w, bh*h)
                item = QtWidgets.QGraphicsRectItem(rect)
            else: # Polygon
                pts = [QtCore.QPointF(parts[i] * w, parts[i+1] * h) for i in range(0, len(parts), 2)]
                item = QtWidgets.QGraphicsPolygonItem(QtGui.QPolygonF(pts))

            if is_selected:
                pen = QtGui.QPen(QtGui.QColor("#00FF00"), 3)
                brush = QtGui.QBrush(QtGui.QColor(0, 255, 0, 40))
                item.setZValue(10)
            else:
                pen = QtGui.QPen(QtGui.QColor("#FF0000"), 1)
                brush = QtGui.QBrush(QtCore.Qt.NoBrush)
                item.setZValue(5)

            item.setPen(pen)
            item.setBrush(brush)
            self.scene.addItem(item)
        except:
            pass

    # ─────────────────── 이벤트 ───────────────────
    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1 / 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_D: self.next_label()
        elif key == QtCore.Qt.Key_A: self.prev_label()
        elif key == QtCore.Qt.Key_W: self.delete_current_label()
        elif key == QtCore.Qt.Key_Q: self.change_label_dialog()
        elif key == QtCore.Qt.Key_R: self.undo_last_deletion()
        elif key == QtCore.Qt.Key_S: 
            self.show_overlay = not self.show_overlay
            self.update_image()
        elif key == QtCore.Qt.Key_Space:
            self.toggle_auto_play()
        elif key == QtCore.Qt.Key_T: # [추가] 테마 토글 키
            self.toggle_theme()

    # ─────────────────── 백업 폴더 경로 계산 ───────────────────
    def _get_error_dirs(self, img_path: Path):
        try:
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
        except:
            error_root = self.image_dir.parent.parent.with_name(self.image_dir.parent.parent.name + "_error")
            err_img_dir = error_root / "images"
            err_lbl_dir = error_root / "labels"
            err_img_dir.mkdir(parents=True, exist_ok=True)
            err_lbl_dir.mkdir(parents=True, exist_ok=True)
            return err_img_dir, err_lbl_dir

    # ─────────────────── 삭제 (W) 로직 ───────────────────
    def delete_current_label(self):
        if not self.label_list: 
            QtWidgets.QMessageBox.information(self, "알림", "삭제할 라벨이 없습니다.")
            return

        line = self.label_list[self.label_index]
        cls_id = int(line.split()[0])
        cls_name = self.class_id_to_name.get(cls_id, str(cls_id))
        
        reply = QtWidgets.QMessageBox.question(
            self, "삭제 확인", 
            f"선택한 라벨을 삭제하시겠습니까?\n\n클래스: {cls_name} (ID: {cls_id})\n\n(삭제된 데이터는 _error 폴더에 백업됩니다)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.No:
            return

        img_path = self.image_files[self.current_index]
        lbl_path = self.label_dir / img_path.with_suffix(".txt").name
        
        err_img_dir, err_lbl_dir = self._get_error_dirs(img_path)
        
        # 백업
        backup_img = err_img_dir / img_path.name
        if not backup_img.exists():
            try: shutil.copy2(str(img_path), str(backup_img))
            except Exception as e: print(f"이미지 백업 실패: {e}")

        backup_lbl = err_lbl_dir / lbl_path.name
        try:
            with open(backup_lbl, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e: print(f"라벨 백업 실패: {e}")

        # Undo 스택
        self.undo_stack.append({
            "img_idx": self.current_index,
            "label_idx": self.label_index,
            "content": line
        })

        del self.label_list[self.label_index]
        self._save_current_labels(lbl_path)

        if self.label_list:
            self.label_index = min(self.label_index, len(self.label_list) - 1)
        self.update_image()

    def undo_last_deletion(self):
        if not self.undo_stack:
            QtWidgets.QMessageBox.information(self, "알림", "복구할 내역이 없습니다.")
            return

        action = self.undo_stack.pop()
        self.current_index = action["img_idx"]
        
        img_path = self.image_files[self.current_index]
        lbl_path = self.label_dir / img_path.with_suffix(".txt").name
        
        current_lines = []
        if lbl_path.exists():
            with open(lbl_path, "r", encoding='utf-8') as f:
                current_lines = [l.strip() for l in f if l.strip()]

        insert_idx = min(action["label_idx"], len(current_lines))
        current_lines.insert(insert_idx, action["content"])

        with open(lbl_path, "w", encoding='utf-8') as f:
            f.write("\n".join(current_lines) + "\n")

        self.label_index = insert_idx
        self.update_image()

    def _save_current_labels(self, path):
        with open(path, "w", encoding='utf-8') as f:
            if self.label_list:
                f.write("\n".join(self.label_list) + "\n")
            else:
                f.write("") 

    # ─────────────────── 네비게이션 ───────────────────
    def next_label(self):
        if self.label_list and self.label_index < len(self.label_list) - 1:
            self.label_index += 1
        else:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.label_index = 0
        self.update_image()

    def prev_label(self):
        if self.label_index > 0:
            self.label_index -= 1
        else:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            tmp_path = self.image_files[self.current_index]
            lbl = self.label_dir / tmp_path.with_suffix(".txt").name
            cnt = 0
            if lbl.exists():
                with open(lbl) as f: cnt = len(f.readlines())
            self.label_index = max(0, cnt - 1)
        self.update_image()

    def change_label_dialog(self):
        if not self.label_list: return

        class_items = [
            f"{cid:03d}  {name}"
            for cid, name in sorted(self.class_id_to_name.items())
            if name
        ]

        cur_line = self.label_list[self.label_index]
        cur_id = int(cur_line.split()[0])
        cur_name = self.class_id_to_name.get(cur_id, f"class_{cur_id}")

        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle("클래스 변경 (Q)")
        dlg.setLabelText(f"현재: {cur_id} {cur_name}\n새 클래스를 선택하거나 검색하세요:")
        dlg.setComboBoxItems(class_items)
        dlg.setComboBoxEditable(True)
        dlg.resize(400, 200)

        cb = dlg.findChild(QtWidgets.QComboBox)
        cb.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        completer = QtWidgets.QCompleter(class_items, cb)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        cb.setCompleter(completer)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_text = dlg.textValue().strip()
            if not new_text: return
            try:
                new_id = int(new_text.split()[0])
            except ValueError: return

            if new_id == cur_id: return

            parts = cur_line.split()
            parts[0] = str(new_id)
            self.label_list[self.label_index] = " ".join(parts)
            
            img_path = self.image_files[self.current_index]
            lbl_path = self.label_dir / img_path.with_suffix(".txt").name
            self._save_current_labels(lbl_path)
            self.update_image()

    def toggle_auto_play(self, interval_ms=500):
        if self.auto_playing:
            self.timer.stop()
            self.auto_playing = False
            self.text_overlay.setStyleSheet("color: #EEE; font-weight: bold; background-color: rgba(0, 0, 0, 150);")
        else:
            self.timer.start(interval_ms)
            self.auto_playing = True
            self.text_overlay.setStyleSheet("color: #0F0; font-weight: bold; background-color: rgba(0, 0, 0, 180); border: 1px solid #0F0;")

# ──────────────────────────────────────────────────────────────
# 메인 윈도
# ──────────────────────────────────────────────────────────────
class MainApp(QtWidgets.QMainWindow):
    def __init__(self, img_dir, lbl_dir, class_map):
        super().__init__()
        self.setWindowTitle("YOLO Reviewer (Ultimate Integrated)")
        self.resize(1280, 800)
        
        self.viewer = ImageViewer(img_dir, lbl_dir, class_map)
        self.setCentralWidget(self.viewer)

        # 메뉴바
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_act = QtWidgets.QAction("Open Folder...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.viewer.load_new_folder)
        file_menu.addAction(open_act)
        
        exit_act = QtWidgets.QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # 툴바
        toolbar = QtWidgets.QToolBar("Tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.page_edit = QtWidgets.QLineEdit()
        self.page_edit.setFixedWidth(50)
        self.page_edit.setPlaceholderText("No.")
        self.page_edit.returnPressed.connect(self.go_to_page)
        
        toolbar.addWidget(QtWidgets.QLabel("  Go to: "))
        toolbar.addWidget(self.page_edit)
        
        btn_go = QtWidgets.QPushButton("Go")
        btn_go.clicked.connect(self.go_to_page)
        toolbar.addWidget(btn_go)

        toolbar.addSeparator()

        btn_help = QtWidgets.QPushButton("Help (?)")
        btn_help.clicked.connect(self.show_help)
        toolbar.addWidget(btn_help)

    def go_to_page(self):
        try:
            idx = int(self.page_edit.text()) - 1
            self.viewer.goto_image_by_index(idx)
        except:
            pass

    def show_help(self):
        msg = """
        <b>단축키 안내</b><br>
        -----------------------------<br>
        <b>Ctrl + O</b> : 다른 폴더 열기<br>
        <b>A / D</b> : 이전 / 다음 라벨<br>
        <b>W</b> : 라벨 삭제 (백업 저장됨)<br>
        <b>R</b> : 삭제 복구 (Undo)<br>
        <b>Q</b> : 클래스 변경<br>
        <b>S</b> : 박스 토글<br>
        <b>T</b> : 다크/라이트 모드 토글<br>
        <b>Space</b> : 자동 재생
        """
        QtWidgets.QMessageBox.information(self, "Help", msg)

def run_app():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    yaml_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select YAML file", "", "YAML (*.yaml *.yml)")
    if not yaml_path: return
    try:
        class_map = load_class_id_to_name_from_yaml(yaml_path)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", str(e))
        return

    root_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Dataset Root")
    if not root_path: return
    
    root = Path(root_path)
    img_dir = root / "images"
    lbl_dir = root / "labels"

    if not img_dir.exists():
        if (root / "train" / "images").exists():
            img_dir = root / "train" / "images"
            lbl_dir = root / "train" / "labels"
        else:
            QtWidgets.QMessageBox.critical(None, "Error", "images 폴더를 찾을 수 없습니다.")
            return

    win = MainApp(img_dir, lbl_dir, class_map)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()