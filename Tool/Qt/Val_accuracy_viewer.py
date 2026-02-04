#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validation_accuracy_viewer.py (정탐 고정 + 미탐/오탐 토글 전환)
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QFileDialog, QListWidget, QListWidgetItem,
    QLabel, QHBoxLayout, QVBoxLayout, QPushButton, QSplitter,
    QToolBar, QAction, QStyle, QStatusBar, QMessageBox, QMenu, QSizePolicy
)
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
from PyQt5.QtGui import QColor   # 색상 강조용

# (선택) pyperclip
try:
    import pyperclip  # type: ignore
    HAVE_PYPERCLIP = True
except Exception:
    HAVE_PYPERCLIP = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUDE_DIRS = {"__pycache__", "cache", ".cache", ".git", "logs", "_logs", "tmp", "temp", ".ds_store"}

ALT_NAMES = {
    "TP": "정탐", "tp": "정탐", "true_positive": "정탐", "true": "정탐", "posit": "정탐",
    "FP": "오탐", "fp": "오탐", "false_positive": "오탐", "false": "오탐",
    "FN": "미탐", "fn": "미탐", "miss": "미탐", "missed": "미탐",
    "정탐": "정탐", "오탐": "오탐", "미탐": "미탐"
}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def normalize_verdict_name(name: str) -> str:
    return ALT_NAMES.get(name, name)


def enumerate_images_under(dir_path: Path) -> List[Path]:
    imgs: List[Path] = []
    if not dir_path.exists() or not dir_path.is_dir():
        return imgs
    for p in dir_path.iterdir():
        if is_image(p):
            imgs.append(p)
    for sub in dir_path.iterdir():
        if sub.is_dir():
            for q in sub.iterdir():
                if is_image(q):
                    imgs.append(q)
    return sorted(imgs)


def find_subfolder(class_dir: Path, wanted_kor_name: str) -> Path:
    direct = class_dir / wanted_kor_name
    if direct.exists():
        return direct
    candidates = ["정탐", "오탐", "미탐", "TP", "FP", "FN", "tp", "fp", "fn",
                  "true_positive", "false_positive", "miss", "missed", "true", "false", "posit"]
    for cand in candidates:
        if normalize_verdict_name(cand) == wanted_kor_name:
            p = class_dir / cand
            if p.exists():
                return p
    for sub in class_dir.iterdir():
        if not sub.is_dir():
            continue
        low = sub.name.lower()
        if wanted_kor_name == "정탐" and ("정탐" in low or "tp" in low or "true" in low or "posit" in low):
            return sub
        if wanted_kor_name == "오탐" and ("오탐" in low or "fp" in low or "false" in low):
            return sub
        if wanted_kor_name == "미탐" and ("미탐" in low or "fn" in low or "miss" in low):
            return sub
    return class_dir / wanted_kor_name

class NumericTreeWidgetItem(QTreeWidgetItem):
    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        # 정탐율(1열)은 숫자로 정렬
        if column == 1:
            try:
                return float(self.text(1)) < float(other.text(1))
            except ValueError:
                return self.text(1) < other.text(1)
        return super().__lt__(other)


# ───────── 이미지 행 위젯 ─────────
class ImageRow(QWidget):
    def __init__(self, title: str, fill_mode_getter, parent=None):
        super().__init__(parent)
        self.title = title
        self.images: List[Path] = []
        self.page_start: int = 0
        self._get_fill_mode = fill_mode_getter

        # 왼쪽 제목
        self.lbl_title = QLabel(self.title)
        self.lbl_title.setStyleSheet("font-weight:700; font-size:14px;")
        self.lbl_title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color:#777; font-size:12px;")
        self.lbl_info.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        title_row = QHBoxLayout()
        title_row.addWidget(self.lbl_title)
        title_row.addWidget(self.lbl_info)

        title_container = QWidget()
        title_container.setLayout(title_row)
        title_container.setMaximumWidth(100)
        title_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # 중앙 이미지 3개
        self.img_labels = []
        img_row = QHBoxLayout()
        img_row.setSpacing(2)
        img_row.setContentsMargins(0, 0, 0, 0)
        for _ in range(3):
            lab = QLabel("이미지 없음")
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet("background:#111; color:#bbb; border:1px solid #222;")
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            lab.setContextMenuPolicy(Qt.CustomContextMenu)
            lab.customContextMenuRequested.connect(self._context_menu_for_label)
            self.img_labels.append(lab)
            img_row.addWidget(lab, 1)

        # 오른쪽 버튼
        self.btn_prev = QPushButton("이전 3장")
        self.btn_next = QPushButton("다음 3장")
        for b in (self.btn_prev, self.btn_next):
            b.setFixedHeight(30)
            b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_prev)
        btn_row.addWidget(self.btn_next)

        btn_container = QWidget()
        btn_container.setLayout(btn_row)
        btn_container.setMaximumWidth(180)
        btn_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # 전체 행
        main_row = QHBoxLayout()
        main_row.setSpacing(0)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.addWidget(title_container)
        main_row.addLayout(img_row, 1)
        main_row.addWidget(btn_container)

        self.setLayout(main_row)



    def set_images(self, images: List[Path]):
        self.images = images or []
        self.page_start = 0
        self.update_view()

    def set_info(self, count: int, percent: float):
        self.lbl_info.setText(f"{count}장 ({percent:.1f}%)")

    def prev_page(self):
        if self.page_start >= 3:
            self.page_start -= 3
            self.update_view()

    def next_page(self):
        if self.page_start + 3 < len(self.images):
            self.page_start += 3
            self.update_view()

    def load_images(self, root_dir):
        self.images = sorted(Path(root_dir).rglob("*.*"))
        self.images = [p for p in self.images if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
        self.page_start = 0
        self.update_stats()   # ✅ 퍼센티지 계산
        self.update_view()    # 기존 기능 유지





    def update_view(self):
        total = len(self.images)
        self.btn_prev.setEnabled(self.page_start > 0)
        self.btn_next.setEnabled(self.page_start + 3 < total)

        for i in range(3):
            idx = self.page_start + i
            lab = self.img_labels[i]
            if 0 <= idx < total:
                p = self.images[idx]
                pix = QPixmap(str(p))
                if not pix.isNull():
                    tgt = QSize(max(1, lab.width()), max(1, lab.height()))
                    pix = pix.scaled(tgt, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    # 🧼 ✅ 오버레이 제거 — 텍스트 없음
                    lab.setPixmap(pix)
                    lab.setToolTip(str(p))
                    lab.setText("")
                else:
                    lab.setPixmap(QPixmap())
                    lab.setText("불러오기 실패")
            else:
                lab.setPixmap(QPixmap())
                lab.setText("이미지 없음")



    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_view()

    def _context_menu_for_label(self, pos):
        lab = self.sender()
        if not isinstance(lab, QLabel):
            return
        i = self.img_labels.index(lab)
        idx = self.page_start + i
        if idx < 0 or idx >= len(self.images):
            return
        img_path = self.images[idx]
        menu = QMenu(self)
        act_open = QAction("파일 위치 열기", self)
        act_copy = QAction("경로 복사", self)

        def _open_folder():
            try:
                if sys.platform.startswith("win"):
                    subprocess.Popen(r'explorer /select,"{}"'.format(str(img_path)))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", str(img_path)])
                else:
                    subprocess.Popen(["xdg-open", str(img_path.parent)])
            except Exception as ex:
                QMessageBox.warning(self, "오류", f"열기 실패: {ex}")

        def _copy_path():
            try:
                if HAVE_PYPERCLIP:
                    pyperclip.copy(str(img_path))
                else:
                    QApplication.clipboard().setText(str(img_path))
            except Exception:
                QApplication.clipboard().setText(str(img_path))

        act_open.triggered.connect(_open_folder)
        act_copy.triggered.connect(_copy_path)
        menu.addAction(act_open)
        menu.addAction(act_copy)
        menu.exec_(QCursor.pos())


# ───────── 메인 ─────────
class ResultsViewer(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("검증 결과 뷰어 (정탐 고정 + 미탐/오탐 토글)")
        self.resize(1500, 900)

        
        self.results_root: Optional[Path] = None
        self._fill_mode = False

        # 사이드바 생성 ✅
        sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(sidebar_widget)

        # 전체 퍼센티지 라벨 추가 ✅
        self.stats_label = QLabel("정탐: 0%  오탐: 0%  미탐: 0%")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 12px; font-weight: 600;")
        self.sidebar_layout.addWidget(self.stats_label)

        # ✅ QTreeWidget으로 변경
        self.class_tree = QTreeWidget()
        self.class_tree.setColumnCount(2)
        self.class_tree.setHeaderLabels(["클래스", "정탐율(%)"])
        self.class_tree.header().setStretchLastSection(False)
        self.class_tree.header().resizeSection(0, 150)   # 클래스 컬럼 폭
        self.class_tree.header().resizeSection(1, 80)    # 퍼센트 컬럼 폭
        self.class_tree.setAlternatingRowColors(True)
        self.class_tree.itemClicked.connect(self._on_class_clicked_tree)
        self.sidebar_layout.addWidget(self.class_tree)

        # 현재 선택 클래스 표시
        self.lbl_current_class = QLabel("클래스: (없음)")
        self.lbl_current_class.setAlignment(Qt.AlignCenter)
        self.lbl_current_class.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #FFDD55;
            padding: 10px;
            background-color: #333;
        """)

        # 우측 영역
        self.lbl_summary = QLabel("정탐 0.0% | 오탐 0.0% | 미탐 0.0% (총 0)")
        self.lbl_summary.setStyleSheet("font-size:15px; font-weight:600;")

        self.btn_toggle_fp = QPushButton("오탐 표시")
        self.btn_toggle_fp.setCheckable(True)
        self.btn_toggle_fp.setFixedHeight(35)
        self.btn_toggle_fp.toggled.connect(self._toggle_fp_fn)

        self.row_tp = ImageRow("정탐", self.get_fill_mode)
        self.row_fp = ImageRow("오탐", self.get_fill_mode)
        self.row_fn = ImageRow("미탐", self.get_fill_mode)

        self.row_fp.setVisible(False)
        self.row_fn.setVisible(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.lbl_summary)
        right_layout.addWidget(self.btn_toggle_fp)
        right_layout.addWidget(self.row_tp)
        right_layout.addWidget(self.row_fn)
        right_layout.addWidget(self.row_fp)

        right_container = QWidget()
        right_container.setLayout(right_layout)

        # Splitter에 좌우 배치
        splitter = QSplitter()
        splitter.addWidget(sidebar_widget)        # ✅ 사이드바 전체를 추가
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # 툴바
        tb = QToolBar("Main")
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        act_open = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "결과 루트 열기", self)
        act_open.triggered.connect(self._choose_root)
        tb.addAction(act_open)
        self.lbl_root = QLabel("루트: (미선택)")
        tb.addWidget(self.lbl_root)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _load_class_list(self):
        self.class_tree.clear()
        if not self.results_root or not self.results_root.exists():
            return

        class_stats = []
        for class_dir in sorted(self.results_root.iterdir()):
            if not class_dir.is_dir():
                continue

            tp_dir = find_subfolder(class_dir, "정탐")
            fp_dir = find_subfolder(class_dir, "오탐")
            fn_dir = find_subfolder(class_dir, "미탐")

            # ✅ 여기가 중요!
            c_tp = len(list(tp_dir.glob("*.jpg"))) + len(list(tp_dir.glob("*.png")))
            c_fp = len(list(fp_dir.glob("*.jpg"))) + len(list(fp_dir.glob("*.png")))
            c_fn = len(list(fn_dir.glob("*.jpg"))) + len(list(fn_dir.glob("*.png")))

            total = c_tp + c_fp + c_fn
            tp_ratio = (c_tp / total * 100.0) if total > 0 else 0.0  # ← 여기서 정의

            class_stats.append((class_dir.name, tp_ratio, class_dir))  # ← 여기서 사용

        for name, ratio, path in class_stats:
            item = NumericTreeWidgetItem([name, f"{ratio:.1f}"])
            item.setData(0, Qt.UserRole, str(path))
            item.setData(1, Qt.UserRole, ratio)

            item.setForeground(0, QColor("black"))
            if ratio >= 90:
                item.setForeground(1, QColor("green"))
            elif ratio >= 70:
                item.setForeground(1, QColor("orange"))
            else:
                item.setForeground(1, QColor("red"))

            self.class_tree.addTopLevelItem(item)

        # 정렬 호출
        self.class_tree.sortItems(1, Qt.DescendingOrder)

    def _toggle_fp_fn(self, checked: bool):
        self.row_fn.setVisible(not checked)
        self.row_fp.setVisible(checked)
        self.btn_toggle_fp.setText("오탐 숨기기" if checked else "오탐 표시")

    def get_fill_mode(self) -> bool:
        return self._fill_mode

    def _choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "결과 루트 선택")
        if not d:
            return

        self.results_root = Path(d)
        self.lbl_root.setText(f"루트: {d}")

        self._load_class_list()
        self.calculate_global_stats_once()   # ✅ 캐싱 1회만
        self.update_global_stats()           # ✅ 사이드바에 표시



    def _scan_classes(self):
        self.class_list.clear()
        if not self.results_root or not self.results_root.exists():
            self.status.showMessage("루트가 없거나 접근 불가")
            return
        classes = []
        for p in self.results_root.iterdir():
            if not p.is_dir():
                continue
            name_low = p.name.lower()
            if name_low in EXCLUDE_DIRS or name_low.startswith(".") or name_low.startswith("_"):
                continue
            classes.append(p)
        classes = sorted(classes, key=lambda x: x.name.lower())
        for c in classes:
            item = QListWidgetItem(c.name)
            item.setData(Qt.UserRole, str(c))
            self.class_list.addItem(item)
        self.status.showMessage(f"클래스 {len(classes)}개 로드 완료")
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self._on_class_clicked(self.class_list.item(0))

    def calculate_global_stats_once(self):
        """루트 선택 시 한 번만 전체 정탐/오탐/미탐 통계를 계산해서 캐싱"""
        self.cached_global_stats = {"tp": 0, "fp": 0, "fn": 0}

        if not self.results_root:
            return

        for class_dir in self.results_root.iterdir():
            if not class_dir.is_dir():
                continue
            for sub, key in [("정탐", "tp"), ("오탐", "fp"), ("미탐", "fn")]:
                p = class_dir / sub
                if p.exists():
                    self.cached_global_stats[key] += sum(
                        1 for _ in p.glob("*.jpg")
                    ) + sum(
                        1 for _ in p.glob("*.png")
                    )            

    def update_global_stats(self):
        stats = getattr(self, "cached_global_stats", None)
        if not stats:
            self.stats_label.setText("정탐: 0%  오탐: 0%  미탐: 0%")
            return

        total = stats["tp"] + stats["fp"] + stats["fn"]
        if total == 0:
            self.stats_label.setText("정탐: 0%  오탐: 0%  미탐: 0%")
            return

        tp_ratio = (stats["tp"] / total) * 100
        fp_ratio = (stats["fp"] / total) * 100
        fn_ratio = (stats["fn"] / total) * 100

        self.stats_label.setText(
            f"정탐: {tp_ratio:.1f}%    "
            f"오탐: {fp_ratio:.1f}%    "
            f"미탐: {fn_ratio:.1f}%"
        )

    def _on_class_clicked_tree(self, item, column):
        class_path = Path(item.data(0, Qt.UserRole))
        self.lbl_current_class.setText(f"클래스: {class_path.name}")
        self._load_class(class_path)
        self.update_global_stats()

    def update_stats(self):
        """전체 정탐/오탐/미탐 퍼센티지 계산"""
        tp = len(self.row_tp.images)
        fp = len(self.row_fp.images)
        fn = len(self.row_fn.images)
        total = tp + fp + fn

        if total == 0:
            self.stats_label.setText("정탐: 0%  오탐: 0%  미탐: 0%")
            return

        tp_ratio = (tp / total) * 100
        fp_ratio = (fp / total) * 100
        fn_ratio = (fn / total) * 100

        self.stats_label.setText(
            f"정탐: {tp_ratio:.1f}%  오탐: {fp_ratio:.1f}%  미탐: {fn_ratio:.1f}%"
        )

    def _load_class(self, class_dir: Path):
        p_tp = find_subfolder(class_dir, "정탐")
        p_fp = find_subfolder(class_dir, "오탐")
        p_fn = find_subfolder(class_dir, "미탐")

        imgs_tp = enumerate_images_under(p_tp)
        imgs_fp = enumerate_images_under(p_fp)
        imgs_fn = enumerate_images_under(p_fn)

        c_tp, c_fp, c_fn = len(imgs_tp), len(imgs_fp), len(imgs_fn)
        total = c_tp + c_fp + c_fn
        pct = lambda x: (x / total * 100.0) if total > 0 else 0.0

        self.lbl_summary.setText(
            f"정탐 {pct(c_tp):.1f}% | 오탐 {pct(c_fp):.1f}% | 미탐 {pct(c_fn):.1f}% (총 {total})"
        )

        self.row_tp.set_images(imgs_tp)
        self.row_fp.set_images(imgs_fp)
        self.row_fn.set_images(imgs_fn)

        self.row_tp.set_info(c_tp, pct(c_tp))
        self.row_fp.set_info(c_fp, pct(c_fp))
        self.row_fn.set_info(c_fn, pct(c_fn))

        self.update_stats()   # ✅ 전체 퍼센티지 갱신
        self.status.showMessage(f"{class_dir.name} 로드 완료")



    def keyPressEvent(self, event):
        key = event.key()

        # 정탐
        if key == Qt.Key_Q:
            self.row_tp.prev_page()
        elif key == Qt.Key_E:
            self.row_tp.next_page()

        # 미탐 or 오탐 (토글 상태에 따라 분기)
        elif key == Qt.Key_A:
            if self.btn_toggle_fp.isChecked():
                self.row_fp.prev_page()
            else:
                self.row_fn.prev_page()
        elif key == Qt.Key_D:
            if self.btn_toggle_fp.isChecked():
                self.row_fp.next_page()
            else:
                self.row_fn.next_page()


def main():
    app = QApplication(sys.argv)
    win = ResultsViewer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
