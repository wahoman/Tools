#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data_classified  →  base_data_by_class 로 클래스별 YOLO 이미지·라벨 ‘이동’ GUI
- 개선: 전체 선택, 기본 경로 설정, 로그 자동 스크롤, 빈 폴더 정리
"""

import sys, yaml, shutil, os
from pathlib import Path
from collections import defaultdict
from PyQt5 import QtWidgets, QtCore, QtGui

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ═════ 사용자 기본 경로 설정 (편의성) ═════
DEFAULT_SRC = ""  # 예: r"C:\Data\Data_classified"
DEFAULT_DST = ""  # 예: r"C:\Data\base_data_by_class"
DEFAULT_YAML = "" # 예: r"C:\Data\data.yaml"
# ══════════════════════════════════════════

# ────────────────────────── util ──────────────────────────

def load_yaml(path: Path) -> dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(f"YAML 파일이 없습니다: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {i: (v or f"cls_{i}") for i, v in enumerate(names)}
    raise ValueError("YAML에 names 항목이 없습니다.")


def yolo_pairs(root: Path, split: str, cls: str):
    """split/<cls>/images|labels 에서 (img, txt) 튜플 yield"""
    img_dir = root / split / cls / "images"
    lbl_dir = root / split / cls / "labels"
    if not img_dir.exists():
        return
    # generator 대신 list로 반환하여 파일 처리 중 디렉토리 변경 오류 방지
    files = []
    for img in img_dir.iterdir():
        if img.suffix.lower() in IMG_EXTS:
            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                files.append((img, lbl))
    return files


# ───────────────────────── GUI ────────────────────────────
class ClassMover(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 클래스 데이터 이동기 (Improved)")
        self.resize(900, 600)

        # 1. 상단 경로 설정 영역
        grp_path = QtWidgets.QGroupBox("경로 설정")
        layout_path = QtWidgets.QGridLayout()
        
        self.src_edit = QtWidgets.QLineEdit(DEFAULT_SRC); self.src_edit.setReadOnly(True)
        self.dst_edit = QtWidgets.QLineEdit(DEFAULT_DST); self.dst_edit.setReadOnly(True)
        self.yaml_edit = QtWidgets.QLineEdit(DEFAULT_YAML); self.yaml_edit.setReadOnly(True)
        
        btn_src = QtWidgets.QPushButton("📂 소스 폴더"); btn_src.clicked.connect(self.pick_src)
        btn_dst = QtWidgets.QPushButton("📂 타깃 폴더"); btn_dst.clicked.connect(self.pick_dst)
        btn_yaml = QtWidgets.QPushButton("📄 YAML 파일"); btn_yaml.clicked.connect(self.pick_yaml)

        layout_path.addWidget(QtWidgets.QLabel("Source:"), 0, 0)
        layout_path.addWidget(self.src_edit, 0, 1)
        layout_path.addWidget(btn_src, 0, 2)
        
        layout_path.addWidget(QtWidgets.QLabel("Target:"), 1, 0)
        layout_path.addWidget(self.dst_edit, 1, 1)
        layout_path.addWidget(btn_dst, 1, 2)
        
        layout_path.addWidget(QtWidgets.QLabel("YAML:"), 2, 0)
        layout_path.addWidget(self.yaml_edit, 2, 1)
        layout_path.addWidget(btn_yaml, 2, 2)
        grp_path.setLayout(layout_path)

        # 2. 옵션 영역
        grp_opt = QtWidgets.QGroupBox("옵션")
        layout_opt = QtWidgets.QHBoxLayout()
        self.chk_train = QtWidgets.QCheckBox("Train 포함"); self.chk_train.setChecked(True)
        self.chk_valid = QtWidgets.QCheckBox("Valid 포함"); self.chk_valid.setChecked(True)
        self.chk_cleanup = QtWidgets.QCheckBox("이동 후 빈 폴더 삭제"); self.chk_cleanup.setChecked(True)
        
        self.btn_select_all = QtWidgets.QPushButton("전체 선택")
        self.btn_select_all.clicked.connect(lambda: self.toggle_all(True))
        self.btn_deselect_all = QtWidgets.QPushButton("전체 해제")
        self.btn_deselect_all.clicked.connect(lambda: self.toggle_all(False))

        layout_opt.addWidget(self.chk_train)
        layout_opt.addWidget(self.chk_valid)
        layout_opt.addWidget(self.chk_cleanup)
        layout_opt.addStretch()
        layout_opt.addWidget(self.btn_select_all)
        layout_opt.addWidget(self.btn_deselect_all)
        grp_opt.setLayout(layout_opt)

        # 3. 테이블
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["선택", "클래스명", "Train 잔여", "Train 이동량", "Valid 잔여", "Valid 이동량"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        # 4. 하단 실행 및 로그
        self.move_btn = QtWidgets.QPushButton("🚀 이동 실행")
        self.move_btn.setStyleSheet("font-weight: bold; font-size: 14px; height: 40px; background-color: #E1F5FE;")
        self.move_btn.clicked.connect(self.do_move)
        
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background-color: #F5F5F5; font-family: Consolas;")

        # 메인 레이아웃 조합
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(grp_path)
        main_layout.addWidget(grp_opt)
        main_layout.addWidget(self.table, 1) # 테이블이 공간 차지
        main_layout.addWidget(self.move_btn)
        main_layout.addWidget(self.log, 0) # 로그는 적당히

        # 데이터 초기화
        self.id2name = {}
        self.remain = defaultdict(lambda: {"train": 0, "valid": 0})

        # 시그널 연결 추가
        self.chk_train.stateChanged.connect(self.update_column_visibility)
        self.chk_valid.stateChanged.connect(self.update_column_visibility)
        self.table.itemChanged.connect(self.on_item_changed)

        # 초기값 있으면 로드 시도
        if DEFAULT_SRC and Path(DEFAULT_SRC).exists(): self.src_root = Path(DEFAULT_SRC); self.refresh_table()
        if DEFAULT_DST: self.dst_root = Path(DEFAULT_DST)
        if DEFAULT_YAML and Path(DEFAULT_YAML).exists(): 
            try:
                self.id2name = load_yaml(Path(DEFAULT_YAML))
                self.refresh_table()
            except: pass

    # ───────────── 경로 선택 ─────────────
    def pick_src(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "소스 폴더 선택")
        if d:
            self.src_edit.setText(d)
            self.refresh_table()

    def pick_dst(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "타깃 폴더 선택")
        if d: self.dst_edit.setText(d)

    def pick_yaml(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "YAML 선택", "", "YAML (*.yaml *.yml)")
        if f:
            self.yaml_edit.setText(f)
            try:
                self.id2name = load_yaml(Path(f))
                self.refresh_table()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "오류", str(e))

    # ──────────── 테이블 로직 ────────────
    def refresh_table(self):
        src_path = self.src_edit.text()
        if not src_path or not self.id2name: return

        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.remain.clear()

        # 이름순 정렬
        sorted_cls = sorted(self.id2name.values())

        for cls in sorted_cls:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # 0: 체크박스
            chk = QtWidgets.QTableWidgetItem()
            chk.setCheckState(QtCore.Qt.Unchecked)
            self.table.setItem(row, 0, chk)

            # 1: 클래스명
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(cls))

            # 2, 4: 잔여량 (초기값 '-')
            item_tr_rem = QtWidgets.QTableWidgetItem("-"); item_tr_rem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 2, item_tr_rem)
            
            item_va_rem = QtWidgets.QTableWidgetItem("-"); item_va_rem.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 4, item_va_rem)

            # 3, 5: 이동량 (SpinBox)
            sp_tr = QtWidgets.QSpinBox(); sp_tr.setRange(0, 0); sp_tr.setAlignment(QtCore.Qt.AlignCenter)
            sp_va = QtWidgets.QSpinBox(); sp_va.setRange(0, 0); sp_va.setAlignment(QtCore.Qt.AlignCenter)
            self.table.setCellWidget(row, 3, sp_tr)
            self.table.setCellWidget(row, 5, sp_va)

        self.table.blockSignals(False)
        self.update_column_visibility()

    def toggle_all(self, state):
        """전체 선택/해제"""
        check_state = QtCore.Qt.Checked if state else QtCore.Qt.Unchecked
        self.table.blockSignals(True) # 대량 변경 시 시그널 차단 필수
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setCheckState(check_state)
            # 체크되면 수량 계산 (수동 호출)
            if state: self.calculate_row(row)
        self.table.blockSignals(False)

    def on_item_changed(self, item):
        """개별 체크박스 변경 시 호출"""
        if item.column() == 0 and item.checkState() == QtCore.Qt.Checked:
            self.calculate_row(item.row())

    def calculate_row(self, row):
        """해당 행의 파일 수 계산 (이미 계산됐으면 스킵)"""
        if self.table.item(row, 2).text() != "-": return

        cls = self.table.item(row, 1).text()
        src_root = Path(self.src_edit.text())

        # 커서 변경 (계산 중임을 알림)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # yolo_pairs가 리스트를 반환하도록 수정됨
            tr_files = yolo_pairs(src_root, "train", cls) or []
            va_files = yolo_pairs(src_root, "valid", cls) or []
            
            tr = len(tr_files)
            va = len(va_files)

            self.remain[cls]["train"] = tr
            self.remain[cls]["valid"] = va

            self.table.item(row, 2).setText(str(tr))
            self.table.item(row, 4).setText(str(va))
            
            self.table.cellWidget(row, 3).setRange(0, tr)
            self.table.cellWidget(row, 5).setRange(0, va)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def update_column_visibility(self):
        show_tr = self.chk_train.isChecked()
        show_va = self.chk_valid.isChecked()
        self.table.setColumnHidden(2, not show_tr)
        self.table.setColumnHidden(3, not show_tr)
        self.table.setColumnHidden(4, not show_va)
        self.table.setColumnHidden(5, not show_va)

    # ───────────── 이동 실행 ─────────────
    def do_move(self):
        src_txt = self.src_edit.text()
        dst_txt = self.dst_edit.text()
        if not src_txt or not dst_txt:
            QtWidgets.QMessageBox.warning(self, "경고", "소스 및 타깃 폴더를 설정하세요.")
            return

        total_moved = 0
        src_root = Path(src_txt)
        dst_root = Path(dst_txt)

        self.log.append("🚀 이동 시작...")
        
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != QtCore.Qt.Checked:
                continue
            
            cls = self.table.item(row, 1).text()
            
            # SpinBox 값 가져오기
            n_tr = self.table.cellWidget(row, 3).value() if self.chk_train.isChecked() else 0
            n_va = self.table.cellWidget(row, 5).value() if self.chk_valid.isChecked() else 0

            if n_tr > 0: total_moved += self.move_files(src_root, dst_root, "train", cls, n_tr)
            if n_va > 0: total_moved += self.move_files(src_root, dst_root, "valid", cls, n_va)

            # UI 갱신 (잔여량)
            cur_tr = self.remain[cls]["train"] - n_tr
            cur_va = self.remain[cls]["valid"] - n_va
            self.remain[cls]["train"] = max(0, cur_tr)
            self.remain[cls]["valid"] = max(0, cur_va)
            
            self.table.item(row, 2).setText(str(self.remain[cls]["train"]))
            self.table.item(row, 4).setText(str(self.remain[cls]["valid"]))
            
            # 이동한 만큼 최대값 줄이기 & 값 0으로 리셋
            self.table.cellWidget(row, 3).setRange(0, self.remain[cls]["train"])
            self.table.cellWidget(row, 3).setValue(0)
            self.table.cellWidget(row, 5).setRange(0, self.remain[cls]["valid"])
            self.table.cellWidget(row, 5).setValue(0)

        self.log.append(f"✅ 총 {total_moved}개 파일 세트 이동 완료!")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum()) # 자동 스크롤

    def move_files(self, src_root, dst_root, split, cls, count):
        files = yolo_pairs(src_root, split, cls)
        if not files: return 0
        
        # 이름순 정렬
        files.sort(key=lambda p: p[0].name)
        target_files = files[:count]
        
        moved_cnt = 0
        for img, txt in target_files:
            try:
                # 타깃 경로 생성
                d_img = dst_root / split / cls / "images"
                d_lbl = dst_root / split / cls / "labels"
                d_img.mkdir(parents=True, exist_ok=True)
                d_lbl.mkdir(parents=True, exist_ok=True)

                shutil.move(str(img), str(d_img / img.name))
                shutil.move(str(txt), str(d_lbl / txt.name))
                moved_cnt += 1
            except Exception as e:
                self.log.append(f"❌ 오류 ({img.name}): {e}")

        self.log.append(f" -> {cls} ({split}): {moved_cnt}개 이동됨")

        # 빈 폴더 정리 옵션
        if self.chk_cleanup.isChecked():
            self.cleanup_empty_dirs(src_root / split / cls)

        return moved_cnt

    def cleanup_empty_dirs(self, cls_dir: Path):
        """images, labels 폴더가 비었으면 삭제"""
        if not cls_dir.exists(): return
        for sub in ["images", "labels"]:
            d = cls_dir / sub
            if d.exists() and not any(d.iterdir()):
                try: 
                    d.rmdir()
                except: pass
        # 클래스 폴더 자체도 비었으면 삭제
        if not any(cls_dir.iterdir()):
            try: cls_dir.rmdir()
            except: pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # 폰트 가독성
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    win = ClassMover()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()