#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO Folder-based Label Batch Editor (Rename & Merge & Sort)
------------------------------------------------------------
- 기능: 폴더 단위 클래스 ID 일괄 변경 + 폴더명 자동 변경 (병합 기능 포함)
- 개선: 중복 폴더 병합, 진행률 표시바, 예외 처리 강화
"""

import sys
import shutil
import re
from pathlib import Path
import yaml

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QComboBox, QMessageBox, QScrollArea, QCheckBox, 
    QCompleter, QProgressBar, QGroupBox
)
from PyQt5.QtCore import Qt

# ─────────────────────────────────────────────────────────────────────────────
# [1] 개별 폴더 제어 위젯 (UI)
# ─────────────────────────────────────────────────────────────────────────────
class FolderRowWidget(QWidget):
    def __init__(self, folder_name, candidates, parent=None):
        super().__init__(parent)
        self.folder_name = folder_name
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 1. 체크박스 + 폴더명
        self.chk_select = QCheckBox(folder_name)
        self.chk_select.setChecked(False) 
        font = self.chk_select.font()
        font.setBold(True)
        font.setPointSize(10)
        self.chk_select.setFont(font)
        layout.addWidget(self.chk_select, stretch=2)

        # 2. 화살표 (시각적 구분)
        arrow_lbl = QLabel(" ➜ ")
        arrow_lbl.setStyleSheet("color: #555; font-weight: bold;")
        layout.addWidget(arrow_lbl)

        # 3. 타겟 클래스 선택 콤보박스
        self.combo = QComboBox()
        self.combo.setFixedWidth(300)
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.NoInsert)
        self.combo.addItem("--- (변경 없음) ---", None)
        
        text_list = []
        for cid, cname in candidates:
            disp_text = f"[{cid}] {cname}"
            self.combo.addItem(disp_text, cid)
            text_list.append(disp_text)

        completer = QCompleter(text_list, self.combo)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.combo.setCompleter(completer)

        # 체크박스 상태에 따라 콤보박스 활성/비활성
        self.chk_select.stateChanged.connect(self.combo.setEnabled)
        self.combo.setEnabled(False) # 기본 비활성

        layout.addWidget(self.combo, stretch=3)

    def get_data(self):
        if not self.chk_select.isChecked(): return None
        
        target_id = self.combo.currentData()
        if target_id is None: return None # 선택 안함
            
        return self.folder_name, int(target_id)
    
    def set_checked(self, state):
        self.chk_select.setChecked(state)


# ─────────────────────────────────────────────────────────────────────────────
# [2] 메인 윈도우
# ─────────────────────────────────────────────────────────────────────────────
class YOLOFolderTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 라벨 일괄 수정 & 폴더 병합 툴 (Enhanced)")
        self.resize(900, 700)

        self.root_dir = None
        self.name_map = {}   # {id: name}
        self.row_widgets = []

        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setSpacing(10)
        vbox.setContentsMargins(15, 15, 15, 15)

        # 1. 설정 그룹
        grp_setting = QGroupBox("설정")
        hbox_top = QHBoxLayout()
        
        self.btn_yaml = QPushButton("📄 1) YAML 로드")
        self.btn_yaml.clicked.connect(self.load_yaml)
        self.btn_yaml.setStyleSheet("padding: 6px;")
        
        self.btn_root = QPushButton("📂 2) 데이터셋 ROOT 선택")
        self.btn_root.clicked.connect(self.select_root)
        self.btn_root.setStyleSheet("padding: 6px;")
        
        hbox_top.addWidget(self.btn_yaml)
        hbox_top.addWidget(self.btn_root)
        grp_setting.setLayout(hbox_top)
        vbox.addWidget(grp_setting)

        # 상태 라벨
        self.lbl_status = QLabel("YAML 파일과 데이터셋 폴더를 선택해주세요.")
        self.lbl_status.setStyleSheet("color: #0055AA; font-weight: bold; margin-bottom: 5px;")
        vbox.addWidget(self.lbl_status)

        # 2. 리스트 영역 (스크롤)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        vbox.addWidget(self.scroll, 1) # stretch 1
        
        # 3. 하단 컨트롤
        hbox_bottom = QHBoxLayout()
        btn_all = QPushButton("전체 선택")
        btn_all.clicked.connect(lambda: self.toggle_all(True))
        btn_none = QPushButton("전체 해제")
        btn_none.clicked.connect(lambda: self.toggle_all(False))
        
        hbox_bottom.addWidget(btn_all)
        hbox_bottom.addWidget(btn_none)
        hbox_bottom.addStretch()
        
        self.btn_run = QPushButton("🚀 3) 변경 실행 (ID 수정 + 폴더 병합)")
        self.btn_run.setStyleSheet("background-color: #E6F4EA; font-weight: bold; padding: 10px 20px; border: 1px solid #4CAF50; color: #2E7D32;")
        self.btn_run.clicked.connect(self.run_update)
        self.btn_run.setEnabled(False)
        
        hbox_bottom.addWidget(self.btn_run)
        vbox.addLayout(hbox_bottom)

        # 진행바
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        vbox.addWidget(self.progress)

    # ──────────────────────────────────────────────────────────
    # 로직
    # ──────────────────────────────────────────────────────────
    def load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "YAML 선택", "", "YAML (*.yaml *.yml)")
        if not path: return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            names = data.get('names', {})
            if isinstance(names, list):
                self.name_map = {i: str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self.name_map = {int(k): str(v) for k, v in names.items()}
            else:
                self.name_map = {}
            
            self.lbl_status.setText(f"✅ YAML 로드 완료: {len(self.name_map)}개 클래스 감지됨.")
            if self.root_dir: self.refresh_folder_list()

        except Exception as e:
            QMessageBox.critical(self, "에러", f"YAML 로드 실패:\n{e}")

    def select_root(self):
        path = QFileDialog.getExistingDirectory(self, "ROOT 폴더 선택 (train/valid 상위)")
        if not path: return
        self.root_dir = Path(path)
        self.refresh_folder_list()

    def refresh_folder_list(self):
        if not self.root_dir: return
        
        train_dir = self.root_dir / "train"
        if not train_dir.exists():
            QMessageBox.warning(self, "경고", "'train' 폴더를 찾을 수 없습니다.\n올바른 데이터셋 루트인지 확인하세요.")
            return

        # 기존 위젯 제거
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.row_widgets.clear()

        # 폴더 스캔 & 정렬
        subfolders = [p for p in train_dir.iterdir() if p.is_dir()]
        
        def natural_key(path_obj):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', path_obj.name)]

        subfolders.sort(key=natural_key)
        candidates = sorted(self.name_map.items())

        # 위젯 생성
        for folder in subfolders:
            row = FolderRowWidget(folder.name, candidates)
            self.scroll_layout.addWidget(row)
            self.row_widgets.append(row)
        
        self.lbl_status.setText(f"📂 ROOT: {self.root_dir.name} ({len(subfolders)}개 폴더 로드됨)")
        self.btn_run.setEnabled(True)

    def toggle_all(self, state):
        for w in self.row_widgets: w.set_checked(state)

    def run_update(self):
        if not self.name_map:
            QMessageBox.warning(self, "오류", "먼저 YAML 파일을 로드해주세요.")
            return

        tasks = []
        for w in self.row_widgets:
            res = w.get_data()
            if res: tasks.append(res)

        if not tasks:
            QMessageBox.warning(self, "알림", "변경할 폴더를 하나 이상 체크하고 타겟 클래스를 선택하세요.")
            return

        msg = f"총 {len(tasks)}개 폴더 작업을 수행합니다.\n\n" \
              "1. txt 파일 내부 클래스 ID 일괄 변경\n" \
              "2. 폴더명 변경 및 중복 시 자동 병합(Merge)\n\n" \
              "정말 진행하시겠습니까?"
        
        if QMessageBox.question(self, "작업 확인", msg) != QMessageBox.Yes:
            return

        # 작업 시작
        self.progress.setVisible(True)
        self.progress.setRange(0, len(tasks))
        self.progress.setValue(0)
        self.btn_run.setEnabled(False)

        total_files = 0
        merged_folders = 0
        renamed_folders = 0
        errors = []

        for i, (old_fname, new_id) in enumerate(tasks):
            # 타겟 폴더명 (클래스 이름)
            target_cls_name = self.name_map.get(new_id, str(new_id)).strip()
            
            # 폴더명에 사용할 수 없는 특수문자 제거 (안전장치)
            target_cls_name = re.sub(r'[\\/:*?"<>|]', '_', target_cls_name)

            for split in ["train", "valid", "test"]:
                old_dir = self.root_dir / split / old_fname
                if not old_dir.exists(): continue

                # 1. 파일 ID 수정
                labels_dir = old_dir / "labels"
                if labels_dir.exists():
                    for txt in labels_dir.glob("*.txt"):
                        try:
                            if self.update_file_class(txt, new_id):
                                total_files += 1
                        except Exception as e:
                            errors.append(f"File Error ({txt.name}): {e}")

                # 2. 폴더 이동/병합
                if old_fname == target_cls_name: continue # 이름 같으면 패스

                new_dir = self.root_dir / split / target_cls_name
                
                try:
                    if new_dir.exists():
                        # [병합 로직] 기존 폴더가 있으면 내용물을 그 안으로 이동
                        self.merge_folders(old_dir, new_dir)
                        merged_folders += 1
                        # 병합 후 빈 원본 폴더 삭제
                        shutil.rmtree(old_dir)
                    else:
                        # [이름 변경] 없으면 그냥 rename
                        old_dir.rename(new_dir)
                        renamed_folders += 1
                except Exception as e:
                    errors.append(f"Folder Error ({old_fname}): {e}")

            self.progress.setValue(i + 1)
            QApplication.processEvents() # UI 멈춤 방지

        self.progress.setVisible(False)
        self.refresh_folder_list() # 리스트 갱신

        res_msg = f"✅ 작업이 완료되었습니다!\n\n" \
                  f"- 수정된 파일: {total_files}개\n" \
                  f"- 이름 변경된 폴더: {renamed_folders}개\n" \
                  f"- 병합된 폴더: {merged_folders}개"
        
        if errors:
            res_msg += f"\n\n⚠️ {len(errors)}건의 오류가 발생했습니다. (콘솔 확인)"
            for e in errors: print(e)
        
        QMessageBox.information(self, "완료", res_msg)

    def update_file_class(self, file_path, new_id):
        """txt 파일의 첫 번째 숫자를 new_id로 변경"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if not parts: continue # 빈 줄 무시
            
            # 첫 번째 요소가 숫자인지 확인 (class_id)
            if parts[0].isdigit():
                if parts[0] != str(new_id): # 다를 때만 변경
                    parts[0] = str(new_id)
                    modified = True
                new_lines.append(" ".join(parts))
            else:
                # 포맷이 이상한 줄은 그대로 유지하거나 스킵 (여기선 유지)
                new_lines.append(line.strip())
        
        if modified and new_lines:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(new_lines) + "\n")
            return True
        return False

    def merge_folders(self, src_dir, dst_dir):
        """src_dir의 모든 내용을 dst_dir로 이동 (덮어쓰기 방지)"""
        # images, labels 각각 이동
        for sub in ["images", "labels"]:
            s_sub = src_dir / sub
            d_sub = dst_dir / sub
            if not s_sub.exists(): continue
            
            d_sub.mkdir(parents=True, exist_ok=True)
            
            for src_file in s_sub.iterdir():
                if src_file.is_file():
                    dst_file = d_sub / src_file.name
                    # 중복 파일 처리 (덮어쓰지 않고 로그 남김 or 건너뜀)
                    if not dst_file.exists():
                        shutil.move(str(src_file), str(dst_file))
                    else:
                        print(f"[Merge Skip] 중복 파일 존재: {src_file.name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 윈도우 스타일 폰트 적용
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    window = YOLOFolderTool()
    window.show()
    sys.exit(app.exec_())