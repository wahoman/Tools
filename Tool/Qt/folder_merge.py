# -*- coding: utf-8 -*-
"""
Cluster Merge Tool (Improved MOVE version)
- ROOT/train 하위 폴더 리스트업
- 선택된 폴더들의 images/labels 파일을 하나의 타겟 폴더로 이동(Move)
- 라벨 파일(.txt)은 내용 이어붙이기(Append)
- 이동 후 빈 폴더는 안전하게 삭제
- 작업 로그 실시간 출력 기능 추가
"""

import sys, shutil, re
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QLabel,
    QLineEdit, QMessageBox, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt


class ClusterMergeTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cluster Merge Tool (Improved)")
        self.setGeometry(500, 200, 600, 700)

        self.root_dir = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # 1. ROOT 선택
        btn_root = QPushButton("📂 1) ROOT 폴더 선택 (train 상위)")
        btn_root.clicked.connect(self.select_root)
        btn_root.setStyleSheet("font-weight: bold; padding: 8px;")
        self.lbl_root = QLabel("선택된 경로: 없음")
        self.lbl_root.setStyleSheet("color: gray;")
        
        main_layout.addWidget(btn_root)
        main_layout.addWidget(self.lbl_root)

        # 2. 폴더 리스트 & 로그창 (Splitter로 영역 조절 가능)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 상단: 리스트
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(QLabel("2) 이동할 소스 폴더 선택 (다중 선택 가능)"))
        self.folder_list = QListWidget()
        list_layout.addWidget(self.folder_list)
        splitter.addWidget(list_widget)

        # 하단: 로그
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("📝 작업 로그"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_widget)

        main_layout.addWidget(splitter)

        # 3. 새 폴더명
        main_layout.addWidget(QLabel("3) 병합될 새 폴더명 입력"))
        self.new_folder = QLineEdit()
        self.new_folder.setPlaceholderText("예: merged_cluster_01")
        main_layout.addWidget(self.new_folder)

        # 4. 실행 버튼
        btn_merge = QPushButton("🚀 4) MERGE 실행 (Move + 원본 삭제)")
        btn_merge.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 10px;")
        btn_merge.clicked.connect(self.do_merge)
        main_layout.addWidget(btn_merge)

        self.setLayout(main_layout)

    def log(self, message):
        """로그창에 메시지 출력"""
        self.log_text.append(message)
        # 스크롤 최하단으로 이동
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
        QApplication.processEvents() # UI 갱신

    # -------------------------------------------------------
    def select_root(self):
        path = QFileDialog.getExistingDirectory(self, "ROOT 선택")
        if not path:
            return
        self.root_dir = Path(path)
        self.lbl_root.setText(str(path))
        self.load_folders()

    # -------------------------------------------------------
    def load_folders(self):
        """train 폴더 하위의 폴더들을 숫자 정렬하여 표시"""
        self.folder_list.clear()
        self.log_text.clear()

        if not self.root_dir: return

        train_dir = self.root_dir / "train"
        if not train_dir.exists():
            QMessageBox.warning(self, "오류", "'train' 폴더가 존재하지 않습니다.")
            return

        def natural_key(path_obj):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', path_obj.name)]

        try:
            folder_paths = sorted([p for p in train_dir.iterdir() if p.is_dir()], key=natural_key)
        except Exception as e:
            self.log(f"❌ 폴더 목록 로드 중 오류: {e}")
            return

        for p in folder_paths:
            item = QListWidgetItem(p.name)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.folder_list.addItem(item)
        
        self.log(f"✅ 폴더 목록 로드 완료: {len(folder_paths)}개")

    # -------------------------------------------------------
    def do_merge(self):
        if not self.root_dir:
            QMessageBox.warning(self, "오류", "ROOT 폴더를 먼저 선택하세요.")
            return

        merged_name = self.new_folder.text().strip()
        if not merged_name:
            QMessageBox.warning(self, "오류", "새 폴더명을 입력하세요.")
            return

        selected_folders = []
        for i in range(self.folder_list.count()):
            it = self.folder_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                selected_folders.append(it.text())

        if not selected_folders:
            QMessageBox.warning(self, "오류", "병합할 폴더를 하나 이상 선택하세요.")
            return

        # 병합 대상이 자기 자신인지 체크
        if merged_name in selected_folders:
             QMessageBox.warning(self, "경고", "병합될 새 폴더명이 소스 폴더명과 같습니다.\n다른 이름을 사용하세요.")
             return

        reply = QMessageBox.question(self, "확인", 
                                     f"선택한 {len(selected_folders)}개 폴더를\n'{merged_name}' 폴더로 이동하시겠습니까?\n\n(주의: 원본 폴더는 삭제됩니다)",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return

        self.log("🚀 병합 작업 시작...")

        # train, valid 각각 수행
        for mode in ["train", "valid"]:
            mode_root = self.root_dir / mode
            if mode_root.exists():
                self.move_to_merged(mode_root, selected_folders, merged_name)
            else:
                self.log(f"⚠️ '{mode}' 폴더가 없어 건너뜁니다.")

        # 작업 완료 후 새로고침
        self.log("🔄 목록 갱신 중...")
        self.load_folders()
        self.new_folder.clear()
        
        QMessageBox.information(self, "완료", "병합 및 정리가 완료되었습니다!")

    # -------------------------------------------------------
    def move_to_merged(self, src_root, folder_names, merged_name):
        """실제 이동 로직"""
        dst_root = src_root / merged_name
        img_dst = dst_root / "images"
        lbl_dst = dst_root / "labels"
        
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for fname in folder_names:
            src_folder = src_root / fname
            if not src_folder.exists():
                continue

            self.log(f"📂 처리 중: {fname}")

            # 1. 이미지 이동
            img_src = src_folder / "images"
            if img_src.exists():
                for p in img_src.glob("*"):
                    if p.is_file():
                        dst_file = img_dst / p.name
                        if dst_file.exists():
                            self.log(f"  ⚠️ 중복 이미지(건너뜀): {p.name}")
                            continue
                        try:
                            shutil.move(str(p), str(dst_file))
                        except Exception as e:
                            self.log(f"  ❌ 이미지 이동 실패: {e}")

            # 2. 라벨 이동 (내용 병합)
            lbl_src = src_folder / "labels"
            if lbl_src.exists():
                for p in lbl_src.glob("*.txt"):
                    dst_file = lbl_dst / p.name
                    try:
                        # 기존 파일이 있으면 내용 추가 (Append)
                        if dst_file.exists():
                            content = p.read_text(encoding="utf-8")
                            if content.strip():
                                with open(dst_file, "a", encoding="utf-8") as fw:
                                    if dst_file.stat().st_size > 0: # 파일이 비어있지 않으면 줄바꿈
                                        fw.write("\n")
                                    fw.write(content)
                            p.unlink() # 원본 삭제
                        else:
                            # 없으면 그냥 이동
                            shutil.move(str(p), str(dst_file))
                    except Exception as e:
                        self.log(f"  ❌ 라벨 처리 실패: {p.name} -> {e}")

            # 3. 빈 폴더 삭제 (Clean up)
            # shutil.rmtree 대신 안전하게 내부가 비었는지 확인 후 삭제
            # (혹시 이동 안 된 파일이 있을 수 있으므로)
            try:
                # images, labels 폴더 먼저 삭제 시도
                if img_src.exists() and not any(img_src.iterdir()):
                    img_src.rmdir()
                if lbl_src.exists() and not any(lbl_src.iterdir()):
                    lbl_src.rmdir()
                
                # 상위 폴더 삭제 시도
                if not any(src_folder.iterdir()):
                    src_folder.rmdir()
                    self.log(f"  🗑️ 폴더 삭제 완료: {fname}")
                else:
                    self.log(f"  ⚠️ 폴더가 비어있지 않아 삭제하지 않음: {fname}")
            except Exception as e:
                self.log(f"  ❌ 폴더 삭제 중 오류: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 폰트 가독성 설정
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
    
    w = ClusterMergeTool()
    w.show()
    sys.exit(app.exec())