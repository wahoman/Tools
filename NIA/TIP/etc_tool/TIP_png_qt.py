import sys
import os
import random
import re
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFileDialog, QListWidget, QListWidgetItem, QProgressBar, QPlainTextEdit,
    QLabel, QGroupBox, QSpinBox, QMessageBox, QSplitter, QCheckBox, QGridLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# ── [합성 로직] ─────────────────────────────────────────────────────
def overlay_images(bg, fg, x, y):
    h, w = fg.shape[:2]
    roi = bg[y:y+h, x:x+w]
    
    if fg.shape[2] == 4:
        alpha = fg[:, :, 3] / 255.0
        fg_bgr = fg[:, :, :3]
        for c in range(3):
            roi[:, :, c] = (alpha * fg_bgr[:, :, c] + (1.0 - alpha) * roi[:, :, c])
    else:
        fg_gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(fg_gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg_fg = cv2.bitwise_and(fg, fg, mask=mask)
        roi = cv2.add(bg_bg, fg_fg)

    bg[y:y+h, x:x+w] = roi.astype(np.uint8)
    return bg

# ── [백그라운드 워커] ───────────────────────────────────────────────
class Worker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, settings, classes):
        super().__init__()
        self.s = settings
        self.selected_classes = classes
        self.is_running = True

    def get_file_index(self, filename):
        """파일명 뒤의 숫자 추출 (문자열 반환)"""
        match = re.search(r'_(\d+)\.[a-zA-Z]+$', filename)
        if match:
            return match.group(1) 
        return None

    def run(self):
        self.log.emit(">>> 작업 시작: 배경 이미지 인덱싱 중...")
        
        # 1. 배경 이미지 로드
        bg_root = Path(self.s['bg_root'])
        bg_files = list(bg_root.glob("**/*.png")) + list(bg_root.glob("**/*.jpg"))
        
        bg_map = {} 
        for p in bg_files:
            idx = self.get_file_index(p.name)
            if idx:
                # 인덱스 문자열 그대로 키 사용 ('0', '1' ...)
                if idx not in bg_map: bg_map[idx] = []
                bg_map[idx].append(str(p))
        
        if not bg_map:
            self.log.emit("❌ 오류: 배경 이미지를 찾을 수 없습니다.")
            self.finished.emit()
            return
            
        bg_indices = sorted(list(bg_map.keys()), key=lambda x: int(x))
        self.log.emit(f"✅ 배경 그룹: {len(bg_map)}개 (인덱스: {bg_indices})")

        # 사용자가 선택한 TIP 뷰 (예: ['1', '2', '3'])
        user_selected_views = [str(v) for v in self.s['target_views']]
        self.log.emit(f"🎯 사용자 선택 TIP 뷰: {user_selected_views}")

        total_classes = len(self.selected_classes)
        total_target_cnt = self.s['target_count']

        for cls_idx, cls_name in enumerate(self.selected_classes):
            if not self.is_running: break
            
            self.log.emit(f"\n📂 [{cls_name}] 매칭 계산 중...")
            tip_dir = Path(self.s['tip_root']) / cls_name
            tip_files = list(tip_dir.glob("*.png"))
            
            tip_map = {}
            for p in tip_files:
                idx = self.get_file_index(p.name)
                if idx:
                    if idx not in tip_map: tip_map[idx] = []
                    tip_map[idx].append(str(p))
            
            if not tip_map:
                self.log.emit(f"   ⚠️ 스킵: {cls_name} 폴더 비어있음")
                continue

            # ── [핵심 로직] 매칭 쌍(Pair) 찾기 (BG = TIP - 1) ──
            # valid_pairs: [(bg_idx, tip_idx), ...] 리스트
            valid_pairs = []
            
            # 가지고 있는 TIP 인덱스를 순회하며 짝이 맞는 배경이 있는지 확인
            for tip_idx in tip_map.keys():
                # 1. 사용자가 선택한 뷰인지 확인
                if tip_idx not in user_selected_views:
                    continue
                
                try:
                    # 2. 배경 인덱스 계산 (TIP - 1)
                    target_bg_idx = str(int(tip_idx) - 1)
                    
                    # 3. 해당 배경이 존재하는지 확인
                    if target_bg_idx in bg_map:
                        valid_pairs.append((target_bg_idx, tip_idx))
                except ValueError:
                    continue
            
            # 정렬 (TIP 번호 기준)
            valid_pairs.sort(key=lambda x: int(x[1]))

            if not valid_pairs:
                self.log.emit(f"   ⚠️ 매칭 실패: {cls_name} (TIP-1 = BG 규칙을 만족하는 쌍이 없음)")
                self.log.emit(f"      보유 TIP: {list(tip_map.keys())}")
                continue

            # ── [균등 분배] ──
            num_pairs = len(valid_pairs)
            base_count = total_target_cnt // num_pairs
            remainder = total_target_cnt % num_pairs
            
            self.log.emit(f"   ℹ️ 매칭 성공 쌍(BG, TIP): {valid_pairs}")
            
            class_created_total = 0
            
            # 저장 폴더
            out_img_dir = Path(self.s['dst_root']) / "images" / cls_name
            out_lbl_dir = Path(self.s['dst_root']) / "labels" / cls_name
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            # 각 쌍별로 생성
            for i, (bg_key, tip_key) in enumerate(valid_pairs):
                if not self.is_running: break

                count_for_this_pair = base_count + (1 if i < remainder else 0)
                
                for _ in range(count_for_this_pair):
                    if not self.is_running: break

                    # 1. 랜덤 선택
                    bg_p = random.choice(bg_map[bg_key])
                    tip_p = random.choice(tip_map[tip_key])
                    
                    # 2. 읽기
                    bg = cv2.imread(bg_p, cv2.IMREAD_COLOR)
                    tip = cv2.imread(tip_p, cv2.IMREAD_UNCHANGED)

                    if bg is None or tip is None: continue

                    # 3. 리사이즈 & 합성
                    s = random.uniform(0.4, 1.0)
                    if tip.shape[0] * s < 10 or tip.shape[1] * s < 10: continue
                    tip = cv2.resize(tip, (0,0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                    
                    bh, bw = bg.shape[:2]
                    th, tw = tip.shape[:2]
                    
                    if bw < tw or bh < th: continue 

                    x = random.randint(0, bw - tw - 1)
                    y = random.randint(0, bh - th - 1)

                    overlay_images(bg, tip, x, y)

                    # 4. 저장 (파일명: TIP번호와 BG번호 명시)
                    # 예: Knife_T1_B0_00001.png
                    filename_str = f"{cls_name}_T{tip_key}_B{bg_key}_{class_created_total:05d}"
                    
                    stem = out_img_dir / filename_str
                    cv2.imwrite(str(stem.with_suffix(".png")), bg)

                    cx, cy = (x + tw/2)/bw, (y + th/2)/bh
                    nw, nh = tw/bw, th/bh
                    with (out_lbl_dir / f"{filename_str}.txt").open("w") as f:
                        f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                    class_created_total += 1
                    
                    if class_created_total % 100 == 0:
                        overall_progress = int((cls_idx / total_classes * 100) + (class_created_total / total_target_cnt * (100 / total_classes)))
                        self.progress.emit(overall_progress)

            self.log.emit(f"   ✅ {cls_name} 완료: {class_created_total}장")

        self.progress.emit(100)
        self.log.emit("\n🎉 모든 작업이 완료되었습니다!")
        self.finished.emit()

    def stop(self):
        self.is_running = False

# ── [메인 GUI] ──────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TIP 합성기 (TIP-1 = BG 매칭)")
        self.resize(1000, 800)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 1. 경로
        grp_path = QGroupBox("경로 설정")
        lay_path = QVBoxLayout()
        self.edt_tip_root = self.create_file_input(lay_path, "TIP 폴더 (bare_image_crop)", "D:/hgyeo/BCAS_TIP/bare_image_crop")
        self.edt_bg_root = self.create_file_input(lay_path, "배경 폴더 (Bag5_ColorPNG)", "D:/hgyeo/BCAS_TIP/APIDS Bare Bags_ColorPNG")
        self.edt_dst_root = self.create_file_input(lay_path, "결과 저장 폴더", "D:/hgyeo/BCAS_TIP/TIP_output")
        grp_path.setLayout(lay_path)
        main_layout.addWidget(grp_path)

        # 2. 옵션 (수량 + TIP 뷰 선택)
        grp_opt = QGroupBox("합성 설정")
        lay_opt = QVBoxLayout()
        
        lay_cnt = QHBoxLayout()
        lay_cnt.addWidget(QLabel("클래스당 목표 수량:"))
        self.spn_target = QSpinBox()
        self.spn_target.setRange(1, 1000000)
        self.spn_target.setValue(10000)
        self.spn_target.setSingleStep(100)
        lay_cnt.addWidget(self.spn_target)
        lay_opt.addLayout(lay_cnt)
        
        line = QLabel(); line.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        lay_opt.addWidget(line)

        # TIP 뷰 선택 (1~9)
        # 대상이 1~9 이므로 체크박스도 1~9로 생성
        lay_opt.addWidget(QLabel("합성할 대상(TIP) 뷰 선택 (자동으로 BG = TIP-1 매칭):"))
        lay_views = QGridLayout()
        self.chk_views = []
        # 1번부터 9번까지 생성
        for i in range(1, 10): 
            chk = QCheckBox(f"TIP {i}")
            chk.setChecked(True) 
            self.chk_views.append(chk) # 리스트 인덱스 0 -> TIP 1, 인덱스 1 -> TIP 2 ...
            lay_views.addWidget(chk, 0, i-1)
        
        lay_opt.addLayout(lay_views)
        grp_opt.setLayout(lay_opt)
        main_layout.addWidget(grp_opt)

        # 3. 클래스 & 로그
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        grp_cls = QGroupBox("클래스 목록")
        lay_cls = QVBoxLayout()
        lay_btns = QHBoxLayout()
        self.btn_load_cls = QPushButton("새로고침")
        self.btn_load_cls.clicked.connect(self.load_classes)
        self.btn_select_all = QPushButton("전체 선택/해제")
        self.btn_select_all.clicked.connect(self.toggle_select_all)
        lay_btns.addWidget(self.btn_load_cls)
        lay_btns.addWidget(self.btn_select_all)
        self.list_cls = QListWidget()
        lay_cls.addLayout(lay_btns)
        lay_cls.addWidget(self.list_cls)
        grp_cls.setLayout(lay_cls)
        
        grp_log = QGroupBox("로그")
        lay_log = QVBoxLayout()
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        lay_log.addWidget(self.txt_log)
        grp_log.setLayout(lay_log)

        splitter.addWidget(grp_cls)
        splitter.addWidget(grp_log)
        splitter.setSizes([350, 650])
        main_layout.addWidget(splitter, stretch=1)

        lay_btm = QVBoxLayout()
        self.pbar = QProgressBar()
        self.btn_start = QPushButton("합성 시작")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("font-weight: bold; font-size: 16px; background-color: #4CAF50; color: white;")
        self.btn_start.clicked.connect(self.start_process)
        lay_btm.addWidget(self.pbar)
        lay_btm.addWidget(self.btn_start)
        main_layout.addLayout(lay_btm)

        self.setLayout(main_layout)
        self.load_classes()

    def create_file_input(self, layout, label, default=""):
        hlay = QHBoxLayout()
        hlay.addWidget(QLabel(label))
        edt = QLineEdit(default)
        btn = QPushButton("찾기")
        btn.clicked.connect(lambda: self.browse_dir(edt))
        hlay.addWidget(edt)
        hlay.addWidget(btn)
        layout.addLayout(hlay)
        return edt

    def browse_dir(self, edt):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택", edt.text())
        if path: edt.setText(path)
        if edt == self.edt_tip_root: self.load_classes()

    def load_classes(self):
        tip_root = self.edt_tip_root.text()
        self.list_cls.clear()
        if not os.path.isdir(tip_root): return
        dirs = [d for d in os.listdir(tip_root) if os.path.isdir(os.path.join(tip_root, d))]
        for d in sorted(dirs):
            item = QListWidgetItem(d)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list_cls.addItem(item)
        self.txt_log.appendPlainText(f"클래스 {len(dirs)}개 로드 완료.")

    def toggle_select_all(self):
        count = self.list_cls.count()
        if count == 0: return
        first_item = self.list_cls.item(0)
        new_state = Qt.CheckState.Unchecked if first_item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
        for i in range(count):
            self.list_cls.item(i).setCheckState(new_state)

    def start_process(self):
        # 선택된 TIP 뷰 (1~9)
        selected_views = []
        for i, chk in enumerate(self.chk_views):
            if chk.isChecked():
                # chk_views[0]은 TIP 1, chk_views[1]은 TIP 2 ...
                selected_views.append(i + 1)

        settings = {
            'tip_root': self.edt_tip_root.text(),
            'bg_root': self.edt_bg_root.text(),
            'dst_root': self.edt_dst_root.text(),
            'target_count': self.spn_target.value(),
            'target_views': selected_views
        }

        selected_classes = []
        for i in range(self.list_cls.count()):
            item = self.list_cls.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_classes.append(item.text())

        if not selected_classes:
            QMessageBox.warning(self, "경고", "클래스를 선택해주세요.")
            return
        
        if not selected_views:
            QMessageBox.warning(self, "경고", "대상(TIP) 뷰를 하나 이상 선택해주세요.")
            return

        self.btn_start.setEnabled(False)
        self.btn_start.setText("작업 중...")
        self.txt_log.clear()
        self.pbar.setValue(0)
        
        self.worker = Worker(settings, selected_classes)
        self.worker.progress.connect(self.pbar.setValue)
        self.worker.log.connect(self.txt_log.appendPlainText)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("합성 시작")
        QMessageBox.information(self, "완료", "작업이 종료되었습니다.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())