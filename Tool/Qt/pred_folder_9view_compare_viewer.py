#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, math
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QWidget,
    QGridLayout, QHBoxLayout, QVBoxLayout, QMessageBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QPainter, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QSize

# ───── 경로 설정 ─────
# ※ 실제 경로가 존재하는지 꼭 확인해주세요.
PRED_PARENT = Path(r"C:/Users/hgy84/Desktop/0513/gpu2testdata/pred_folder")

# 정규식 및 설정
VIEW_PAT    = re.compile(r"_([0-8])(?=[._])")    # 파일명 내 _0~_8 추출
ID_DIR_PAT  = re.compile(r"^\d{3,}$")            # ID 폴더명 (숫자 3자리 이상)
VALID_EXT   = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'} # 이미지 확장자 필터

# ───── 클릭 가능한 썸네일 ─────
class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)  # view_idx
    def __init__(self, view_idx=None, parent=None):
        super().__init__(parent)
        self.view_idx = view_idx
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border:1px solid #555; background-color: #000;") # 배경 검정

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.view_idx is not None:
            self.clicked.emit(self.view_idx)

def _draw_idx(pm: QPixmap, idx: int, font_px: int):
    """이미지 위에 번호 오버레이"""
    if pm.isNull(): return pm
    p = QPainter(pm)
    f = QFont("Arial", max(9, font_px)); f.setBold(True)
    p.setFont(f)
    
    # 가독성을 위해 글자 테두리(Stroke) 효과 흉내
    p.setPen(Qt.white)
    p.drawText(5, int(font_px*1.2)-1, str(idx))
    p.drawText(7, int(font_px*1.2)+1, str(idx))
    
    p.setPen(Qt.black)
    p.drawText(6, int(font_px*1.2), str(idx))
    p.end()
    return pm

def scaled_pix(src_pix: QPixmap, w: int, h: int, overlay_idx: int=None):
    """메모리에 있는 QPixmap을 리사이즈하여 반환"""
    if src_pix is None or src_pix.isNull():
        pm = QPixmap(w, h); pm.fill(Qt.black); return pm
    
    # KeepAspectRatio로 리사이즈
    pm = src_pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    if overlay_idx is not None:
        pm = _draw_idx(pm, overlay_idx, max(9, min(w, h)//12))
    return pm

class Viewer(QWidget):
    GAP = 6                 
    PAD = 4                 # 패딩 약간 추가
    TITLE_H = 24
    PANEL_GAP_2 = 20        

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pred_folder 비교 뷰어")
        self.setMinimumSize(1000, 700)
        
        # [최적화] 현재 ID의 이미지들을 메모리에 캐싱할 딕셔너리
        # 구조: self.img_cache[setting_index][view_idx] = QPixmap
        self.img_cache = {} 
        self.current_loaded_id = None # 현재 캐시에 로드된 ID

        # ── 하위 폴더 스캔 ──
        if not PRED_PARENT.exists():
            QMessageBox.critical(self, "오류", f"경로가 없습니다:\n{PRED_PARENT}")
            sys.exit(1)
            
        children = [p for p in sorted(PRED_PARENT.iterdir()) if p.is_dir()]
        if len(children) < 2:
            QMessageBox.critical(self, "오류", f"비교할 하위 폴더가 2개 이상 필요합니다.\n경로: {PRED_PARENT}")
            sys.exit(1)

        # 설정 로드 (최대 4개)
        self.settings = []
        for child in children[:4]:
            title = child.name
            pred_root = child / "images_pred"
            if not pred_root.exists():
                pred_root = child
            self.settings.append((title, pred_root))

        # ── ID 목록 구성 ──
        id_names = set()
        for _, root in self.settings:
            if not root.exists(): continue
            for d in root.iterdir():
                if not d.is_dir(): continue
                name = d.name
                if name.endswith("_labels"): continue
                if not ID_DIR_PAT.match(name): continue
                id_names.add(name)
        
        if not id_names:
            QMessageBox.critical(self, "오류", "ID 폴더(예: 0003)를 찾지 못했습니다.")
            sys.exit(1)
            
        self.id_list = sorted(id_names)
        self.idx = 0
        self.view_idx = 3
        self.mode_grid = True 

        self._build_ui()
        self._load_current_id_images() # 초기 이미지 로드
        self._refresh_ui()             # 화면 그리기

    # ───── UI ─────
    def _build_ui(self):
        # 상단 버튼 영역
        btn_prev = QPushButton("← 이전 ID (A)"); btn_next = QPushButton("다음 ID → (D)")
        btn_prev.clicked.connect(self.prev_id); btn_next.clicked.connect(self.next_id)
        self.btn_toggle = QPushButton("전체/단일 보기 토글 (G)")
        self.btn_toggle.clicked.connect(self.toggle_mode)

        top = QHBoxLayout()
        for w in (btn_prev, btn_next, self.btn_toggle): w.setFixedHeight(32)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(self.GAP)
        top.addWidget(btn_prev); top.addWidget(btn_next); top.addWidget(self.btn_toggle); top.addStretch(1)

        # 메인 그리드 영역
        n = len(self.settings)
        cols = 2 if n == 2 else max(1, int(math.ceil(math.sqrt(n))))
        
        self.containers = []
        grid_all = QGridLayout()
        grid_all.setContentsMargins(0, 0, 0, 0)

        self._panel_gap_h = self.PANEL_GAP_2 if n == 2 else self.GAP
        grid_all.setHorizontalSpacing(self._panel_gap_h)
        grid_all.setVerticalSpacing(self.GAP)

        for i, (title_text, _) in enumerate(self.settings):
            box = QGroupBox()
            box.setStyleSheet("""
            QGroupBox {
                border: 2px solid #666;
                border-radius: 6px;
                background-color: #EEE;
                margin-top: 0px;
            }
            """)

            v = QVBoxLayout()
            v.setContentsMargins(self.PAD, self.PAD, self.PAD, self.PAD)
            v.setSpacing(self.GAP)

            ttl = QLabel(title_text) 
            ttl.setFixedHeight(self.TITLE_H)
            ttl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            ttl.setStyleSheet("font-weight: bold; color:#333; border-bottom:1px solid #CCC;")

            # 3x3 그리드 뷰
            g = QGridLayout()
            g.setContentsMargins(0, 0, 0, 0)
            g.setSpacing(2) # 썸네일 간격 최소화
            labels9 = []
            for r in range(3):
                for c in range(3):
                    vi = r*3 + c
                    lbl = ClickableLabel(vi)
                    lbl.clicked.connect(self.from_grid_click)
                    g.addWidget(lbl, r, c)
                    labels9.append(lbl)

            grid_holder = QWidget(); grid_holder.setLayout(g)

            # 단일 뷰
            single = QLabel(); single.setAlignment(Qt.AlignCenter)
            single.setStyleSheet("background-color: black;")
            single_holder = QWidget(); s_lay = QVBoxLayout(single_holder)
            s_lay.setContentsMargins(0,0,0,0); s_lay.addWidget(single)

            v.addWidget(ttl); v.addWidget(grid_holder); v.addWidget(single_holder)
            box.setLayout(v)
            grid_all.addWidget(box, i//cols, i%cols)

            grid_holder.setVisible(True); single_holder.setVisible(False)

            self.containers.append({
                "title": ttl,
                "grid_holder": grid_holder,
                "grid_labels": labels9,
                "single_holder": single_holder,
                "single_label": single
            })

        main = QVBoxLayout(self)
        main.setContentsMargins(self.PAD, self.PAD, self.PAD, self.PAD)
        main.setSpacing(self.GAP)
        main.addLayout(top)
        main.addLayout(grid_all)

    # ───── 데이터 로딩 (최적화 핵심) ─────
    def _load_current_id_images(self):
        """
        ID가 변경될 때만 호출됨.
        해당 ID의 모든 뷰(0~8) 이미지를 디스크에서 읽어 메모리(self.img_cache)에 저장.
        """
        cur_id = self.id_list[self.idx]
        if self.current_loaded_id == cur_id:
            return # 이미 로드됨

        self.img_cache = {} # 초기화
        self.current_loaded_id = cur_id
        
        for set_idx, (_, pred_root) in enumerate(self.settings):
            self.img_cache[set_idx] = {}
            target_dir = pred_root / cur_id
            
            if not target_dir.exists():
                continue

            # 파일 찾기
            for fp in target_dir.iterdir():
                if fp.suffix.lower() not in VALID_EXT: continue # 이미지 확장자 필터
                
                m = VIEW_PAT.search(fp.name)
                if m:
                    v_idx = int(m.group(1))
                    # 이미지 로드 (원본 크기)
                    self.img_cache[set_idx][v_idx] = QPixmap(str(fp))

    # ───── 렌더링 (리사이즈 시 호출됨) ─────
    def _compute_thumb_size(self):
        w = max(1, self.width()); h = max(1, self.height())
        top_h = 42
        avail_w = w - self.PAD*2
        avail_h = h - top_h - self.PAD*2

        n = len(self.settings)
        cols = 2 if n == 2 else max(1, int(math.ceil(math.sqrt(n))))
        rows = int(math.ceil(n / cols))

        # 패널 하나당 가용 공간
        panel_w = (avail_w - (cols - 1) * self._panel_gap_h) / cols
        panel_h = (avail_h - (rows - 1) * self.GAP) / rows

        # 패널 내부 공간 (타이틀 제외)
        inner_h = max(1, panel_h - self.TITLE_H - self.GAP*2 - 10) # 여유분
        inner_w = max(1, panel_w - 10)

        # 3x3 썸네일 크기 계산
        tw = (inner_w - 4) / 3.0  # spacing 고려
        th = (inner_h - 4) / 3.0
        return int(max(20, min(tw, th)))

    def _refresh_ui(self):
        """메모리에 있는 이미지를 UI에 뿌려줌 (디스크 I/O 없음)"""
        cur_id = self.id_list[self.idx]
        
        # 윈도우 타이틀 업데이트
        mode = "전체보기" if self.mode_grid else f"단일(view={self.view_idx})"
        self.setWindowTitle(f"[ID: {cur_id}] {mode}  ({self.idx+1}/{len(self.id_list)}) - pred_folder 비교")

        if self.mode_grid:
            TH = self._compute_thumb_size()
            for set_idx, cont in enumerate(self.containers):
                cont["grid_holder"].setVisible(True)
                cont["single_holder"].setVisible(False)
                
                cached_imgs = self.img_cache.get(set_idx, {})
                
                for vi, lbl in enumerate(cont["grid_labels"]):
                    lbl.setFixedSize(TH, TH) # 사이즈 강제 적용
                    
                    if vi in cached_imgs:
                        lbl.setPixmap(scaled_pix(cached_imgs[vi], TH, TH, overlay_idx=vi))
                    else:
                        pm = QPixmap(TH, TH); pm.fill(Qt.black); lbl.setPixmap(pm)
        else:
            # 단일 보기 모드
            for set_idx, cont in enumerate(self.containers):
                cont["grid_holder"].setVisible(False)
                cont["single_holder"].setVisible(True)
                
                lbl = cont["single_label"]
                w = max(100, lbl.width()); h = max(100, lbl.height())
                
                cached_imgs = self.img_cache.get(set_idx, {})
                
                if self.view_idx in cached_imgs:
                    lbl.setPixmap(scaled_pix(cached_imgs[self.view_idx], w, h))
                else:
                    lbl.setText("이미지 없음")

    # ───── 이벤트 핸들러 ─────
    def from_grid_click(self, vi: int):
        self.view_idx = vi
        self.mode_grid = False
        self._refresh_ui()

    def toggle_mode(self):
        self.mode_grid = not self.mode_grid
        self._refresh_ui()

    def prev_id(self):
        self.idx = (self.idx - 1) % len(self.id_list)
        self._load_current_id_images() # ID 변경 시 디스크 로드
        self._refresh_ui()

    def next_id(self):
        self.idx = (self.idx + 1) % len(self.id_list)
        self._load_current_id_images() # ID 변경 시 디스크 로드
        self._refresh_ui()

    def next_view(self):
        self.view_idx = (self.view_idx + 1) % 9
        self.mode_grid = False
        self._refresh_ui()

    def prev_view(self):
        self.view_idx = (self.view_idx - 1) % 9
        self.mode_grid = False
        self._refresh_ui()

    def resizeEvent(self, e):
        # 리사이즈 시에는 디스크 읽기 없이 메모리 캐시에서 크기만 조절
        super().resizeEvent(e)
        self._refresh_ui()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Left, Qt.Key_A):    self.prev_id()
        elif e.key() in (Qt.Key_Right, Qt.Key_D): self.next_id()
        elif e.key() == Qt.Key_G:                 self.toggle_mode()
        elif e.key() == Qt.Key_E:                 self.next_view()
        elif e.key() == Qt.Key_Q:                 self.prev_view()
        elif e.key() == Qt.Key_Escape:            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget{font-family: 'Segoe UI', sans-serif;}")
    v = Viewer()
    v.show()
    sys.exit(app.exec_())