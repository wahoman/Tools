#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json, re
from pathlib import Path
from typing import Optional  # 호환성을 위해 추가

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QGridLayout, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSplitter, QSizePolicy, QInputDialog
)
from PyQt5.QtGui import QPixmap, QTextCursor
from PyQt5.QtCore import Qt

# ====== 경로 설정 ======
# ※ 실제 경로가 존재하는지 꼭 확인해주세요.
IMAGES_ROOT   = Path(r"C:\Users\hgy84\Desktop\0513\gpu2testdata\images")            # 원본 9뷰
PRED_IMG_ROOT = Path(r"C:\Users\hgy84\Downloads\images_pred")                      # 예측 PNG/JSON 루트
GT_JSON_PATH  = IMAGES_ROOT / "gt_folder_level.json"                               # GT JSON 경로

# 파일명에서 뷰 인덱스 추출: *_<숫자>.* 패턴
_VIEW_NUM_RE = re.compile(r'_(\d+)(?:\D|$)')

def extract_view_index_from_name(path: Path, fallback_idx: int) -> int:
    m = _VIEW_NUM_RE.search(path.stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return fallback_idx

def iter_pred_jsons(images_root: Path, pred_img_root: Path, folder_name: str):
    bases = []
    if pred_img_root:
        bases.append(pred_img_root)
    bases += [images_root.parent, images_root]

    for base in bases:
        for predroot in base.glob("*_pred"):
            labeldir = predroot / f"{folder_name}_labels"
            if labeldir.is_dir():
                for jp in sorted(labeldir.glob("*.json")):
                    yield jp

    for base in bases:
        labeldir = base / f"{folder_name}_labels"
        if labeldir.is_dir():
            for jp in sorted(labeldir.glob("*.json")):
                yield jp

    for base in bases:
        predroot = base / f"{folder_name}_pred"
        if predroot.is_dir():
            for jp in sorted(predroot.glob("*.json")):
                yield jp

# Python 3.9 이하 호환성을 위해 Type Hint 수정 (Path | None -> Optional[Path])
def find_pred_image_for(original_img_path: Path, pred_folder: Path) -> Optional[Path]:
    if not pred_folder.exists():
        return None
    same = pred_folder / original_img_path.name
    if same.exists():
        return same
    idx = extract_view_index_from_name(original_img_path, fallback_idx=-1)
    if idx >= 0:
        candidates = sorted([p for p in pred_folder.iterdir()
                             if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
                             and f"_{idx}" in p.stem])
        if candidates:
            return candidates[0]
    return None

def collect_images_sorted_by_view(folder: Path):
    if not folder.exists():
        return []
    imgs = [p for p in folder.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
    def keyfunc(p: Path):
        m = _VIEW_NUM_RE.search(p.stem)
        return (int(m.group(1)) if m else 1_000_000, p.name.lower())
    return sorted(imgs, key=keyfunc)


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("9-View Viewer (S toggles left & right to _pred, J to jump folder)")
        self.setGeometry(80, 60, 1800, 1000)

        # 경로가 없으면 빈 리스트 처리
        if IMAGES_ROOT.exists():
            self.folder_list = sorted(
                [p for p in IMAGES_ROOT.iterdir()
                 if p.is_dir() and not p.name.endswith("_pred") and not p.name.endswith("_labels")]
            )
        else:
            self.folder_list = []
            print(f"경고: IMAGES_ROOT 경로를 찾을 수 없습니다: {IMAGES_ROOT}")

        self.folder_idx = 0
        self.view_idx = 0
        self.use_pred_mode = False   # True면 왼쪽 썸네일 + 오른쪽 미리보기도 _pred 기준

        # ===== 왼쪽: 3×3 썸네일 =====
        self.left_container = QWidget()
        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        self.left_container.setLayout(self.grid)

        self.labels = []
        for i in range(9):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: white;")
            lbl.setMinimumSize(250, 200)
            lbl.mousePressEvent = self._make_onclick(i)
            self.labels.append(lbl)

        for i, lbl in enumerate(self.labels):
            self.grid.addWidget(lbl, i // 3, i % 3)

        # ===== 오른쪽: 위-아래 분할 =====
        self.right_preview = QLabel()
        self.right_preview.setAlignment(Qt.AlignCenter)
        self.right_preview.setStyleSheet("background-color: white;")
        self.right_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.result_edit = QTextEdit()
        self.result_edit.setReadOnly(True)
        self.result_edit.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.result_edit.setAcceptDrops(True)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.right_preview)
        right_split.addWidget(self.result_edit)
        right_split.setSizes([600, 400])

        main = QHBoxLayout()
        main.addWidget(self.left_container, stretch=3)
        main.addWidget(right_split, stretch=2)
        self.setLayout(main)

        self.images = []
        self._load_folder()

    # -------- 상호작용 ----------
    def _make_onclick(self, idx):
        def onclick(event):
            if not self.images:
                return
            self.view_idx = min(idx, len(self.images) - 1)
            self._update_right_preview()
        return onclick

    def _current_folder_for_mode(self) -> Path:
        base = PRED_IMG_ROOT if self.use_pred_mode else IMAGES_ROOT
        return base / self.folder_name

    def _load_GT(self):
        """GT를 매번 다시 로드해서 최신 반영."""
        try:
            if GT_JSON_PATH.exists():
                with open(GT_JSON_PATH, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _load_folder(self):
        if not self.folder_list:
            for lbl in self.labels:
                lbl.setText("폴더 없음")
            self.right_preview.setText("이미지 없음")
            self.result_edit.setPlainText("폴더 없음")
            return

        cur_folder = self.folder_list[self.folder_idx]
        self.folder_name = cur_folder.name

        img_folder = self._current_folder_for_mode()
        if self.use_pred_mode and not img_folder.exists():
            self.use_pred_mode = False
            img_folder = self._current_folder_for_mode()

        self.images = collect_images_sorted_by_view(img_folder)

        for i, lbl in enumerate(self.labels):
            lbl.clear()
            lbl.setStyleSheet("background-color: white; color: gray;")
            if i < len(self.images):
                pix = QPixmap(str(self.images[i])).scaled(
                    lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                lbl.setPixmap(pix)
            else:
                lbl.setText("(없음)")

        self.view_idx = min(self.view_idx, max(0, len(self.images) - 1)) if self.images else 0
        self._update_right_preview()
        self._update_results_text()

    def _update_right_preview(self):
        if not self.images:
            self.right_preview.setText("이미지 없음")
            return
        show_path = self.images[self.view_idx]
        if self.use_pred_mode and not show_path.exists():
            # 안전 폴백: 원본에서 대응 예측 찾기
            orig_images = collect_images_sorted_by_view(IMAGES_ROOT / self.folder_name)
            if self.view_idx < len(orig_images):
                alt = find_pred_image_for(orig_images[self.view_idx], PRED_IMG_ROOT / self.folder_name)
                if alt:
                    show_path = alt
        pix = QPixmap(str(show_path)).scaled(
            self.right_preview.width(), self.right_preview.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.right_preview.setPixmap(pix)

    def _update_results_text(self):
        """오른쪽 하단 결과창: 탐지율 요약 + 정답/탐지 표(가독성 개선)."""
        # 최신 GT 로드
        try:
            if GT_JSON_PATH.exists():
                with open(GT_JSON_PATH, encoding="utf-8") as f:
                    GT_DICT = json.load(f)
            else:
                GT_DICT = {}
        except Exception:
            GT_DICT = {}

        gt_dict = GT_DICT.get(self.folder_name, {})
        gt_classes = sorted(set(gt_dict.keys()))

        # 예측 JSON에서 '클래스별 탐지된 뷰 인덱스' 수집
        detected_views_by_class = {}
        found_any = False
        for idx, jp in enumerate(iter_pred_jsons(IMAGES_ROOT, PRED_IMG_ROOT, self.folder_name)):
            found_any = True
            v = extract_view_index_from_name(jp, fallback_idx=idx)
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            
            # predictions 혹은 objects 키 처리
            preds = data.get("predictions", None)
            if preds is None and "objects" in data:
                preds = data["objects"]
            
            if not preds:
                continue

            for obj in preds:
                cname = obj.get("class_name") or obj.get("name") or obj.get("class")
                if not cname:
                    continue
                s = detected_views_by_class.setdefault(cname, set())
                s.add(v)

        # ── 탐지율 계산(정답 클래스 중 1회 이상 탐지된 클래스 비율) ──
        total_gt = len(gt_classes)
        detected_gt = sum(1 for c in gt_classes if len(detected_views_by_class.get(c, set())) > 0)
        rate = (detected_gt / total_gt * 100.0) if total_gt > 0 else 0.0
        rate_str = f"{detected_gt} / {total_gt} ({rate:.0f}%)" if total_gt > 0 else "0 / 0 (0%)"

        # 정답에 없는데 탐지된 클래스
        extra = sorted([c for c in detected_views_by_class.keys() if c not in gt_classes])

        # ── HTML 구성(가독성 개선) ──
        mode_str = "_pred" if self.use_pred_mode else "원본"
        html = []

        # 기본 스타일
        html.append("""
        <style>
        .summary { border:1px solid #ddd; padding:10px; margin:6px 0 12px 0; background:#fafafa; }
        .title { font-weight:bold; margin:2px 0 8px 0; }
        .kv { font-size:14px; }
        .bar { width:100%; height:10px; background:#eee; border:1px solid #ddd; border-radius:4px; overflow:hidden; }
        .barfill { height:100%; background:#8ecae6; }
        table { border-collapse:collapse; margin-top:6px; }
        th, td { border:1px solid #bbb; padding:6px 8px; font-size:14px; }
        th { background:#f1f1f1; }
        tr:nth-child(even) td { background:#fcfcfc; }
        .nodet { background:#f6f6f6; color:#666; }
        .subhead { margin-top:10px; font-weight:bold; }
        .mono { font-family: Consolas, 'Courier New', monospace; }
        </style>
        """)

        # 헤더 + 요약(탐지율)
        html.append(f"<div class='mono'><b>폴더:</b> {self.folder_name} &nbsp;&nbsp; <b>표시 모드(S):</b> {mode_str}</div>")
        html.append("<hr>")
        html.append("<div class='summary'>")
        html.append("<div class='title'>탐지율</div>")
        html.append(f"<div class='kv mono' style='margin-bottom:6px;'>정답 중 탐지됨: <b>{rate_str}</b></div>")
        # progress bar
        html.append("<div class='bar'><div class='barfill' style='width: %d%%;'></div></div>" % (rate if rate <= 100 else 100))
        html.append("</div>")

        # 정답 클래스 & 탐지된 뷰
        html.append("<div class='title'>정답 클래스 & 탐지된 뷰</div>")
        if gt_classes:
            html.append("<table>")
            html.append("<tr><th>클래스</th><th>탐지된 뷰 인덱스</th></tr>")
            for cls in gt_classes:
                views = sorted(detected_views_by_class.get(cls, set()))
                if views:
                    vstr = ", ".join(map(str, views))
                    html.append(f"<tr><td>{cls}</td><td class='mono'>{vstr}</td></tr>")
                else:
                    html.append(f"<tr><td>{cls}</td><td class='mono nodet'>(없음)</td></tr>")
            html.append("</table>")
        else:
            html.append("<div class='mono'>(GT 없음)</div>")

        # 정답에 없는데 탐지된 클래스
        html.append("<div class='subhead'>정답에 없는데 탐지된 클래스</div>")
        if extra:
            html.append("<table>")
            html.append("<tr><th>클래스</th><th>뷰 인덱스</th></tr>")
            for cls in extra:
                views = sorted(detected_views_by_class[cls])
                vstr = ", ".join(map(str, views))
                html.append(f"<tr><td>{cls}</td><td class='mono'>{vstr}</td></tr>")
            html.append("</table>")
        else:
            html.append("<div class='mono'>(없음)</div>")

        if not found_any:
            html.append("<div class='mono' style='margin-top:8px;color:#666;'>※ 예측 JSON을 찾지 못했습니다. "
                        " *_pred/&lt;folder&gt;_labels/*.json, &lt;folder&gt;_labels/*.json, &lt;folder&gt;_pred/*.json</div>")

        # === 새 섹션: 'GT 템플릿(검출 빈도 내림차순)' 출력 (한 줄 버전) ===
        freq_pairs = []
        for cls, vset in detected_views_by_class.items():
            freq_pairs.append((cls, len(vset)))
        freq_pairs.sort(key=lambda x: (-x[1], x[0].lower()))
        cur = self.folder_name

        html.append("<div class='subhead' style='margin-top:14px;'>GT 템플릿 (검출 빈도 내림차순)</div>")
        if freq_pairs:
            inner = ", ".join([f"\"{cls}\": \"1\"" for cls, _freq in freq_pairs])
            line = f"\"{cur}\": {{{inner}}}"
            html.append("<pre style='background:#f7f7f7;border:1px solid #ddd;padding:8px;"
                        "white-space:pre;overflow:auto;margin-top:6px;'>")
            html.append(line)
            html.append("</pre>")
        else:
            html.append("<div class='mono'>(현재 폴더에서 검출된 클래스가 없습니다)</div>")

        self.result_edit.setHtml("".join(html))
        self.result_edit.moveCursor(QTextCursor.Start)

    # 리사이즈 시 재스케일
    def resizeEvent(self, event):
        super().resizeEvent(event)
        for i, lbl in enumerate(self.labels):
            if i < len(self.images) and lbl.pixmap() is not None:
                pix = QPixmap(str(self.images[i])).scaled(
                    lbl.width(), lbl.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                lbl.setPixmap(pix)
        self._update_right_preview()

    # ---- 폴더 점프 ----
    def _goto_folder_by_name(self, name: str) -> bool:
        """이름 정확 일치 → 숫자면 4자리 zero-pad 시도 → 부분일치(첫 번째) 순으로 이동."""
        names = [p.name for p in self.folder_list]
        # 정확 일치
        if name in names:
            self.folder_idx = names.index(name)
            self.view_idx = 0
            self._load_folder()
            return True
        # 숫자면 4자리 zero-pad 시도
        if name.isdigit() and len(name) < 4:
            pad = name.zfill(4)
            if pad in names:
                self.folder_idx = names.index(pad)
                self.view_idx = 0
                self._load_folder()
                return True
        # 부분 일치(앞에서부터)
        for i, n in enumerate(names):
            if name in n:
                self.folder_idx = i
                self.view_idx = 0
                self._load_folder()
                return True
        return False

    # 단축키
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_S:
            self.use_pred_mode = not self.use_pred_mode
            self._load_folder()

        elif e.key() == Qt.Key_D:
            if self.folder_list:
                self.folder_idx = (self.folder_idx + 1) % len(self.folder_list)
                self.view_idx = 0
                self._load_folder()

        elif e.key() == Qt.Key_A:
            if self.folder_list:
                self.folder_idx = (self.folder_idx - 1) % len(self.folder_list)
                self.view_idx = 0
                self._load_folder()

        elif e.key() == Qt.Key_E:
            if self.images:
                self.view_idx = (self.view_idx + 1) % len(self.images)
                self._update_right_preview()

        elif e.key() == Qt.Key_Q:
            if self.images:
                self.view_idx = (self.view_idx - 1) % len(self.images)
                self._update_right_preview()

        elif e.key() == Qt.Key_J:
            # ★ 폴더 점프
            text, ok = QInputDialog.getText(self, "폴더 이동", "폴더명(예: 0003 또는 숫자):")
            if ok and text.strip():
                moved = self._goto_folder_by_name(text.strip())
                if not moved:
                    # 간단 피드백
                    self.result_edit.append(f"<br><i>폴더 '{text}' 를 찾지 못했습니다.</i>")

        elif e.key() == Qt.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Viewer()
    w.show()
    sys.exit(app.exec_())