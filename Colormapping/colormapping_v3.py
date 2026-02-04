# -*- coding: utf-8 -*-
import sys
import os
import sqlite3
import re
import cv2
import numpy as np
import collections

# PyQt6 라이브러리
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFileDialog, QListWidget, QListWidgetItem, QProgressBar, QPlainTextEdit,
    QLabel, QGroupBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# CuPy (GPU) 라이브러리
try:
    import cupy as cp
except ImportError:
    print("CRITICAL: CuPy가 설치되지 않았습니다. CUDA 버전에 맞게 설치하세요.")
    cp = None

# ======================================================================
# ====== Worker Thread (백그라운드 처리) ======
# ======================================================================
import collections # <--- collections 라이브러리를 import 합니다.

# ======================================================================
# ====== Worker Thread (백그라운드 처리) ======
# ======================================================================
class Worker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_root, selected_classes, db_path):
        super().__init__()
        self.input_root = input_root
        self.selected_classes = selected_classes
        self.db_path = db_path
        self.is_running = True

    def run(self):
            try:
                if cp is None:
                    self.log.emit("오류: CuPy 라이브러리를 찾을 수 없습니다. 프로그램을 종료합니다.")
                    return

                self.log.emit("GPU 장치를 초기화하고 LUT 데이터를 전송합니다...")
                self.initialize_gpu_data()
                self.log.emit("초기화 완료.")

                filename_pattern = re.compile(r'_\d{8}_(\d{8})_.*?_(\d+)\.png$', re.IGNORECASE)
                
                # --- 변경점 1: 그룹화 기준을 '클래스 폴더'로 변경 ---
                files_by_class_folder = collections.defaultdict(list)
                
                paths_to_walk = []
                for class_name in self.selected_classes:
                    paths_to_walk.append(os.path.join(self.input_root, 'train', class_name))
                    paths_to_walk.append(os.path.join(self.input_root, 'valid', class_name))

                self.log.emit("선택된 클래스 폴더에서 PNG 파일을 탐색합니다...")
                for walk_path in paths_to_walk:
                    if not os.path.isdir(walk_path): continue
                    for dirpath, _, filenames in os.walk(walk_path):
                        # 'images' 폴더 안에 있는 파일만 대상으로 함
                        if os.path.basename(dirpath).lower() != 'images':
                            continue

                        for png_filename in filenames:
                            if png_filename.lower().endswith('.png') and filename_pattern.search(png_filename):
                                # 'images' 폴더의 상위 폴더(클래스 폴더)를 키(key)로 사용
                                class_folder_path = os.path.dirname(dirpath)
                                full_path = os.path.join(dirpath, png_filename)
                                files_by_class_folder[class_folder_path].append(full_path)
                
                total_files = sum(len(files) for files in files_by_class_folder.values())
                if total_files == 0:
                    self.log.emit("선택된 폴더에서 처리할 PNG 파일을 찾지 못했습니다.")
                    return

                self.log.emit(f"총 {total_files}개의 PNG 파일을 기준으로 변환을 시작합니다.")
                
                db_conn = sqlite3.connect(self.db_path)
                processed_count = 0

                # --- 변경점 2: 클래스 폴더 단위로 루프를 실행 ---
                for class_path, full_paths_in_class in files_by_class_folder.items():
                    if not self.is_running: break

                    success_in_class = 0
                    total_in_class = len(full_paths_in_class)

                    # 클래스 폴더 내의 모든 파일들을 처리
                    for png_full_path in full_paths_in_class:
                        if not self.is_running: break
                        
                        processed_count += 1
                        # --- 변경점 3: 전체 경로에서 파일 이름 분리 ---
                        png_filename = os.path.basename(png_full_path)

                        match = filename_pattern.search(png_filename)
                        sLID, d_str = match.groups()
                        d_index = int(d_str)

                        if d_index == 0: continue
                        
                        detector_index = d_index - 1
                        target_raw_filename = f"{sLID}_{detector_index}.raw"
                        source_raw_path = self.get_raw_filepath_from_db(db_conn, target_raw_filename)

                        if not source_raw_path or not os.path.exists(source_raw_path):
                            self.log.emit(f"[유지] {png_filename} -> 원본 Raw({target_raw_filename}) 파일 없음, 원본 PNG 유지")
                            continue
                        
                        result = self.process_file(source_raw_path, detector_index)
                        if result:
                            imgH_gpu, ze_gpu = result
                            ok = self.save_pseudo_bgr_from_gpu(imgH_gpu, ze_gpu, png_full_path)
                            
                            if ok:
                                success_in_class += 1
                            else:
                                self.log.emit(f"[저장 실패] {png_filename} (원본 파일 변경 없음)")

                        self.progress.emit(int((processed_count / total_files) * 100))

                    # --- 변경점 4: 한 클래스 폴더의 작업이 끝나면 요약 로그를 출력 ---
                    if total_in_class > 0:
                        class_folder_name = os.path.basename(class_path)
                        self.log.emit(f"-> ✅ [{class_folder_name}] 완료: 총 {total_in_class}개 중 {success_in_class}개 덮어쓰기 성공")

                db_conn.close()

            except Exception as e:
                self.log.emit(f"치명적인 오류 발생: {e}")
            finally:
                self.finished.emit()

    # stop, initialize_gpu_data 및 다른 헬퍼 함수들은 기존과 동일하므로 생략...
    # (위에 제공된 run 메소드 외의 다른 함수들은 수정할 필요가 없습니다)

    def stop(self):
        self.is_running = False
        self.log.emit("작업 중단 요청...")

    # --- 아래는 기존 스크립트의 헬퍼 함수들 (변경 없음) ---
    def initialize_gpu_data(self):
        _zI0 = np.array([98000]*9, dtype=np.float32)
        _Z = np.array([[1.193710157,1.194429011,1.195497014,1.197598952,1.201261874,1.206871776,1.215205132,1.226352526,1.240297472,1.257289302,1.27849552,1.304551537,1.334194598,1.367066563,1.402681085,1.440472022,1.479718307,1.519644992,1.559941821,1.599809092,1.638299752,1.675213149,1.709797614,1.742333096,1.772344093,1.799326033,1.823810917,1.846055878,1.864917778,1.881765843]]*9, dtype=np.float32)
        _ZEFF_LUTS_CPU = self.build_zeff_luts_strict_np(_Z)
        _PSEUDO_Z_CPU, _PSEUDO_K_CPU = self.load_palettes()
        self._ZEFF_LUTS_GPU = cp.asarray(_ZEFF_LUTS_CPU)
        self._zI0_GPU = cp.asarray(_zI0, dtype=cp.float32)
        self._PSEUDO_Z_GPU = None if _PSEUDO_Z_CPU is None else cp.asarray(_PSEUDO_Z_CPU)
        self._PSEUDO_K_GPU = cp.asarray(_PSEUDO_K_CPU)

    def process_file(self, file_path, detector_index):
        WIDTH = 896
        arr = np.fromfile(file_path, dtype=np.uint16)
        if arr.size == 0 or arr.size % WIDTH != 0: return None
        img = arr.reshape(-1, WIDTH)
        rows = img.shape[0]
        if rows < 2 or (rows % 2) != 0: return None
        bkg_lvl = self.bkg_level_cpu_fast(img)
        g = cp.asarray(img, dtype=cp.float32)
        scale = 65535.0 / max(bkg_lvl, 1.0)
        g *= scale
        g = cp.clip(g, 0, 65535).astype(cp.uint16)
        vd = rows // 2
        top, bottom = g[:vd, :], g[vd:, :]
        m_top, m_bot = float(top.mean().get()), float(bottom.mean().get())
        imgH_gpu = top if m_top >= m_bot else bottom
        imgL_gpu = bottom if m_top >= m_bot else top
        ze_gpu = self.zeff_exact_gpu_u16(imgL_gpu, imgH_gpu, detector_index)
        return imgH_gpu, ze_gpu

    def get_raw_filepath_from_db(self, db_conn, raw_filename):
        cursor = db_conn.cursor()
        cursor.execute("SELECT filepath FROM files WHERE filename = ?", (raw_filename,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def load_palettes(self):
        pseudoZ, pseudoK = None, np.zeros((256, 3), dtype=np.uint8)
        try:
            with open("LUT_ZeffU2.csv", "rb") as f:
                rows = [r.strip() for r in f.read().decode("utf-16").splitlines() if r.strip()]
            if len(rows) >= 300:
                pseudoZ = np.zeros((300, 256, 3), dtype=np.uint8)
                for j in range(300):
                    parts = rows[j].split(",")
                    if len(parts) < 256*3: continue
                    arr = np.array(parts[:256*3], dtype=np.int16).reshape(256, 3)
                    pseudoZ[j] = np.clip(arr, 0, 255).astype(np.uint8)
        except FileNotFoundError: self.log.emit("WARN: LUT_ZeffU_0908.csv not found")
        try:
            with open("gray.clut", "rb") as f: buf = f.read()[8:]
            pseudoK = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 4)[:256, :3].copy()
        except FileNotFoundError: self.log.emit("WARN: gray.clut not found")
        return pseudoZ, pseudoK

    def build_zeff_luts_strict_np(self, _Z_np):
        luts = np.zeros((9, 2000), dtype=np.uint16)
        noffset = 1000
        for d in range(9):
            nZ = np.zeros(31, dtype=np.int32)
            vals = np.maximum(((_Z_np[d] + 0.0005) * 1000).astype(np.int32) - noffset, 0)
            nZ[:30], nZ[30] = vals, 2000
            i=0
            for _ in range(int(nZ[0])):
                if i >= 2000: break
                luts[d, i], i = (1 << 8), i + 1
            for j in range(30):
                if i >= 2000: break
                nW = int(nZ[j+1] - nZ[j])
                if nW <= 0: continue
                k = np.arange(nW, dtype=np.int32)
                seg = ((j+1) << 8) + ((k << 8) // nW)
                end = min(i + nW, 2000)
                luts[d, i:end] = seg[:(end - i)].astype(np.uint16)
                i = end
            while i < 2000: luts[d, i], i = (30 << 8), i + 1
        return luts

    def bkg_level_cpu_fast(self, arr):
        bin_center, bin_gap = 50000, 16
        if arr is None or arr.size == 0: return float(bin_center)
        idx = np.floor((arr.astype(np.int32) - bin_center) / bin_gap + 0.5 + 128).astype(np.int32)
        np.clip(idx, 0, 255, out=idx)
        hist = np.bincount(idx.ravel(), minlength=256)
        if hist.sum() == 0: return float(bin_center)
        peak = int(np.argmax(hist))
        return float(bin_center + bin_gap * (peak + 0.5 - 128))

    def zeff_exact_gpu_u16(self, imgLoE_gpu, imgHiE_gpu, d):
        Ilow, Ihigh = cp.maximum(imgLoE_gpu.astype(cp.float32), 1.0), cp.maximum(imgHiE_gpu.astype(cp.float32), 1.0)
        zI0 = self._zI0_GPU[d]
        num, den = cp.log(zI0 / Ilow), cp.log(zI0 / Ihigh)
        den = cp.where(den <= 1e-12, 1e-12, den)
        rate = cp.clip(((num/den - 1.0 + 0.0005) * 1000.0).astype(cp.int32), 0, 1999)
        return self._ZEFF_LUTS_GPU[d][rate].astype(cp.uint16)

    def save_pseudo_bgr_from_gpu(self, imgH_gpu, ze_gpu, output_filepath):
        H, W = imgH_gpu.shape
        if H == 0 or W == 0: return False
        pbSrc = (imgH_gpu >> 8).astype(cp.uint16)
        nzeff = cp.clip(((ze_gpu.astype(cp.uint32) * 10) // 256).astype(cp.int32) - 1, 0, self._PSEUDO_Z_GPU.shape[0]-1)
        mask_gray = nzeff < 0
        rgb = cp.where(mask_gray[..., None], self._PSEUDO_K_GPU[pbSrc], self._PSEUDO_Z_GPU[nzeff, pbSrc])
        bgr = cp.stack([rgb[...,2], rgb[...,1], rgb[...,0]], axis=2)
        outBGR = cp.asnumpy(bgr)
        try:
            ext = os.path.splitext(output_filepath)[1]
            result, buf = cv2.imencode(ext, outBGR)
            if result:
                with open(output_filepath, 'wb') as f: f.write(buf)
                return True
        except Exception as e: self.log.emit(f"[파일 쓰기 오류] {os.path.basename(output_filepath)}: {e}")
        return False

# ======================================================================
# ====== PyQt GUI Application ======
# ======================================================================
class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Raw to PNG Converter (In-place)')
        self.setGeometry(100, 100, 700, 600)
        main_layout = QVBoxLayout(self)

        path_group = QGroupBox("경로 설정")
        path_layout = QVBoxLayout(path_group)
        main_layout.addWidget(path_group)

        self.input_root_edit = self.create_path_selector(path_layout, "기준 폴더 (train/valid 상위)", self.browse_input_root)
        self.input_root_edit.textChanged.connect(self.populate_class_list)
        self.db_path_edit = self.create_path_selector(path_layout, "데이터베이스 파일", self.browse_db_file, is_file=True)
        # <--- 변경됨: 최종 저장 폴더 선택 UI 제거
        # self.output_path_edit = self.create_path_selector(path_layout, "최종 저장 폴더", self.browse_output_folder)

        folder_group = QGroupBox("처리 대상 클래스 선택")
        folder_layout = QVBoxLayout(folder_group)
        main_layout.addWidget(folder_group)

        self.class_list_widget = QListWidget()
        folder_layout.addWidget(self.class_list_widget)
        self.select_all_button = QPushButton("전체 선택")
        self.select_all_button.clicked.connect(self.toggle_select_all)
        folder_layout.addWidget(self.select_all_button)

        action_group = QGroupBox("실행 및 로그")
        action_layout = QVBoxLayout(action_group)
        main_layout.addWidget(action_group)

        self.progress_bar = QProgressBar()
        action_layout.addWidget(self.progress_bar)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        action_layout.addWidget(self.log_edit)
        self.run_button = QPushButton("변환 시작 (덮어쓰기)")
        self.run_button.clicked.connect(self.start_conversion)
        action_layout.addWidget(self.run_button)
        
        main_layout.setStretch(2, 1)

    def create_path_selector(self, parent_layout, label_text, browse_func, is_file=False):
        layout = QHBoxLayout()
        edit = QLineEdit()
        button = QPushButton("찾아보기...")
        button.clicked.connect(lambda: browse_func(edit, is_file))
        layout.addWidget(QLabel(label_text))
        layout.addWidget(edit)
        layout.addWidget(button)
        parent_layout.addLayout(layout)
        return edit

    def browse_input_root(self, edit, is_file):
        dir_path = QFileDialog.getExistingDirectory(self, "기준 폴더 선택")
        if dir_path: edit.setText(dir_path)

    def browse_db_file(self, edit, is_file):
        file_path, _ = QFileDialog.getOpenFileName(self, "DB 파일 선택", "", "Database Files (*.db)")
        if file_path: edit.setText(file_path)

    # <--- 변경됨: browse_output_folder 함수는 더 이상 필요 없음
    # def browse_output_folder(self, edit, is_file):
    #     dir_path = QFileDialog.getExistingDirectory(self, "최종 저장 폴더 선택")
    #     if dir_path: edit.setText(dir_path)

    def populate_class_list(self):
        self.class_list_widget.clear()
        root_path = self.input_root_edit.text()
        if not os.path.isdir(root_path): return

        train_path = os.path.join(root_path, 'train')
        valid_path = os.path.join(root_path, 'valid')

        if not os.path.isdir(train_path) or not os.path.isdir(valid_path):
            self.log_edit.setPlainText("오류: 기준 폴더 내에 'train'과 'valid' 폴더가 모두 존재해야 합니다.")
            return

        try:
            train_classes = {d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))}
            valid_classes = {d for d in os.listdir(valid_path) if os.path.isdir(os.path.join(valid_path, d))}
            
            common_classes = sorted(list(train_classes.intersection(valid_classes)))
            
            self.log_edit.clear()
            if not common_classes:
                self.log_edit.setPlainText("train과 valid 폴더에 공통된 클래스 폴더가 없습니다.")
                return

            for class_name in common_classes:
                item = QListWidgetItem(class_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.class_list_widget.addItem(item)
        except OSError as e:
            self.log_edit.setPlainText(f"폴더를 읽는 중 오류 발생: {e}")

    def toggle_select_all(self):
        all_checked = all(self.class_list_widget.item(i).checkState() == Qt.CheckState.Checked for i in range(self.class_list_widget.count()))
        new_state = Qt.CheckState.Unchecked if all_checked else Qt.CheckState.Checked
        for i in range(self.class_list_widget.count()): self.class_list_widget.item(i).setCheckState(new_state)
        self.select_all_button.setText("전체 해제" if new_state == Qt.CheckState.Checked else "전체 선택")

    def start_conversion(self):
        # <--- 변경됨: output_path 관련 변수 및 로직 제거
        input_root, db_path = self.input_root_edit.text(), self.db_path_edit.text()
        selected_classes = [self.class_list_widget.item(i).text() for i in range(self.class_list_widget.count()) if self.class_list_widget.item(i).checkState() == Qt.CheckState.Checked]

        if not all([input_root, db_path]):
            self.log_edit.appendPlainText("오류: 기준 폴더와 데이터베이스 경로를 설정해야 합니다.")
            return
        if not selected_classes:
            self.log_edit.appendPlainText("오류: 처리할 클래스를 하나 이상 선택해야 합니다.")
            return
        
        self.run_button.setText("처리 중... (중단하려면 클릭)")
        self.run_button.setEnabled(True)
        self.run_button.clicked.disconnect()
        self.run_button.clicked.connect(self.stop_conversion)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        # <--- 변경됨: Worker 생성 시 output_path 인자 제거
        self.worker = Worker(input_root, selected_classes, db_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log_edit.appendPlainText)
        self.worker.finished.connect(self.on_conversion_finished)
        self.worker.start()

    def stop_conversion(self):
        if self.worker:
            self.worker.stop()
            self.run_button.setText("중단 중...")
            self.run_button.setEnabled(False)

    def on_conversion_finished(self):
        self.log_edit.appendPlainText("\n======== 작업 완료 ========")
        self.progress_bar.setValue(100)
        self.run_button.setText("변환 시작 (덮어쓰기)")
        self.run_button.setEnabled(True)
        self.run_button.clicked.disconnect()
        self.run_button.clicked.connect(self.start_conversion)
        self.worker = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())