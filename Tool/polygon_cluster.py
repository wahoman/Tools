#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_polygon_cluster_no_color.py
- 기반: 두 번째 코드 (수정된 Contour 로직 + Polygon Append 저장 방식)
- 수정: Color Feature 및 Gain 제거 (Shape + CNN 모드로 복귀)
- 수정: 타이틀에 min_cluster_size 표시
"""

import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as patheffects
import os, cv2, math, shutil, warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import torch, torch.nn as nn
import torchvision.transforms as T
import umap, hdbscan
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 설정
# =========================
BASE_DIR = Path("/home/hgyeo/Desktop/awl_knife_Driver/data")
OUTPUT_DIR = Path("/home/hgyeo/Desktop/awl_knife_Driver/data_output2")
IMG_SIZE = 224
CROP_PADDING = 16
BATCH_SIZE = 64

BACKBONE    = "dino_v2"     # "clip-vit-b32" or "dino_v2"
USE_CNN    = True
AMP        = True
SAVE_NOISE = True

# 가중치 재설정 (Color 제외 후 재분배)
SHAPE_GAIN = 0.5  # 형태 정보 비중 축소
CNN_GAIN   = 0.5  # 질감/내부 패턴 정보 비중 확대

# 군집/시각화 파라미터
CLUSTER_EPS       = 0.00             # 비슷한 군집 더 잘 합쳐지게
UMAP_N_COMPONENTS = 2                    
UMAP_METRIC       = 'cosine'             
torch.backends.cudnn.benchmark = True

# UMAP 파라미터
UMAP_N_NEIGHBORS = 20   
UMAP_MIN_DIST    = 0.0  

# =========================
# 공용 유틸
# =========================
def rotate_affine(img: np.ndarray, deg: float, border=128) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(border,border,border) if img.ndim==3 else border)

def rotate_affine_mask(mask, deg):
    h,w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(mask, M, (w, h), borderValue=255)

# =========================
# Polygon crop + 회전 정렬
# =========================
def crop_polygon_mask(img: np.ndarray, coords: List[float], crop_padding: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    if (pts <= 1).all():
        pts[:, 0] *= w
        pts[:, 1] *= h

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.int32)
    W, H = int(rect[1][0]), int(rect[1][1])
    if W == 0 or H == 0:
        return None, None

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, H-1], [0, 0], [W-1, 0], [W-1, H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned   = cv2.warpPerspective(masked, M, (W, H))
    aligned_m = cv2.warpPerspective(mask,   M, (W, H))

    bg = np.full_like(aligned, 128)
    aligned = np.where(aligned_m[..., None] > 0, aligned, bg)

    aligned = cv2.copyMakeBorder(aligned, crop_padding, crop_padding, crop_padding, crop_padding,
                                 cv2.BORDER_CONSTANT, value=(128,128,128))
    aligned_m = cv2.copyMakeBorder(aligned_m, crop_padding, crop_padding, crop_padding, crop_padding,
                                   cv2.BORDER_CONSTANT, value=0)
# [수정 코드] 비율 유지 리사이즈 적용
    aligned = resize_with_padding(aligned, target_size=IMG_SIZE, pad_value=128)
    aligned_m = resize_with_padding(aligned_m, target_size=IMG_SIZE, pad_value=0)
    return aligned, aligned_m

# =========================
# 데이터 순회
# =========================
def iter_label_image_pairs(base_dir: Path, splits=("train", "valid")):
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists(): continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir(): continue
            images_dir = class_dir / "images"
            labels_dir = class_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists(): continue
            for lbl in labels_dir.glob("*.txt"):
                stem = lbl.stem
                img = None
                for ext in (".jpg", ".png", ".jpeg"):
                    cand = images_dir / f"{stem}{ext}"
                    if cand.exists():
                        img = cand; break
                if img:
                    yield split, img, lbl

# =========================
# 📊 UMAP + Cluster Visualization (범례 제거 버전 + 타이틀 수정)
# =========================
def visualize_clusters(embedding, labels, output_dir, n_clusters,
                       n_neighbors, min_dist, eps, shape_gain, cnn_gain,
                       min_cluster_size): # [수정] 인자 추가됨

    # 범례가 없으므로 굳이 가로로 길 필요 없음. 정사각형 비율 추천
    plt.figure(figsize=(10, 10))
    
    unique_labels = np.unique(labels)
    
    # 클러스터가 많으면 색상 구분이 잘 되는 'tab20'이나 'turbo' 추천
    if len(unique_labels) > 20:
        base_colors = sns.color_palette("tab20", 20)
        colors = base_colors * (len(unique_labels) // 20 + 1)
        colors = colors[:len(unique_labels)]
    else:
        colors = sns.color_palette("tab10", len(unique_labels))

    # 1. 점 찍기
    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=12, c=[color], alpha=0.6, edgecolors='none'
        )

    # 2. 텍스트 라벨 (맵 위에 숫자 표시)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue 

        mask = labels == cluster_id
        cx = embedding[mask, 0].mean()
        cy = embedding[mask, 1].mean()

        plt.text(
            cx, cy,
            f"{cluster_id}",
            fontsize=11, fontweight='bold', 
            ha='center', va='center',
            color='black',
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
        )

    # [수정] 타이틀에 min_cluster_size 추가
    plt.title(
        f"UMAP Clusters ({n_clusters} detected)\n"
        f"n_neighbors={n_neighbors}, min_dist={min_dist}, eps={eps}\n"
        f"min_cluster_size={min_cluster_size}\n" # <--- 이 부분 추가됨
        f"shape={shape_gain}, cnn={cnn_gain} (No Color)",
        fontsize=14
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    
    plt.tight_layout()
    plt.savefig(output_dir / "umap_cluster_visualization.png", dpi=300)
    plt.close()
    
    print("✅ Saved cluster visualization (No Legend)")

def visualize_clusters_3d(embedding, labels, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=8,
            c=color.reshape(1, -1),
            alpha=0.8
        )

    ax.set_title("UMAP 3D Clusters", fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    plt.tight_layout()
    plt.savefig(output_dir / "umap_3d_clusters.png", dpi=300)
    plt.close()
    print("✅ Saved: umap_3d_clusters.png")

def visualize_clusters_3d_interactive(embedding, labels, output_dir):
    fig = px.scatter_3d(
        x=embedding[:,0],
        y=embedding[:,1],
        z=embedding[:,2],
        color=labels.astype(str),
        opacity=0.75,
        size=[1]*len(labels)
    )

    fig.update_traces(marker=dict(size=1))

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        if lab == -1:
            continue
        pts = embedding[labels == lab]
        cx, cy, cz = pts.mean(axis=0)

        fig.add_scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode='text',
            text=[f"Cluster {lab}"],
            textposition="middle center",
            hovertemplate="Cluster %{text}<extra></extra>"
        )

    fig.update_layout(title="UMAP 3D Interactive Clusters with Hover Labels", showlegend=False)
    html_path = output_dir / "umap_3d_interactive.html"
    fig.write_html(str(html_path))
    print(f"✅ Interactive 3D saved: {html_path}")

def read_yolo_polygon_lines(label_path: Path):
    polys = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 3: continue
            coords = list(map(float, parts[1:]))
            if len(coords) >= 6:
                polys.append(coords)
    return polys

# =========================
# 형태 피처 (수정된 Contour 로직 유지)
# =========================
def _largest_contour(binary: np.ndarray):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    return max(cnts, key=cv2.contourArea)

def _projection_lengths_sorted(cnt: np.ndarray, k: int = 8) -> np.ndarray:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    lengths = []
    for t in range(k):
        theta = np.pi * t / k
        d = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        proj = pts @ d
        lengths.append(float(proj.max() - proj.min()))
    arr = np.array(lengths, dtype=np.float32)
    arr.sort()
    if arr.max() > 0: arr /= arr.max()
    return arr

def _band_width_profile_invariant(binary: np.ndarray, bins: int = 32) -> np.ndarray:
    h, _ = binary.shape
    widths = (binary > 0).sum(axis=1).astype(np.float32)
    idx = np.linspace(0, h-1, bins)
    prof = np.interp(idx, np.arange(h), widths)
    inv = np.maximum(prof, prof[::-1])
    if inv.max() > 0: inv /= inv.max()
    return inv.astype(np.float32)

def _radial_histogram(cnt: np.ndarray, bins: int = 16) -> np.ndarray:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1)
    m = d.max()
    d = d / m if m > 0 else d + 1e-6
    hist, _ = np.histogram(d, bins=bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float32)

def _basic_shape_scalars(cnt: np.ndarray, binary: np.ndarray) -> np.ndarray:
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return np.zeros(6, dtype=np.float32)
    x,y,w,h = cv2.boundingRect(cnt)
    rect_area = float(w*h) if w*h>0 else 1.0
    extent = area / rect_area
    hull = cv2.convexHull(cnt); hull_area = float(cv2.contourArea(hull)) or 1.0
    solidity = area / hull_area
    peri = float(cv2.arcLength(cnt, True)) or 1.0
    circularity = (4.0*np.pi*area)/(peri*peri)
    (cx,cy),(W,H),_ = cv2.minAreaRect(cnt)
    aspect = (W/H) if H>0 else 0.0
    if aspect < 1: aspect = 1.0/aspect if aspect>0 else 0.0
    pts = cnt.reshape(-1,2).astype(np.float32)
    pts_c = pts - pts.mean(axis=0)
    cov = np.cov(pts_c.T)
    evals, _ = np.linalg.eig(cov); evals = np.sort(np.real(evals))[::-1]
    ecc = float(np.sqrt(evals[0]/evals[1])) if len(evals)>=2 and evals[1]>0 else 0.0
    mask_area = float((binary>0).sum())
    fill_ratio = mask_area / rect_area
    return np.array([aspect, extent, solidity, circularity, ecc, fill_ratio], dtype=np.float32)

def compute_shape_features_from_crop(crop_bgr: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    EMPTY_DIM = 214   # 두 번째 코드의 차원 설정 유지

    binary = (crop_mask > 0).astype(np.uint8) * 255
    cnt = _largest_contour(binary)
    if cnt is None or len(cnt) < 5:
        return np.zeros(EMPTY_DIM, dtype=np.float32)

    # Hu Moments
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    obj = cv2.bitwise_and(gray, gray, mask=binary)
    edges = cv2.Canny(obj, 80, 160)
    hu = cv2.HuMoments(cv2.moments(edges)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    hu = hu.astype(np.float32)

    # Proj, Band, Radial
    proj   = _projection_lengths_sorted(cnt, k=32)
    band   = _band_width_profile_invariant(binary, bins=96)
    radial = _radial_histogram(cnt, bins=32)
    scalars = _basic_shape_scalars(cnt, binary)

    # Neck Index
    h = binary.shape[0]
    top_w = (binary[:h//3] > 0).sum(axis=1).mean() if (binary[:h//3]>0).any() else 0.0
    mid_w = (binary[h//3:2*h//3] > 0).sum(axis=1).mean() if (binary[h//3:2*h//3]>0).any() else 0.0
    neck_idx = float((top_w+1e-6)/(mid_w+1e-6))

    # Blurred Edge Hu
    obj_blur = cv2.GaussianBlur(obj, (5,5), 0)
    edges2 = cv2.Canny(obj_blur, 50, 120)
    hu2 = cv2.HuMoments(cv2.moments(edges2)).flatten()
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-12)
    hu2 = hu2.astype(np.float32)

    # Distance Transform Hist
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    dist_hist = np.histogram(dist, bins=32, range=(0, dist.max() if dist.max()>0 else 1))[0].astype(np.float32)

    # Ellipse Ratio
    try:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        ellipse_ratio = np.array([ma / MA], dtype=np.float32)
    except:
        ellipse_ratio = np.array([0.0], dtype=np.float32)

    return np.concatenate([
        hu, hu2,                
        proj, band, radial,     
        scalars, 
        dist_hist,              
        ellipse_ratio,          
        np.array([neck_idx], dtype=np.float32)
    ]).astype(np.float32)

def rotation_invariant_shape(crop_bgr, crop_m):
    rots = list(range(0, 360, 15))
    feats = []
    for deg in rots:
        rot_img  = rotate_affine(crop_bgr, deg)
        rot_mask = rotate_affine_mask(crop_m, deg)
        f = compute_shape_features_from_crop(rot_img, rot_mask)
        # 차원이 깨지면 skip
        if f.shape[0] != 214:
            continue
        feats.append(f)

    if len(feats) == 0:
        return np.zeros(214, dtype=np.float32)

    return np.mean(feats, axis=0)

# =========================
# CNN 임베더 (DINOv2)
# =========================
class Embedder(nn.Module):
    def __init__(self, backbone="dino_v2", img_size=224, amp=True):
        super().__init__()
        self.amp = amp and torch.cuda.is_available()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if backbone == "dino_v2":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
            self.preprocess = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.out_dim = 384
        else:
            self.model = None
            self.preprocess = None
            self.out_dim = 0

    @torch.no_grad()
    def forward(self, batch_imgs: List[np.ndarray]) -> np.ndarray:
        if self.model is None or not batch_imgs:
            return np.empty((0, self.out_dim), dtype=np.float32)
        imgs = [self.preprocess(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in batch_imgs]
        x = torch.stack(imgs).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            feats = self.model(x)
        return feats.float().cpu().numpy()

@torch.no_grad()
def rotated_features_average(embedder: Embedder, crop_bgr: np.ndarray) -> np.ndarray:
    if embedder.model is None:
        return np.empty((0,), dtype=np.float32)
    rots = list(range(0, 360, 15))
    imgs = [rotate_affine(crop_bgr, deg) for deg in rots]
    f = embedder(imgs)   
    return f.mean(axis=0)

# ==========================================
# [추가] 병렬 처리를 위한 Dataset 클래스
# ==========================================
from torch.utils.data import Dataset, DataLoader
import multiprocessing

class PolygonDataset(Dataset):
    def __init__(self, rows, crop_padding):
        self.rows = rows
        self.crop_padding = crop_padding

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        split, img_path, lbl_path, coords = self.rows[idx]
        
        # 1. 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            return None 

        # 2. 크롭
        crop_bgr, crop_m = crop_polygon_mask(img, coords, self.crop_padding)
        if crop_bgr is None:
            return None

        # 3. Shape Feature (여기서 병렬 연산됨)
        # ✅ [수정] 30도 간격 (12회) - 대소문자 일치시킴
        rots = list(range(0, 360, 15)) 
        
        feats = []
        for deg in rots:  # 위에서 선언한 변수명(rots)과 일치해야 함
            rot_img  = rotate_affine(crop_bgr, deg)
            rot_mask = rotate_affine_mask(crop_m, deg)
            f = compute_shape_features_from_crop(rot_img, rot_mask)
            
            # 차원(214)이 깨지면 제외
            if f.shape[0] == 214:
                feats.append(f)
        
        if len(feats) > 0:
            shape_feat = np.mean(feats, axis=0)
        else:
            shape_feat = np.zeros(214, dtype=np.float32)

        # 4. 반환
        return {
            "crop_bgr": crop_bgr,
            "shape_feat": shape_feat,
            "meta": (split, str(img_path), str(lbl_path), str(coords)) 
        }

def collate_fn(batch):
    # 로드 실패한 항목(None) 제거
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    crops = [b['crop_bgr'] for b in batch]
    shapes = np.array([b['shape_feat'] for b in batch])
    metas = [b['meta'] for b in batch]
    
    return crops, shapes, metas

def resize_with_padding(img, target_size=224, pad_value=128):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # 흑백(Mask)인지 컬러(Image)인지 확인하여 패딩 적용
    if img.ndim == 3:
        pad_value = (pad_value, pad_value, pad_value)
        
    new_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
    return new_img

# =========================
# Main (속도 최적화 + 시각화 기능 복원)
# =========================
def main():
    print(f"▶ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 데이터 경로: {BASE_DIR}")
    print(f"📂 결과 저장: {OUTPUT_DIR}")

    rows = []
    print("🔍 데이터 탐색 중...")
    for split, img_path, lbl_path in iter_label_image_pairs(BASE_DIR):
        for coords in read_yolo_polygon_lines(lbl_path):
            rows.append((split, img_path, lbl_path, coords))
    print(f"✅ Found {len(rows)} polygons")
    
    if len(rows) == 0:
        print("❌ 데이터를 찾지 못했습니다.")
        return

    # 1. Dataset & DataLoader 생성
    dataset = PolygonDataset(rows, CROP_PADDING)
    
    # CPU 코어 수 확인
    # n_workers = max(1, multiprocessing.cpu_count() - 2)
    n_workers = 10
    print(f"🚀 병렬 처리 시작 (Workers: {n_workers})")
    
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=n_workers, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    embedder = Embedder(BACKBONE if USE_CNN else "none", IMG_SIZE, AMP)
    cnn_list, shape_list, valid_rows = [], [], []

    # ✅ [수정] 30도 회전 설정 (12방향) - 괄호 오타 수정함
    ROTS = list(range(0, 360, 15)) 

    # 2. 특징 추출 루프
    for batch in tqdm(loader, desc="Extracting Features"):
        if batch is None: continue
        crops, shapes, metas = batch
        
        # (1) Shape Feature 저장
        shape_list.append(shapes)
        
        # (2) CNN 일괄 처리 (Batch Inference) - 속도 핵심!
        if USE_CNN:
            batch_rotated_imgs = []
            for crop in crops:
                for deg in ROTS:
                    batch_rotated_imgs.append(rotate_affine(crop, deg))
            
            # 한 번에 GPU 전송 (Batch * 8)
            all_feats = embedder(batch_rotated_imgs)
            
            # 평균 계산 (N, 8, 384) -> (N, 384)
            N = len(crops)
            n_rots = len(ROTS)
            all_feats = all_feats.reshape(N, n_rots, -1).mean(axis=1)
            
            cnn_list.append(all_feats)
        
        # (3) 메타데이터 복원
        import ast
        for (sp, ip, lp, co_str) in metas:
            try: coords = ast.literal_eval(co_str)
            except: coords = [float(x.strip().replace(',','')) for x in co_str.replace('[','').replace(']','').split()]
            valid_rows.append((sp, Path(ip), Path(lp), coords))

    # --- [수정된 코드] ---
    # 3. Feature 병합

    # (1) 먼저 각각 독립적으로 스케일링 (분산을 1로 맞춤)
    # Shape 피처 스케일링
    shape_feats = np.vstack(shape_list) if shape_list else np.empty((0,214), dtype=np.float32)
    if len(shape_feats) > 0:
        shape_feats = StandardScaler().fit_transform(shape_feats)

    if USE_CNN and cnn_list:
        cnn_feats = np.vstack(cnn_list)
        cnn_feats = StandardScaler().fit_transform(cnn_feats)
        
        # ==========================================================
        # ★ [수정] 차원 수(쪽수) 보정 로직 적용 위치 ★
        # ==========================================================
        # 설명: CNN은 384개, Shape는 214개이므로 단순히 합치면 CNN 영향력이 더 큽니다.
        # 따라서 np.sqrt(차원수)로 나누어 '1개 차원의 영향력'을 공평하게 맞춘 뒤 가중치를 곱합니다.
        
        feat_dim_cnn = cnn_feats.shape[1]     # 384
        feat_dim_shape = shape_feats.shape[1] # 214
        
        # 1) 쪽수 효과 제거 (Normalize by sqrt of dimensions)
        cnn_norm = cnn_feats / np.sqrt(feat_dim_cnn)
        shape_norm = shape_feats / np.sqrt(feat_dim_shape)
        
        # 2) 사용자가 설정한 GAIN 적용
        feats = np.concatenate([
            cnn_norm * CNN_GAIN, 
            shape_norm * SHAPE_GAIN
        ], axis=1)
        # ==========================================================
        
    else:
        feats = shape_feats * SHAPE_GAIN

    # 4. 차원 축소 (UMAP)
    # ★ 중요: 여기서 다시 StandardScaler를 쓰면 안 됩니다! (쓰면 가중치 또 사라짐) ★
    # scaled = StandardScaler().fit_transform(feats)  <-- 삭제
    scaled = feats # 그냥 feats를 사용

    print(f"✅ Final Weighted Feature shape: {scaled.shape}")
    
    # 2D (군집화용)
    print("🤖 Running UMAP (2D)...")
    reducer_2d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, 
        min_dist=UMAP_MIN_DIST, 
        n_components=2, 
        metric=UMAP_METRIC, 
        random_state=42
    )
    embedding_2d = reducer_2d.fit_transform(scaled)
    np.save(OUTPUT_DIR / "embedding_2d.npy", embedding_2d)

    # 3D (시각화용) - ✅ 복원됨
    print("🤖 Running UMAP (3D)...")
    reducer_3d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, 
        min_dist=UMAP_MIN_DIST, 
        n_components=3, 
        metric=UMAP_METRIC, 
        random_state=42
    )
    embedding_3d = reducer_3d.fit_transform(scaled)
    np.save(OUTPUT_DIR / "embedding_3d.npy", embedding_3d)
    
    # 5. 군집화 (HDBSCAN)
    print("🤖 Running HDBSCAN...")
    
    # 👈 [핵심] 여기서 사용할 값을 변수로 지정
    run_min_cluster_size = 50 

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=run_min_cluster_size, # 👈 변수 사용
        min_samples=10, 
        cluster_selection_epsilon=CLUSTER_EPS, 
        metric="euclidean", 
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embedding_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"🤖 Detected {n_clusters} clusters")

    # 6. 시각화 저장
    print("📊 Visualizing Clusters...")
    
    # 2D 시각화
    visualize_clusters(
        embedding_2d, labels, OUTPUT_DIR, n_clusters,
        UMAP_N_NEIGHBORS, UMAP_MIN_DIST, CLUSTER_EPS, SHAPE_GAIN, CNN_GAIN,
        run_min_cluster_size # 👈 [핵심] 아까 지정한 변수를 넘겨줌
    )
    
    # 3D 시각화 (정적 이미지)
    visualize_clusters_3d(embedding_3d, labels, OUTPUT_DIR)
    
    # 3D 인터랙티브 HTML (샘플링)
    VIS = 2000
    if len(embedding_3d) > VIS:
        idx = np.random.choice(len(embedding_3d), VIS, replace=False)
        embedding_3d_vis = embedding_3d[idx]
        labels_vis = labels[idx]
    else:
        embedding_3d_vis = embedding_3d
        labels_vis = labels
        
    visualize_clusters_3d_interactive(embedding_3d_vis, labels_vis, OUTPUT_DIR)

    # 7. 결과 파일 저장
    print("💾 결과 이미지/라벨 저장 중...")
    
    # 폴더 미리 생성
    for split_name in ["train", "valid"]:
        for c in range(n_clusters):
            (OUTPUT_DIR / split_name / f"cluster_{c}" / "images").mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / split_name / f"cluster_{c}" / "labels").mkdir(parents=True, exist_ok=True)
        if SAVE_NOISE:
            (OUTPUT_DIR / split_name / "noise" / "images").mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / split_name / "noise" / "labels").mkdir(parents=True, exist_ok=True)

    copied_images = set()
    for (split, img_path, lbl_path, coords), cid in zip(valid_rows, labels):
        tgt = "noise" if cid == -1 else f"cluster_{cid}"
        
        dst_img = OUTPUT_DIR / split / tgt / "images" / img_path.name
        dst_lbl = OUTPUT_DIR / split / tgt / "labels" / (img_path.stem + ".txt")
        
        # 1) 이미지 복사
        img_key = (split, tgt, img_path.name)
        if img_key not in copied_images:
            shutil.copy2(img_path, dst_img)
            copied_images.add(img_key)
        
        # 2) 라벨 저장
        with open(dst_lbl, "a", encoding="utf-8") as f:
            f.write(f"0 {' '.join(map(str, coords))}\n")
    
    np.save(OUTPUT_DIR / "file_paths.npy", np.array([str(p) for _, p, _, _ in valid_rows], dtype=object))
    print(f"✅ 완료! 결과 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()