#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cnn_polygon_cluster_per_class_fixed.py
- 수정사항: 
  1. min_cluster_size = 150 (고정)
  2. Device = cuda:1
  3. Num_workers = 10
"""

import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as patheffects
import os, cv2, math, shutil, warnings, gc
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
from torch.utils.data import Dataset, DataLoader
import multiprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 설정
# =========================
# 입력 데이터 최상위 경로
BASE_DIR = Path("/home/hgyeo/Desktop/NIA")

# 출력 최상위 경로
OUTPUT_ROOT = Path("/home/hgyeo/Desktop/All_cluster")

IMG_SIZE = 224
CROP_PADDING = 16
BATCH_SIZE = 64

BACKBONE    = "dino_v2"
USE_CNN    = True
AMP        = True
SAVE_NOISE = True

SHAPE_GAIN = 0.35
CNN_GAIN   = 0.65

CLUSTER_EPS       = 0.1
UMAP_N_COMPONENTS = 2
UMAP_METRIC       = 'cosine'
torch.backends.cudnn.benchmark = True

UMAP_N_NEIGHBORS = 150
UMAP_MIN_DIST    = 0.1

# [수정] 최소 군집 크기 150으로 고정
MIN_CLUSTER_SIZE_PER_CLASS = 100 

# [수정] 워커 수 10
NUM_WORKERS = 10

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
    aligned = cv2.resize(aligned, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    aligned_m = cv2.resize(aligned_m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return aligned, aligned_m

# =========================
# 데이터 순회
# =========================
def iter_label_image_pairs(base_dir: Path, target_class: str, splits=("train", "valid")):
    for split in splits:
        class_dir = base_dir / split / target_class
        if not class_dir.exists(): continue
        
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
# 시각화
# =========================
def visualize_clusters(embedding, labels, output_dir, n_clusters,
                       n_neighbors, min_dist, eps, shape_gain, cnn_gain,
                       min_cluster_size, class_name):

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    
    if len(unique_labels) > 20:
        base_colors = sns.color_palette("tab20", 20)
        colors = base_colors * (len(unique_labels) // 20 + 1)
        colors = colors[:len(unique_labels)]
    else:
        colors = sns.color_palette("tab10", len(unique_labels))

    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=12, c=[color], alpha=0.6, edgecolors='none'
        )

    for cluster_id in unique_labels:
        if cluster_id == -1: continue 
        mask = labels == cluster_id
        cx = embedding[mask, 0].mean()
        cy = embedding[mask, 1].mean()
        plt.text(
            cx, cy, f"{cluster_id}",
            fontsize=11, fontweight='bold', 
            ha='center', va='center', color='black',
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
        )

    plt.title(
        f"[{class_name}] Clusters ({n_clusters} detected)\n"
        f"N={n_neighbors}, D={min_dist}, EPS={eps}, MinSz={min_cluster_size}\n"
        f"S={shape_gain}, C={cnn_gain}",
        fontsize=14
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_dir / "umap_cluster_visualization.png", dpi=300)
    plt.close()

def visualize_clusters_3d(embedding, labels, output_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                   s=8, c=color.reshape(1, -1), alpha=0.8)
    ax.set_title("UMAP 3D Clusters")
    plt.tight_layout()
    plt.savefig(output_dir / "umap_3d_clusters.png", dpi=300)
    plt.close()

def visualize_clusters_3d_interactive(embedding, labels, output_dir):
    fig = px.scatter_3d(
        x=embedding[:,0], y=embedding[:,1], z=embedding[:,2],
        color=labels.astype(str), opacity=0.75, size=[1]*len(labels)
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(title="Interactive Clusters", showlegend=False)
    fig.write_html(str(output_dir / "umap_3d_interactive.html"))

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
# 형태 피처
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
    if area <= 0: return np.zeros(6, dtype=np.float32)
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
    EMPTY_DIM = 214
    binary = (crop_mask > 0).astype(np.uint8) * 255
    cnt = _largest_contour(binary)
    if cnt is None or len(cnt) < 5: return np.zeros(EMPTY_DIM, dtype=np.float32)

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    obj = cv2.bitwise_and(gray, gray, mask=binary)
    edges = cv2.Canny(obj, 80, 160)
    hu = cv2.HuMoments(cv2.moments(edges)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    
    proj   = _projection_lengths_sorted(cnt, k=32)
    band   = _band_width_profile_invariant(binary, bins=96)
    radial = _radial_histogram(cnt, bins=32)
    scalars = _basic_shape_scalars(cnt, binary)
    
    h = binary.shape[0]
    top_w = (binary[:h//3] > 0).sum(axis=1).mean() if (binary[:h//3]>0).any() else 0.0
    mid_w = (binary[h//3:2*h//3] > 0).sum(axis=1).mean() if (binary[h//3:2*h//3]>0).any() else 0.0
    neck_idx = float((top_w+1e-6)/(mid_w+1e-6))

    obj_blur = cv2.GaussianBlur(obj, (5,5), 0)
    edges2 = cv2.Canny(obj_blur, 50, 120)
    hu2 = cv2.HuMoments(cv2.moments(edges2)).flatten()
    hu2 = -np.sign(hu2) * np.log10(np.abs(hu2) + 1e-12)
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    dist_hist = np.histogram(dist, bins=32, range=(0, dist.max() if dist.max()>0 else 1))[0]
    
    try:
        (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        ellipse_ratio = np.array([ma / MA], dtype=np.float32)
    except:
        ellipse_ratio = np.array([0.0], dtype=np.float32)

    return np.concatenate([
        hu.astype(np.float32), hu2.astype(np.float32), 
        proj, band, radial, scalars, 
        dist_hist.astype(np.float32), ellipse_ratio, 
        np.array([neck_idx], dtype=np.float32)
    ]).astype(np.float32)

# =========================
# CNN 임베더 (DINOv2)
# =========================
class Embedder(nn.Module):
    def __init__(self, backbone="dino_v2", img_size=224, amp=True):
        super().__init__()
        self.amp = amp and torch.cuda.is_available()
        # [수정] cuda:1 로 설정
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        if backbone == "dino_v2":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device).eval()
            self.preprocess = T.Compose([
                T.ToPILImage(), T.Resize((img_size, img_size)),
                T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.out_dim = 384
        else:
            self.model = None
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

# =========================
# Dataset & Multi-processing
# =========================
class PolygonDataset(Dataset):
    def __init__(self, rows, crop_padding):
        self.rows = rows
        self.crop_padding = crop_padding
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        split, img_path, lbl_path, coords = self.rows[idx]
        img = cv2.imread(str(img_path))
        if img is None: return None 
        crop_bgr, crop_m = crop_polygon_mask(img, coords, self.crop_padding)
        if crop_bgr is None: return None

        rots = list(range(0, 360, 15)) 
        feats = []
        for deg in rots: 
            rot_img  = rotate_affine(crop_bgr, deg)
            rot_mask = rotate_affine_mask(crop_m, deg)
            f = compute_shape_features_from_crop(rot_img, rot_mask)
            if f.shape[0] == 214: feats.append(f)
        
        if len(feats) > 0: shape_feat = np.mean(feats, axis=0)
        else: shape_feat = np.zeros(214, dtype=np.float32)

        return {
            "crop_bgr": crop_bgr,
            "shape_feat": shape_feat,
            "meta": (split, str(img_path), str(lbl_path), str(coords)) 
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    crops = [b['crop_bgr'] for b in batch]
    shapes = np.array([b['shape_feat'] for b in batch])
    metas = [b['meta'] for b in batch]
    return crops, shapes, metas

# =========================
# 클래스별 처리 로직
# =========================
def process_single_class(class_name, base_dir, output_root_dir, n_workers):
    
    print(f"\n{'='*50}")
    print(f"🚀 Processing Class: {class_name}")
    print(f"{'='*50}")

    class_output_dir = output_root_dir / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 수집
    rows = []
    for split, img_path, lbl_path in iter_label_image_pairs(base_dir, class_name):
        for coords in read_yolo_polygon_lines(lbl_path):
            rows.append((split, img_path, lbl_path, coords))
    
    print(f"   -> Found {len(rows)} polygons for {class_name}")
    if len(rows) < 10:
        print(f"⚠️  Not enough data for class {class_name}, skipping...")
        return

    # 데이터 로더
    dataset = PolygonDataset(rows, CROP_PADDING)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=n_workers, collate_fn=collate_fn, pin_memory=True)
    
    embedder = Embedder(BACKBONE if USE_CNN else "none", IMG_SIZE, AMP)
    cnn_list, shape_list, valid_rows = [], [], []
    ROTS = list(range(0, 360, 15))

    # 특징 추출
    for batch in tqdm(loader, desc=f"   -> Extracting {class_name}", leave=False):
        if batch is None: continue
        crops, shapes, metas = batch
        shape_list.append(shapes)
        
        if USE_CNN:
            batch_rotated_imgs = []
            for crop in crops:
                for deg in ROTS:
                    batch_rotated_imgs.append(rotate_affine(crop, deg))
            all_feats = embedder(batch_rotated_imgs)
            N = len(crops)
            all_feats = all_feats.reshape(N, len(ROTS), -1).mean(axis=1)
            cnn_list.append(all_feats)
        
        import ast
        for (sp, ip, lp, co_str) in metas:
            try: coords = ast.literal_eval(co_str)
            except: coords = [float(x.strip().replace(',','')) for x in co_str.replace('[','').replace(']','').split()]
            valid_rows.append((sp, Path(ip), Path(lp), coords))

    del embedder
    torch.cuda.empty_cache()

    shape_feats = np.vstack(shape_list) if shape_list else np.empty((0,214), dtype=np.float32)
    feats = shape_feats * SHAPE_GAIN
    if USE_CNN and cnn_list:
        cnn_feats = np.vstack(cnn_list)
        feats = np.concatenate([cnn_feats * CNN_GAIN, shape_feats * SHAPE_GAIN], axis=1)

    # UMAP & Clustering
    print(f"   -> Running UMAP & HDBSCAN...")
    scaled = StandardScaler().fit_transform(feats)
    
    reducer_2d = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, metric=UMAP_METRIC, random_state=42)
    embedding_2d = reducer_2d.fit_transform(scaled)
    np.save(class_output_dir / "embedding_2d.npy", embedding_2d)

    reducer_3d = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=3, metric=UMAP_METRIC, random_state=42)
    embedding_3d = reducer_3d.fit_transform(scaled)
    np.save(class_output_dir / "embedding_3d.npy", embedding_3d)

    # [수정] 150으로 고정된 값 사용
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE_PER_CLASS, min_samples=None, 
                                cluster_selection_epsilon=CLUSTER_EPS, metric="euclidean", core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(embedding_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   -> {class_name}: Detected {n_clusters} clusters (Min Size: {MIN_CLUSTER_SIZE_PER_CLASS})")

    # 시각화
    visualize_clusters(embedding_2d, labels, class_output_dir, n_clusters,
                       UMAP_N_NEIGHBORS, UMAP_MIN_DIST, CLUSTER_EPS, SHAPE_GAIN, CNN_GAIN, MIN_CLUSTER_SIZE_PER_CLASS, class_name)
    visualize_clusters_3d(embedding_3d, labels, class_output_dir)
    
    VIS = 2000
    if len(embedding_3d) > VIS:
        idx = np.random.choice(len(embedding_3d), VIS, replace=False)
        visualize_clusters_3d_interactive(embedding_3d[idx], labels[idx], class_output_dir)
    else:
        visualize_clusters_3d_interactive(embedding_3d, labels, class_output_dir)

    # 저장
    print(f"   -> Saving images to {class_output_dir}...")
    
    for split_name in ["train", "valid"]:
        for c in range(n_clusters):
            (class_output_dir / split_name / f"cluster_{c}" / "images").mkdir(parents=True, exist_ok=True)
            (class_output_dir / split_name / f"cluster_{c}" / "labels").mkdir(parents=True, exist_ok=True)
        if SAVE_NOISE:
            (class_output_dir / split_name / "noise" / "images").mkdir(parents=True, exist_ok=True)
            (class_output_dir / split_name / "noise" / "labels").mkdir(parents=True, exist_ok=True)

    copied_images = set()
    for (split, img_path, lbl_path, coords), cid in zip(valid_rows, labels):
        tgt = "noise" if cid == -1 else f"cluster_{cid}"
        
        dst_img = class_output_dir / split / tgt / "images" / img_path.name
        dst_lbl = class_output_dir / split / tgt / "labels" / (img_path.stem + ".txt")
        
        img_key = (split, tgt, img_path.name)
        if img_key not in copied_images:
            shutil.copy2(img_path, dst_img)
            copied_images.add(img_key)
        
        with open(dst_lbl, "a", encoding="utf-8") as f:
            f.write(f"0 {' '.join(map(str, coords))}\n")

    print(f"✅ {class_name} Done.")


# =========================
# Main
# =========================
def main():
    print(f"▶ Device: cuda:1 (Target) / Actual: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    train_dir = BASE_DIR / "train"
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"📂 Found {len(classes)} classes: {classes}")
    
    # [수정] 워커 수 전달
    for cls in classes:
        try:
            process_single_class(cls, BASE_DIR, OUTPUT_ROOT, NUM_WORKERS)
        except Exception as e:
            print(f"❌ Error processing class {cls}: {e}")
            import traceback
            traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉 All classes processed successfully!")

if __name__ == "__main__":
    main()