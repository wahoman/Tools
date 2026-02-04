import numpy as np
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as patheffects
from pathlib import Path

# =========================================================
# 🎛️ 여기서 EPS 값만 바꾸세요! (1초 컷)
# =========================================================
TARGET_EPS = 0.05        # 👈 요청하신 0.1 (변경 가능: 0.0 ~ 0.5)
MIN_CLUSTER_SIZE = 150    # 기존 설정 유지 (필요시 변경)

# 파일 경로 (방금 말씀하신 경로)
NPY_PATH = Path("/home/hgyeo/Desktop/BCAS/cluster_data_output/embedding_2d.npy")
# ※ 만약 경로 에러나면 data_output1 인지 확인해보세요.
# OUTPUT_DIR = NPY_PATH.parent / "eps_test_result" # 결과 저장 폴더
OUTPUT_DIR = Path("/home/hgyeo/Desktop/Origin_cluster_base_folder")

# =========================================================
# 시각화 함수 (기존 스타일 유지)
# =========================================================
def visualize_quick(embedding, labels, save_path, eps, min_size):
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 색상 팔레트
    if len(unique_labels) > 20:
        base_colors = sns.color_palette("tab20", 20)
        colors = base_colors * (len(unique_labels) // 20 + 1)
        colors = colors[:len(unique_labels)]
    else:
        colors = sns.color_palette("tab10", len(unique_labels))

    # 1. 산점도 그리기
    for cluster_id, color in zip(unique_labels, colors):
        mask = labels == cluster_id
        # 노이즈(-1)는 연한 회색, 나머지는 컬러
        c = [0.85, 0.85, 0.85] if cluster_id == -1 else color
        alpha = 0.2 if cluster_id == -1 else 0.8
        s = 3 if cluster_id == -1 else 10
        
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=s, c=[c], alpha=alpha, edgecolors='none'
        )

    # 2. 라벨 텍스트 (하얀 테두리 포함)
    for cluster_id in unique_labels:
        if cluster_id == -1: continue
        mask = labels == cluster_id
        cx, cy = embedding[mask, 0].mean(), embedding[mask, 1].mean()
        
        plt.text(
            cx, cy, f"{cluster_id}",
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='black',
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
        )

    plt.title(
        f"Quick EPS Check\n"
        f"Clusters: {n_clusters} (Noise: {(labels==-1).sum()} pts)\n"
        f"EPS: {eps}, Min_Size: {min_size}",
        fontsize=14
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"🖼️ 그래프 저장됨: {save_path}")

# =========================================================
# 메인 실행
# =========================================================
def main():
    if not NPY_PATH.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {NPY_PATH}")
        print("경로가 'data_output' 인지 'data_output1' 인지 확인해주세요.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. 로드 (순식간)
    print(f"📂 Loading embedding: {NPY_PATH.name}...")
    embedding_2d = np.load(NPY_PATH)
    
    # 2. HDBSCAN 실행 (1~2초)
    print(f"🤖 Clustering with EPS={TARGET_EPS}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=10, 
        cluster_selection_epsilon=TARGET_EPS, 
        metric="euclidean", 
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(embedding_2d)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ 결과: 군집 {n_clusters}개 발견")

    # 3. 시각화 저장
    save_name = OUTPUT_DIR / f"result_eps_{TARGET_EPS}.png"
    visualize_quick(embedding_2d, labels, save_name, TARGET_EPS, MIN_CLUSTER_SIZE)

if __name__ == "__main__":
    main()