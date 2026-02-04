import os

# ================= 사용자 설정 =================
GPUS = "0"       # GPU 1번만 사용
NUM_GPUS = 1     

# [수정] Config 파일 경로 (configs 폴더 안을 가리켜야 함)
CONFIG_FILE = "/home/hgyeo/Desktop/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/RT-DETRv2-X.yml" 

# 가중치 파일 (같은 폴더에 있다면 파일명만)
WEIGHT_FILE = "/home/hgyeo/Desktop/RT-DETR/rtdetrv2_pytorch/RT-DETRv2-X.pth" 
# ============================================

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
    print(f"🚀 Using GPU: {GPUS}")

    cmd = f"torchrun --nproc_per_node={NUM_GPUS} tools/train.py -c {CONFIG_FILE}"
    
    if WEIGHT_FILE and os.path.exists(WEIGHT_FILE):
        print(f"🔄 Resuming/Tuning from: {WEIGHT_FILE}")
        cmd += f" -t {WEIGHT_FILE}"
    else:
        print("⚠️ Warning: No checkpoint found. Training from scratch!")

    print(f"▶️ Executing: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()