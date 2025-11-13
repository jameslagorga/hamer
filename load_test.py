
import torch
from mmpose.apis import init_pose_model
import logging
import time

logging.basicConfig(level=logging.INFO)

VIT_DIR = "third-party/ViTPose"
data_dir = "/mnt/nfs/_DATA"
device = "cuda" if torch.cuda.is_available() else "cpu"

config = f'{VIT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py'
model_path = f'{data_dir}/vitpose_ckpts/vitpose+_huge/wholebody.pth'

logging.info(f"Loading model from {model_path}...")
start_time = time.time()
model = init_pose_model(config, model_path, device=device)
end_time = time.time()
logging.info(f"Model loaded in {end_time - start_time} seconds.")
