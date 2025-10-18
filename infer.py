from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import logging
import time
import pickle

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import cam_crop_to_full
import detectron2.data.transforms as T

from vitpose_model import ViTPoseModel

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    logging.Formatter.converter = time.gmtime
    logging.info("Starting infer.py (Inference Job)")
    parser = argparse.ArgumentParser(description='HaMeR inference code')
    parser.add_argument('--data_dir', type=str, default='_DATA', help='Path to _DATA folder')
    parser.add_argument('--checkpoint', type=str, default='hamer_ckpts/checkpoints/hamer.ckpt', help='Path to pretrained model checkpoint, relative to data_dir')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--inference_output_folder', type=str, default='inference_outputs', help='Folder to save raw inference outputs')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--testing', action='store_true', help='If set, only process the first batch_size images')

    args = parser.parse_args()

    # Download and load checkpoints
    data_dir = Path(args.data_dir)
    checkpoint_path = data_dir / args.checkpoint
    model, model_cfg = load_hamer(checkpoint_path, data_dir)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device, args.data_dir)

    # Make inference output directory if it does not exist
    os.makedirs(args.inference_output_folder, exist_ok=True)

    # Get all demo images
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    if args.testing:
        img_paths = img_paths[:args.batch_size]

    # Iterate over all images
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[...,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[...,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            logging.info(f"No hands detected in {img_path}, skipping.")
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        image_output_dir = os.path.join(args.inference_output_folder, img_fn)
        os.makedirs(image_output_dir, exist_ok=True)

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size_current = batch['img'].shape[0]
            for n in range(batch_size_current):
                person_id = int(batch['personid'][n])
                
                inference_data = {
                    'pred_vertices': out['pred_vertices'][n].detach().cpu().numpy(),
                    'pred_cam_t': out['pred_cam_t'][n].detach().cpu().numpy(),
                    'pred_cam_t_full': pred_cam_t_full[n],
                    'is_right': batch['right'][n].cpu().numpy(),
                    'img_size': img_size[n].cpu().numpy(),
                    'original_img_path': str(img_path),
                    'input_patch': batch['img'][n].cpu(),
                    'model_cfg': model_cfg,
                    'faces': model.mano.faces,
                    'focal_length': scaled_focal_length.cpu().numpy(),
                }

                output_filepath = os.path.join(image_output_dir, f'person_{person_id}.pkl')
                with open(output_filepath, 'wb') as f:
                    pickle.dump(inference_data, f)
                logging.info(f"Saved inference output for {img_fn}, person {person_id} to {output_filepath}")

    logging.info("Finished infer.py (Inference Job)")

if __name__ == '__main__':
    main()