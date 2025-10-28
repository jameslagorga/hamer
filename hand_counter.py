import torch
import argparse
import os
import cv2
import numpy as np
import logging
import time
import json
from pathlib import Path
from google.cloud import pubsub_v1

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer as hamer_pkg
from vitpose_model import ViTPoseModel

# --- Global Variables ---
detector = None
cpm = None
device = None

def initialize_models(args):
    """
    Initializes and loads the lightweight models needed for hand detection.
    """
    global detector, cpm, device

    logging.info("Initializing lightweight detection models...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load body detector (ViTDet)
    cfg_path = Path(hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Load keypoint detector (ViTPose)
    cpm = ViTPoseModel(device, args.data_dir)
    logging.info("Detection models initialized successfully.")

def process_image(image_path: str, args):
    """
    Processes a single image to detect hands, count them, and draw bounding boxes.
    """
    try:
        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        # 1. Detect humans in the image
        det_out = detector(img_cv2)
        img_for_pose = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            logging.info(f"No people detected in {image_path}, skipping.")
            return

        # 2. Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_for_pose,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        # 3. Extract hand bounding boxes from keypoints
        hand_boxes = []
        for vitposes in vitposes_out:
            # Process left and right hands
            for hand_keypoints in [vitposes['keypoints'][-42:-21], vitposes['keypoints'][-21:]]:
                valid = hand_keypoints[..., 2] > 0.5 # Check confidence
                if sum(valid) > 3: # Threshold for a valid hand detection
                    x_min, y_min = hand_keypoints[valid, :2].min(axis=0)
                    x_max, y_max = hand_keypoints[valid, :2].max(axis=0)
                    hand_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])

        if not hand_boxes:
            logging.info(f"No hands detected in {image_path}, skipping.")
            return

        # 4. Prepare output
        img_fn_stem = Path(image_path).stem
        stream_name = Path(image_path).parts[-3]
        output_dir = Path(args.output_folder) / stream_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 5. Draw boxes on the image and save it
        annotated_image = img_cv2.copy()
        for (x_min, y_min, x_max, y_max) in hand_boxes:
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        annotated_image_path = output_dir / f"{img_fn_stem}_annotated.jpg"
        cv2.imwrite(str(annotated_image_path), annotated_image)
        logging.info(f"Saved annotated image to {annotated_image_path}")

        # 6. Save JSON data
        output_data = {
            "image_path": image_path,
            "hand_count": len(hand_boxes),
            "bounding_boxes": hand_boxes
        }
        json_output_path = output_dir / f"{img_fn_stem}_data.json"
        with open(json_output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved hand count data to {json_output_path}")

    except Exception as e:
        logging.error(f"An error occurred processing {image_path}: {e}", exc_info=True)

def process_frame_callback(message: pubsub_v1.subscriber.message.Message, args) -> None:
    """
    Callback function to process a single frame message from Pub/Sub.
    """
    try:
        data = json.loads(message.data)
        image_path = data["frame_path"]
        logging.info(f"Received message for image: {image_path}")
        
        process_image(image_path, args)
        
        message.ack()
    except Exception as e:
        logging.error(f"An error occurred processing message: {e}")
        message.nack()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    logging.Formatter.converter = time.gmtime
    
    parser = argparse.ArgumentParser(description='Lightweight Pub/Sub subscriber for hand counting and boxing.')
    parser.add_argument('--project_id', type=str, required=True, help='Your Google Cloud project ID.')
    parser.add_argument('--subscription_id', type=str, required=True, help='The Pub/Sub subscription ID.')
    parser.add_argument('--data_dir', type=str, default='/mnt/nfs/_DATA', help='Path to _DATA folder for model checkpoints.')
    parser.add_argument('--output_folder', type=str, default='/mnt/nfs/hand_counts', help='Folder to save annotated images and JSON data.')
    
    args = parser.parse_args()

    initialize_models(args)

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(args.project_id, args.subscription_id)

    callback = lambda message: process_frame_callback(message, args)

    logging.info(f"Starting hand counter, listening for messages on {subscription_path}...")
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    
    try:
        streaming_pull_future.result()
    except Exception as e:
        streaming_pull_future.cancel()
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        subscriber.close()
        logging.info("Subscriber shut down.")

if __name__ == '__main__':
    main()
