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
import redis

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer as hamer_pkg
from vitpose_model import ViTPoseModel

# --- Global Variables ---
detector = None
cpm = None
device = None
redis_client = None

def initialize_models(args):
    """
    Initializes and loads the lightweight models needed for hand detection.
    """
    global detector, cpm, device, redis_client

    logging.info("Initializing lightweight detection models...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load body detector (ViTDet)
    cfg_path = Path(hamer_pkg.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detectron2_cfg.model.roi_heads.box_predictors[i].test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Load keypoint detector (ViTPose)
    cpm = ViTPoseModel(device, args.data_dir)
    
    # Initialize Redis client
    redis_client = redis.Redis(host='redis-service', port=6379, db=0)
    
    logging.info("Detection models and Redis client initialized successfully.")

def non_max_suppression(boxes, scores, iou_threshold):
    """
    Apply non-maximum suppression to filter overlapping bounding boxes.
    """
    if not boxes:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [boxes[i].tolist() for i in keep]

def update_time_series_data(stream_name, hand_count, frame_stem):
    """
    Adds a new data point to a Redis Sorted Set for the given stream.
    """
    if redis_client is None:
        logging.error("Redis client not initialized.")
        return

    key = f"stream:{stream_name}:frames"
    timestamp = int(time.time() * 1000)
    
    member_data = {
        "hand_count": hand_count,
        "frame_file": frame_stem
    }
    
    try:
        redis_client.zadd(key, {json.dumps(member_data): timestamp})
        logging.info(f"Successfully added hand_count {hand_count} for {frame_stem} to {key}")
    except Exception as e:
        logging.error(f"An error occurred while updating time series data: {e}")


def process_image(image_path: str, args):
    """
    Processes a single image to detect hands, count them, and draw bounding boxes.
    """
    try:
        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        img_fn_stem = Path(image_path).stem
        stream_name = Path(image_path).parts[-3]

        # 1. Detect humans in the image
        det_out = detector(img_cv2)
        img_for_pose = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        hand_count = 0
        hand_boxes = []
        if len(pred_bboxes) > 0:
            # 2. Detect human keypoints for each person
            vitposes_out = cpm.predict_pose(
                img_for_pose,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            # 3. Extract hand bounding boxes from keypoints
            
            hand_scores = []
            for vitposes in vitposes_out:
                # Process left and right hands
                for hand_keypoints in [vitposes['keypoints'][-42:-21], vitposes['keypoints'][-21:]]:
                    valid = hand_keypoints[..., 2] > 0.5 # Check confidence
                    if sum(valid) > 3: # Threshold for a valid hand detection
                        x_min, y_min = hand_keypoints[valid, :2].min(axis=0)
                        x_max, y_max = hand_keypoints[valid, :2].max(axis=0)
                        hand_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                        hand_scores.append(hand_keypoints[valid, 2].mean())
            
            # 4. Apply Non-Maximum Suppression
            if hand_boxes:
                hand_boxes = non_max_suppression(hand_boxes, hand_scores, iou_threshold=0.5)
            
            hand_count = len(hand_boxes)

        # 5. Update time series data
        update_time_series_data(stream_name, hand_count, img_fn_stem)

        # 6. Prepare output paths
        annotations_dir = Path("/mnt/nfs/streams") / stream_name / "hamer" / "annotations"
        results_dir = Path("/mnt/nfs/streams") / stream_name / "hamer" / "results"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        # 7. Draw boxes on the image and save it
        if hand_count > 0:
            annotated_image = img_cv2.copy()
            for (x_min, y_min, x_max, y_max) in hand_boxes:
                cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            annotated_image_path = annotations_dir / f"{img_fn_stem}_annotated.jpg"
            cv2.imwrite(str(annotated_image_path), annotated_image)
            logging.info(f"Saved annotated image to {annotated_image_path}")

        # 8. Save JSON data
        output_data = {
            "image_path": image_path,
            "hand_count": hand_count,
            "bounding_boxes": hand_boxes if hand_count > 0 else []
        }
        json_output_path = results_dir / f"{img_fn_stem}_data.json"
        with open(json_output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved hand count data to {json_output_path}")

    except Exception as e:
        logging.error(f"An error occurred processing {image_path}: {e}", exc_info=True)

def process_frame_callback(message: pubsub_v1.subscriber.message.Message, args) -> None:
    """
    Callback function to process a single frame message from Pub/Sub.
    """
    logging.info("Entered process_frame_callback.")
    try:
        data = json.loads(message.data)
        image_path = data["frame_path"]
        logging.info(f"Received message for image: {image_path}")

        stream_name = Path(image_path).parts[-3]
        last_processed_key = f"stream:{stream_name}:last_processed"
        current_time = time.time()

        time_between_messages = 1.0 / args.fps
        last_processed_time_str = redis_client.get(last_processed_key)
        if last_processed_time_str:
            last_processed_time = float(last_processed_time_str)
            if current_time - last_processed_time < time_between_messages:
                logging.info(f"Rate limit: skipping message for stream {stream_name}.")
                message.ack()
                return
        
        process_image(image_path, args)
        
        redis_client.set(last_processed_key, current_time)
        logging.info(f"Updated last processed time for stream {stream_name}.")

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
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second to process.')
    
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
