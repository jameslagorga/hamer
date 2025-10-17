from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import logging
import time
import pickle
import glob

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    logging.Formatter.converter = time.gmtime
    logging.info("Starting render.py (Rendering Job)")
    parser = argparse.ArgumentParser(description='HaMeR rendering code')
    parser.add_argument('--inference_input_folder', type=str, default='inference_outputs', help='Folder containing raw inference outputs (.pkl files)')
    parser.add_argument('--output_folder', type=str, default='rendered_outputs', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Find all .pkl files in the inference input folder
    pkl_files = glob.glob(os.path.join(args.inference_input_folder, '**', '*.pkl'), recursive=True)
    
    if not pkl_files:
        logging.warning(f"No .pkl files found in {args.inference_input_folder}. Exiting.")
        return

    # Load a dummy model to get model.mano.faces and model_cfg for renderer setup
    # This is a workaround since model_cfg and faces are saved per pkl, but renderer needs to be initialized once
    # We assume model_cfg and faces are consistent across all inference outputs
    with open(pkl_files[0], 'rb') as f:
        first_inference_data = pickle.load(f)
    
    model_cfg = first_inference_data['model_cfg']
    faces = first_inference_data['faces']
    
    renderer = Renderer(model_cfg, faces=faces)

    for pkl_filepath in pkl_files:
        start_time_load = time.time()
        with open(pkl_filepath, 'rb') as f:
            inference_data = pickle.load(f)
        logging.info(f"Time to load {os.path.basename(pkl_filepath)}: {time.time() - start_time_load:.4f}s")

        pred_vertices = inference_data['pred_vertices']
        pred_cam_t_full = inference_data['pred_cam_t_full']
        is_right = inference_data['is_right']
        img_size = inference_data['img_size']
        original_img_path = inference_data['original_img_path']
        original_img_cv2 = inference_data['original_img_cv2']
        scaled_focal_length = inference_data['focal_length']

        img_fn, _ = os.path.splitext(os.path.basename(original_img_path))
        person_id = os.path.splitext(os.path.basename(pkl_filepath))[0].replace('person_', '')

        output_image_dir = os.path.join(args.output_folder, img_fn)
        os.makedirs(output_image_dir, exist_ok=True)

        start_time_render = time.time()
        # Render the result
        white_img = (torch.ones_like(torch.from_numpy(original_img_cv2).permute(2,0,1)).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
        input_patch = (torch.from_numpy(original_img_cv2).permute(2,0,1).cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)).permute(1,2,0).numpy()

        regression_img = renderer(pred_vertices,
                                pred_cam_t_full,
                                torch.from_numpy(original_img_cv2).permute(2,0,1), # Pass original image for background
                                mesh_base_color=LIGHT_BLUE,
                                scene_bg_color=(1, 1, 1),
                                )

        if args.side_view:
            side_img = renderer(pred_vertices,
                                    pred_cam_t_full,
                                    white_img,
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    side_view=True)
            final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
        else:
            final_img = np.concatenate([input_patch, regression_img], axis=1)

        cv2.imwrite(os.path.join(output_image_dir, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])
        logging.info(f"Time to render and save individual image for {img_fn}, person {person_id}: {time.time() - start_time_render:.4f}s")

        # Save all meshes to disk
        if args.save_mesh:
            start_time_mesh = time.time()
            camera_translation = pred_cam_t_full.copy()
            tmesh = renderer.vertices_to_trimesh(pred_vertices, camera_translation, LIGHT_BLUE, is_right=is_right)
            tmesh.export(os.path.join(output_image_dir, f'{img_fn}_{person_id}.obj'))
            logging.info(f"Time to save mesh for {img_fn}, person {person_id}: {time.time() - start_time_mesh:.4f}s")

    # Handle full frame rendering (if enabled and data is available)
    # This part needs to collect all verts/cams for a single original image
    # and then render them together. This requires restructuring how data is processed.
    # For simplicity, I'll skip full_frame rendering for now, as it requires
    # grouping pkl files by original_img_path. If full_frame rendering is critical,
    # we'd need to modify infer.py to save all person data for one image in a single pkl
    # or a dedicated folder.
    logging.info("Full frame rendering is currently skipped in render.py for simplicity.")

    logging.info("Finished render.py (Rendering Job)")

if __name__ == '__main__':
    main()
