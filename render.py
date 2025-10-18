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
from collections import defaultdict

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer

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
    all_pkl_files = glob.glob(os.path.join(args.inference_input_folder, '**', '*.pkl'), recursive=True)
    
    if not all_pkl_files:
        logging.warning(f"No .pkl files found in {args.inference_input_folder}. Exiting.")
        return

    # Group pkl files by original_img_path
    grouped_pkl_files = defaultdict(list)
    for pkl_filepath in all_pkl_files:
        with open(pkl_filepath, 'rb') as f:
            inference_data = pickle.load(f)
        original_img_path = inference_data['original_img_path']
        grouped_pkl_files[original_img_path].append(pkl_filepath)

    for original_img_path, pkl_filepaths_for_image in grouped_pkl_files.items():
        logging.info(f"Processing image: {original_img_path}")
        # Load data from the first pkl file to initialize renderer (assuming model_cfg, faces, etc. are consistent)
        with open(pkl_filepaths_for_image[0], 'rb') as f:
            first_inference_data = pickle.load(f)
        
        model_cfg = first_inference_data['model_cfg']
        faces = first_inference_data['faces']
        renderer = Renderer(model_cfg, faces=faces)

        img_fn, _ = os.path.splitext(os.path.basename(original_img_path))
        image_output_dir = os.path.join(args.output_folder, img_fn)
        os.makedirs(image_output_dir, exist_ok=True)

        all_verts = []
        all_cam_t = []
        all_right = []
        img_size = None
        scaled_focal_length = None

        for pkl_filepath in pkl_filepaths_for_image:
            start_time_load = time.time()
            with open(pkl_filepath, 'rb') as f:
                inference_data = pickle.load(f)
            logging.info(f"Time to load {os.path.basename(pkl_filepath)}: {time.time() - start_time_load:.4f}s")

            person_id = os.path.splitext(os.path.basename(pkl_filepath))[0].replace('person_', '')
            pred_vertices = inference_data['pred_vertices']
            pred_cam_t = inference_data['pred_cam_t']
            pred_cam_t_full = inference_data['pred_cam_t_full']
            is_right = inference_data['is_right']
            input_patch_tensor = inference_data['input_patch']
            img_size = inference_data['img_size']
            scaled_focal_length = inference_data['focal_length']

            start_time_render = time.time()
            # Render the result
            white_img = (torch.ones_like(input_patch_tensor) - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = input_patch_tensor * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            regression_img = renderer(pred_vertices,
                                    pred_cam_t, # Use patch-relative camera translation
                                    input_patch_tensor, # Use patch as background
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )

            if args.side_view:
                side_img = renderer(pred_vertices,
                                        pred_cam_t, # Use patch-relative camera translation
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            else:
                final_img = np.concatenate([input_patch, regression_img], axis=1)

            cv2.imwrite(os.path.join(image_output_dir, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])
            logging.info(f"Time to render and save individual image for {img_fn}, person {person_id}: {time.time() - start_time_render:.4f}s")

            # Save all meshes to disk
            if args.save_mesh:
                start_time_mesh = time.time()
                verts_for_mesh = pred_vertices.copy()
                verts_for_mesh[:,0] = (2*is_right-1)*verts_for_mesh[:,0]
                tmesh = renderer.vertices_to_trimesh(verts_for_mesh, pred_cam_t_full, LIGHT_BLUE, is_right=is_right)
                tmesh.export(os.path.join(image_output_dir, f'{img_fn}_{person_id}.obj'))
                logging.info(f"Time to save mesh for {img_fn}, person {person_id}: {time.time() - start_time_mesh:.4f}s")

            # Collect data for full frame rendering
            verts = pred_vertices.copy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            all_verts.append(verts)
            all_cam_t.append(pred_cam_t_full)
            all_right.append(is_right)

        # Render full frame
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size, is_right=all_right, **misc_args)

            # Overlay image
            input_img = cv2.imread(original_img_path).astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(image_output_dir, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    logging.info("Finished render.py (Rendering Job)")

if __name__ == '__main__':
    main()
