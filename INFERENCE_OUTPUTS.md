### Inference Outputs Explained

The `infer.py` script processes each image and, for every detected hand, saves a Python pickle file (`.pkl`) containing a dictionary of raw inference results. These `.pkl` files are organized into subdirectories within your specified `--inference_output_folder`, with each subdirectory corresponding to an original input image.

Each `.pkl` file (e.g., `inference_outputs/test1/person_0.pkl`) contains a dictionary with the following keys and their explanations:

*   **`pred_vertices`** (`numpy.ndarray`, shape: `[num_vertices, 3]`)
    *   **Explanation**: These are the 3D coordinates (x, y, z) of the mesh vertices predicted by the HAMER model for a single hand. This array defines the precise 3D shape of the hand model in a canonical space.

*   **`pred_cam_t_full`** (`numpy.ndarray`, shape: `[3,]`)
    *   **Explanation**: This is a 3D translation vector (x, y, z) that positions the 3D hand model within the camera's coordinate system. It effectively tells the renderer where to place the hand in the 3D scene to match its detected position in the 2D image.

*   **`is_right`** (`numpy.ndarray`, scalar, e.g., `0` or `1`)
    *   **Explanation**: A flag indicating whether the predicted hand is a right hand (`1`) or a left hand (`0`). This is important for rendering, as left and right hands are often mirrored or have specific orientations.

*   **`img_size`** (`numpy.ndarray`, shape: `[2,]`)
    *   **Explanation**: The original width and height (e.g., `[width, height]`) of the input image in pixels. This is used by the renderer for correct scaling and positioning of the 3D mesh onto the 2D image plane.

*   **`original_img_path`** (`str`)
    *   **Explanation**: The absolute file path to the original image from which this specific hand inference was derived. Useful for tracing back results and for the renderer to load the correct background image.

*   **`original_img_cv2`** (`numpy.ndarray`, shape: `[height, width, 3]`)
    *   **Explanation**: The raw pixel data of the original input image, loaded by OpenCV (typically in BGR format). The rendering job uses this to draw the 3D mesh overlay directly on top of the actual 2D image.

*   **`model_cfg`** (`CfgNode` object from Detectron2)
    *   **Explanation**: The full configuration object that was used to initialize the HAMER model. This contains various parameters (e.g., `FOCAL_LENGTH`, `IMAGE_SIZE`) that are crucial for the `Renderer` to accurately project the 3D mesh onto the 2D image plane.

*   **`faces`** (`numpy.ndarray`, shape: `[num_faces, 3]`)
    *   **Explanation**: This array defines the triangular faces of the 3D hand mesh. Each row specifies three vertex indices that form a triangle, connecting the `pred_vertices` to create the surface of the 3D model. This is directly used by the `Renderer` to construct the 3D mesh.

*   **`focal_length`** (`numpy.ndarray`, scalar)
    *   **Explanation**: The effective focal length used for the camera projection. This value is derived from `model_cfg` and `img_size` and is a critical parameter for the `Renderer` to perform accurate 3D-to-2D projection.

These outputs collectively provide all the necessary information for a separate rendering process to reconstruct the 3D hand pose and overlay it onto the original image without needing to re-run the computationally expensive inference models.
