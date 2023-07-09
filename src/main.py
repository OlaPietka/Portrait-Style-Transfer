import argparse
from typing import Tuple

import numpy as np
import tensorflow.compat.v1 as tf

from configs import (
    ALPHA, BETA, THETA, GAMMA, DEPTH,
    ITERATIONS_P1, ITERATIONS_P2, LEARNING_RATE_P1, LEARNING_RATE_P2
)
from io_utils import (
    get_image,
    save_image
)
from keypoints import (
    get_keypoints,
    get_detector_and_predictor
)
from training import (
    phase_1,
    phase_2
)
from transform_images import (
    warp_image,
    swap_image,
    get_bounding_box,
    crop_image,
    reconstruct_image
)
from transform_tensors import (
    to_tensor_image,
    deprocess_tensor_image
)

tf.disable_v2_behavior()
tf.enable_eager_execution()
tf.config.run_functions_eagerly(True)


def face_swap(source_img: np.array, target_img: np.array, predictor_path: str) -> Tuple[np.array, np.array, Tuple]:
    """
    Performs warping and aligment of source image into target image
    """
    detector, predictor = get_detector_and_predictor(predictor_path)

    source_keypoints = get_keypoints(source_img, detector, predictor)
    target_keypoints = get_keypoints(target_img, detector, predictor)

    warped_img = warp_image(source_keypoints, target_keypoints, source_img, target_img)

    bbox_points = get_bounding_box(target_keypoints, x_offset=20, y_offset=20)

    swapped_img, mask_img = swap_image(warped_img, target_img, target_keypoints, bbox_points)

    return swapped_img, mask_img, bbox_points


def prepare_inputs(
        source_img: np.array, target_img: np.array, mask_img: np.array, bbox_points: Tuple[int, int, int, int]
) -> Tuple[np.array, np.array, np.array]:
    """
    Crops images to bounding box
    """
    source_img = crop_image(source_img, bbox_points)
    target_crop = crop_image(target_img, bbox_points)
    mask_crop = crop_image(mask_img, bbox_points)

    return source_img, target_crop, mask_crop


def transfer_style(
    content_img: np.array, 
    style_img: np.array, 
    mask_img: np.array
) -> np.array:
    """
    Runs style transfer process
    """
    content_tensor = to_tensor_image(content_img)
    style_tensor = to_tensor_image(style_img)
    output_tensor = to_tensor_image(content_img)
    mask_tensor = mask_img.copy()

    print(f"--PHASE 1--")
    output_interim = phase_1(
        content_tensor, 
        style_tensor, 
        output_tensor, 
        mask_tensor,
        depth=DEPTH,
        weights=[ALPHA, BETA, 0, 0], 
        iterations=ITERATIONS_P1, 
        lr=LEARNING_RATE_P1
    ) 

    print(f"--PHASE 2--")
    output = phase_2(
        content_tensor, 
        style_tensor, 
        output_interim, 
        mask_tensor,
        weights=[ALPHA, BETA, THETA, GAMMA], 
        iterations=ITERATIONS_P2, 
        lr=LEARNING_RATE_P2
    )

    return deprocess_tensor_image(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_image", type=str, required=True)
    parser.add_argument("-s", "--source_image", type=str, required=True)
    parser.add_argument("-t", "--target_image", type=str, required=True)
    parser.add_argument("-p", "--predictor", type=str, default="data/shape_predictor_68_face_landmarks.dat")
    args = parser.parse_args()

    source_img = get_image(args.source_image)
    target_img = get_image(args.tagret_image)

    swapped_img, mask_img, bbox_points = face_swap(source_img, target_img, args.predictor)

    source_crop, target_crop, mask_crop = prepare_inputs(swapped_img, target_img, mask_img, bbox_points)

    output_crop = transfer_style(source_crop, target_crop, mask_crop)

    output_img = reconstruct_image(target_img, target_crop, output_crop, mask_crop, bbox_points)

    save_image(output_img, args.output_image)
