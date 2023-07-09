import cv2
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple


def warp_image(
    source_keypoints: List[Tuple[int, int]], 
    target_keypoints: List[Tuple[int, int]], 
    source_img: np.array, 
    target_img: np.array
) -> np.array:
    """
    Warps source image to target image
    """
    tri = Delaunay(target_keypoints, qhull_options='QJ')

    source_points = np.array(source_keypoints, np.int32)
    target_points = np.array(target_keypoints, np.int32)

    affine_transforms = _calculate_affine_transform(source_points, target_points, tri)

    out_img = np.zeros(target_img.shape, dtype=np.uint8)
    triangles = target_points[tri.simplices].astype(np.int32)

    for i, tri in enumerate(triangles):
        mask = _get_convex_hull_mask(tri, target_img.shape[:2])
        mask = np.expand_dims(mask, axis=-1)

        warped_img = cv2.warpAffine(
            source_img, 
            affine_transforms[i], 
            (target_img.shape[1], target_img.shape[0]), 
            None, 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.uint8)

        out_img = (out_img * (1 - mask)) + (warped_img * mask)

    return out_img


def swap_image(
    source_img: np.array, target_img: np.array, keypoints: List[Tuple[int, int]], points: Tuple[int, int, int, int]
) -> Tuple[np.array, np.array]:
    """
    Swaps face from source image into target image
    """
    center = _calculate_center(points)

    mask_img = _get_mask_from_points(keypoints, target_img.shape[:2])

    adjusted_img = _color_correction(source_img, target_img, mask_img)

    swap = cv2.seamlessClone(adjusted_img, target_img, mask_img * 255, center, cv2.NORMAL_CLONE)

    return swap, mask_img


def get_bounding_box(keypoints: List[Tuple[int, int]], x_offset: int, y_offset: int) -> Tuple[int, int, int, int]:
    """
    Gets bounding box from keypoints
    """
    x0, y0, x1, y1 = np.inf, np.inf, -np.inf, -np.inf

    for keypoint in keypoints:
        x, y = keypoint

        if x > x1: x1 = x
        if x < x0: x0 = x
        if y > y1: y1 = y
        if y < y0: y0 = y

    return x0 - x_offset, y0 - y_offset, x1 + x_offset, y1 + y_offset


def crop_image(img: np.array, points: Tuple[int, int, int, int]) -> np.array:
    """
    Crops image 
    """
    x0, y0, x1, y1 = points
    
    return img[y0:y1, x0:x1]


def reconstruct_image(
        target_img: np.array,
        target_crop: np.array,
        source_crop: np.array,
        mask_crop: np.array,
        points: Tuple[int, int, int, int]
) -> np.array:
    """
    Reconstructs cropped image into original shape
    """
    x0, y0, x1, y1 = points

    reconstructed_img = target_img.copy()
    mask_crop = _dilate_mask(mask_crop.copy())

    reconstructed_crop = source_crop * mask_crop + target_crop * (1 - mask_crop)
    reconstructed_img[y0:y1, x0:x1] = reconstructed_crop

    return reconstructed_img


def _dilate_mask(mask: np.array) -> np.array:
    """
    Dilates the mask with Gaussian blur
    """
    mask = np.stack((mask,)*3, axis=-1)
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15,15), 15/3)
    
    return mask


def _calculate_center(points: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculates center of convexhull shape
    """
    x0, y0, x1, y1 = points

    return x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2


def _color_correction(source_img: np.array, target_img: np.array, mask: np.array) -> np.array:
    """
    Correct collors of source image based on the target image in mask region
    """
    source_region = (source_img * np.expand_dims(mask, axis=-1))
    target_region = (target_img * np.expand_dims(mask, axis=-1))

    mean_ratio = lambda c: np.nanmean(target_region[:, :, c]) / np.nanmean(source_region[:, :, c])
    color_scale_factor = [mean_ratio(c) for c in range(source_region.shape[-1])]

    source_img_adj = (source_img * np.array(color_scale_factor))

    return np.clip(source_img_adj, 0, 255).astype('uint8')


def _calculate_affine_transform(points1: np.array, points2: np.array, tri: Delaunay) -> List:
    """
    Calculates affine transformation on the set of points
    """
    tri1, tri2 = points1[tri.simplices].astype(np.float32), points2[tri.simplices].astype(np.float32)
    
    return [cv2.getAffineTransform(tri1[i, :, :], tri2[i, :, :]) for i in range(tri1.shape[0])]


def _get_convex_hull_mask(poly: np.array, shape: Tuple[int, int]) -> np.array:
    """
    Gets convex hull mask
    """
    mask = np.zeros(shape, np.uint8)

    return cv2.fillConvexPoly(mask, poly, 1)


def _get_mask_from_points(keypoints: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.array:
    points = np.array(keypoints, np.int32)
    convexhull = cv2.convexHull(points.round().astype('int64'))

    return _get_convex_hull_mask(convexhull, shape)
