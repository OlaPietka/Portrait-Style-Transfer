import numpy as np
import cv2

import tensorflow.compat.v1 as tf
from tensorflow.keras.applications import vgg19


def to_tensor_image(img: np.array) -> tf.Tensor:
    """
    Formats image into appropriate tensor
    """
    img = img.astype(np.float32)
    img = np.expand_dims(img.copy(), axis=0)
    img = vgg19.preprocess_input(img)
    
    return tf.convert_to_tensor(img)


def deprocess_tensor_image(img: tf.Tensor) -> np.array:
    """
    Deprocess tensors into image
    """
    img = img.numpy()

    _, h, w, c = img.shape
    img = img.reshape((h, w, 3))
    
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    img = img[:, :, ::-1]

    return np.clip(img, 0, 255).astype("uint8")


def clip_tensor_image(img: tf.Tensor) -> tf.Tensor:
    """
    Clips tensor image to min and max value
    """
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = - norm_means
    max_vals = 255 - norm_means
    
    return tf.clip_by_value(img, min_vals, max_vals)


def mask_tensor_image(img: tf.Tensor, mask: np.array) -> tf.Tensor:
    """
    Masks Tensor image with mask
    """
    def _preprocess_mask(mask: np.array) -> tf.Tensor:
        # Formats mask into appropriate tensor
        h, w, c = img.get_shape().as_list()

        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_AREA)

        mask = tf.convert_to_tensor(mask)

        tensors = []
        for _ in range(c): 
            tensors.append(mask)
        mask = tf.stack(tensors, axis=2)
        mask = tf.stack(mask, axis=0)

        return mask

    mask_preprocessed = _preprocess_mask(mask)

    return img * tf.stop_gradient(mask_preprocessed)

