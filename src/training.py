from typing import List, Tuple

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_graphics.image import pyramid

from configs import (
    CONTENT, STYLE, OUTPUT,
    CONTENT_LAYERS, STYLE_LAYERS, HISTOGRAM_LAYERS,
    N_CONTENT_LAYERS, N_STYLE_LAYERS, N_HISTOGRAM_LAYERS,
    LOSS_REPORT_ITER,
)

from features import (
    feature_extractor,
    gain_maps
)

from io_utils import report_stats

from losses import (
    content_loss,
    style_loss,
    histogram_loss,
    total_variation_loss
)

from transform_tensors import (
    clip_tensor_image,
    mask_tensor_image,
)


def _sum_content_loss(features: Model) -> tf.Tensor:
    """
    Extracts the content layers + content loss
    """
    loss = 0.0

    for layer_name, layer_weight in CONTENT_LAYERS.items():
        if layer_weight == 0:
            continue

        content_features = features[layer_name][CONTENT, :, :, :]
        style_features = features[layer_name][STYLE, :, :, :]
        output_features = features[layer_name][OUTPUT, :, :, :]

        content_features = gain_maps(content_features, style_features, 0.7, 5.0)

        loss += content_loss(content_features, output_features) * layer_weight
    loss /= N_CONTENT_LAYERS

    return loss


def _sum_style_loss(features: Model, mask: np.array) -> tf.Tensor:
    """
    Extracts the content layers + content loss
    """
    loss = 0.0

    for layer_name, layer_weight in STYLE_LAYERS.items():
        if layer_weight == 0:
            continue

        style_features = features[layer_name][STYLE, :, :, :]
        output_features = features[layer_name][OUTPUT, :, :, :]

        style_features = mask_tensor_image(style_features, mask)
        output_features = mask_tensor_image(output_features, mask)
    
        loss += style_loss(style_features, output_features) * layer_weight
    loss /= N_STYLE_LAYERS

    return loss


def _sum_histogram_loss(features: Model, mask: np.array) -> tf.Tensor:
    """
    Extracts the histogram layers + histogram loss
    """
    loss = 0.0

    for layer_name, layer_weight in HISTOGRAM_LAYERS.items():
        if layer_weight == 0:
            continue

        style_features = features[layer_name][STYLE, :, :, :]
        output_features = features[layer_name][OUTPUT, :, :, :]

        style_features = mask_tensor_image(style_features, mask)
        output_features = mask_tensor_image(output_features, mask)
        
        loss += histogram_loss(style_features, output_features) * layer_weight
    loss /= N_HISTOGRAM_LAYERS

    return loss


def _loss_function(
    content: tf.Tensor, 
    style: tf.Tensor, 
    output: tf.Tensor,
    mask: np.array,
    weights: List[int]
) -> Tuple[float, float, float, float, float]:
    """
    Computes total loss
    """
    input_tensor = tf.concat([content, style, output], axis=0)
    features = feature_extractor(input_tensor)

    alpha, beta, theta, gamma = weights
    c_loss = s_loss = h_loss = tv_loss = 0
    if alpha:
        c_loss = alpha * _sum_content_loss(features)
    if beta:
        s_loss = beta * _sum_style_loss(features, mask)
    if theta:
        h_loss = theta * _sum_histogram_loss(features, mask)
    if gamma:
        tv_loss = gamma * total_variation_loss(output)

    loss = c_loss + s_loss + h_loss + tv_loss

    return loss, c_loss, s_loss, h_loss, tv_loss


@tf.function()
def _compute_loss_and_grads(
    content: tf.Tensor, 
    style: tf.Tensor, 
    output: tf.Tensor,
    mask: np.array,
    weights: List[int],
    optimizer
) -> Tuple[float, float, float, float, float, float]:
    """
    Computes total loss
    """
    with tf.GradientTape() as tape:
        loss, c_loss, s_loss, h_loss, tv_loss = _loss_function(
            content, style, output, mask, weights
        )
    
    grads = tape.gradient(loss, output)
    optimizer.apply_gradients([(grads, output)])

    clipped = clip_tensor_image(output)
    output.assign(clipped)

    return loss, c_loss, s_loss, h_loss, tv_loss, grads


def _train(
    content: tf.Tensor, 
    style: tf.Tensor,
    output: tf.Tensor,
    mask: np.array,
    weights: List[int],
    iterations: int,
    lr: float
) -> tf.Tensor:
    """
    Runs single learning phase for n iterations
    """
    
    optimizer = Adam(learning_rate=lr, beta_1=0.99, epsilon=1e-1)
    for i in range(1, iterations+1): 
        loss, c_loss, s_loss, h_loss, tv_loss, grads = _compute_loss_and_grads(
            content, style, output, mask, weights, optimizer
        )

        if i % LOSS_REPORT_ITER == 0:
            report_stats(i, loss, c_loss, s_loss, h_loss, tv_loss)

    return output


def phase_1(
    content: tf.Tensor, 
    style: tf.Tensor,
    output: tf.Tensor,
    mask: np.array,
    depth: int,
    weights: List[int],
    iterations: int,
    lr: float
) -> List:
    """
    Runs phase 1 of learning process on each lvl of pyramid
    """
    # get pyramids for each tensor
    content_pyramid = pyramid.downsample(content, depth)
    style_pyramid = pyramid.downsample(style, depth)
    output_pyramid = pyramid.split(output, depth)
    
    # init output to be the last level of pyramid
    output = output_pyramid[depth]
    for d in range(depth, -1, -1):
        content = content_pyramid[d]
        style = style_pyramid[d]
        output = tf.Variable(output)

        print(f"Run pyramid lvl {d}")
        output = _train(
            content, style, output, mask, weights, iterations, lr
        )

        if d == 0: 
            return output

        upsample_output = pyramid.upsample(output, 1)[1]
        next_output = output_pyramid[d-1]

        # reshape due to possible upsample artifacts
        _, h, w, c = next_output.get_shape().as_list()
        output = upsample_output[:, :h, :w, :] + next_output


def phase_2(
    content: tf.Tensor, 
    style: tf.Tensor,
    output: tf.Tensor,
    mask: np.array,
    weights: List[int],
    iterations: int,
    lr: float
) -> tf.Tensor:
    """
    Runs phase 2 of learning process on output from phase 1
    """
    output = tf.Variable(output)

    output = _train(
        content, style, output, mask, weights, iterations, lr
    )

    return output
