import tensorflow.compat.v1 as tf


def content_loss(content: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """
    Calculates content loss -> MSE(content, output)
    """
    return tf.reduce_mean(tf.square(output - content)) 


def style_loss(style: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """
    Calculates style loss -> MSE(G(content), G(output))
    """
    def _gram_matrix(x: tf.Tensor, M: int, N: int) -> tf.Tensor:
        # Calculates gram matrix
        x = tf.reshape(x, (M, N))

        return tf.matmul(tf.transpose(x), x)

    h, w, c = output.get_shape().as_list()

    M = h*w
    N = c

    S = _gram_matrix(style, M, N)
    O = _gram_matrix(output, M, N)

    return tf.reduce_mean(tf.square(S - O)) / (4 * M**2 * N**2)


def total_variation_loss(output: tf.Tensor) -> tf.Tensor:
    """
    Calculates total variance loss
    """
    x_deltas = output[:, 1:, :, :] - output[:, :-1, :, :]
    y_deltas = output[:, :, 1:, :] - output[:, :, :-1, :]
    sum_axis = (1, 2, 3)

    return tf.reduce_sum(tf.abs(x_deltas), axis=sum_axis) + tf.reduce_sum(tf.abs(y_deltas), axis=sum_axis)


def histogram_loss(style: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """
    Calculates histogram loss -> MSE(output, histogram)
    """
    def _quantiles(hist: tf.Tensor) -> tf.Tensor:
        # Computes quantiles for given histogram
        quantiles = tf.cumsum(hist)
        last_element = tf.subtract(tf.size(quantiles), tf.constant(1))
        
        return tf.divide(quantiles, tf.gather(quantiles, last_element))

    hist_bins = 255
    shape = output.get_shape()

    output = tf.layers.flatten(output)
    style = tf.layers.flatten(style)

    max_value = tf.reduce_max([tf.reduce_max(output), tf.reduce_max(style)])
    min_value = tf.reduce_min([tf.reduce_min(output), tf.reduce_min(style)])
    hist_delta = (max_value - min_value) / hist_bins

    hist_range = tf.range(min_value, max_value, hist_delta) + (hist_delta / 2)

    o_hist = tf.histogram_fixed_width(
        output, [min_value, max_value], nbins=hist_bins, dtype=tf.int64
    )
    s_hist = tf.histogram_fixed_width(
        style, [min_value, max_value], nbins=hist_bins, dtype=tf.int64
    )

    o_quantiles = _quantiles(o_hist)
    s_quantiles = _quantiles(s_hist)

    nearest_indices = tf.map_fn(
        lambda x: tf.argmin(tf.abs(tf.subtract(s_quantiles, x))), o_quantiles, dtype=tf.int64)

    o_bin_index = tf.cast(tf.divide(output, hist_delta), tf.int64)
    o_bin_index = tf.clip_by_value(o_bin_index, 0, 254)

    matched_to_s = tf.gather(hist_range, tf.gather(nearest_indices, o_bin_index))
    histogram = tf.reshape(matched_to_s, shape)
    output = tf.reshape(output, shape)  

    return tf.reduce_mean(tf.square(output - histogram))
