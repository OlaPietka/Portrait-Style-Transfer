import tensorflow.compat.v1 as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras import Model


def feature_extractor(img: tf.Tensor) -> Model:
    """
    Gets features from VGG19 for given Tensor
  
    Parameters:
    img: Tensor with images (b, h, w, 3) (where `b` is number of images)
  
    Returns: Features model
    """
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs=dict([(layer.name, layer.output) for layer in model.layers])
    
    return Model(inputs=model.inputs, outputs=outputs)(img)


def gain_maps(
    content: tf.Tensor, style: tf.Tensor, min_g: float = 0.7, max_g: float = 5.0
) -> tf.Tensor:
    """
    Modifies future maps
  
    Parameters:
    content: Content features Tensor (h, w, c)
    style: Style features Tensor (h, w, c)
    min_g: Minimal gain
    max_g: Maximal gain

    Returns: Modified content features 
    """
    etha = 1e-4

    g = style / (content + etha)
    g = tf.clip_by_value(g, min_g, max_g)

    return content * g
