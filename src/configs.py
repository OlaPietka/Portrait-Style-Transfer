STYLE_LAYERS = {
    "block1_conv1": 0.4,
    "block2_conv1": 0.2,
    "block3_conv1": 0.1,
    "block4_conv1": 0.2,
    "block5_conv1": 0.1
}
CONTENT_LAYERS = {
    "block1_conv1": 0.0,
    "block2_conv1": 0.0,
    "block3_conv1": 0.4,
    "block4_conv1": 0.6,
    "block5_conv2": 0.0
}
HISTOGRAM_LAYERS = {
    "block1_conv1": 0.5,
    "block2_conv1": 0.0,
    "block3_conv1": 0.0,
    "block4_conv1": 0.5,
    "block5_conv1": 0.0
}

N_STYLE_LAYERS = sum([1 for i in STYLE_LAYERS.values() if i > 0])
N_CONTENT_LAYERS = sum([1 for i in CONTENT_LAYERS.values() if i > 0])
N_HISTOGRAM_LAYERS = sum([1 for i in HISTOGRAM_LAYERS.values() if i > 0])

ALPHA = 5.0
BETA = 10000.0
THETA = 2000.0
GAMMA = 0.05
DEPTH = 2
ITERATIONS_P1 = 250
ITERATIONS_P2 = 500
LEARNING_RATE_P1 = 2.0
LEARNING_RATE_P2 = 1.0

CONTENT, STYLE, OUTPUT = 0, 1, 2

LOSS_REPORT_ITER = 10
