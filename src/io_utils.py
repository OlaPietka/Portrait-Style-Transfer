import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_image(img_path: str) -> np.array:
    """
    Gets image 
    """
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def save_image(img: np.array, file_path: str) -> None:
    """
    Saves image
    """
    plt.imsave(file_path, img)


def report_stats(
    i: int,
    loss: float, 
    c_loss: float, 
    s_loss: float, 
    h_loss: float, 
    tv_loss: float
) -> None:
    """
    Prints losses stats
    """
    print("Iteration %d: t_l=%.2f c_l=%.2f s_l=%.2f h_l=%.2f tv_l=%.2f" % (i, loss, c_loss, s_loss, h_loss, tv_loss))

