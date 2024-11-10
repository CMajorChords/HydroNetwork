import yaml
from numpy import ndarray
import os
from IPython import get_ipython


def set_keras_backend():
    with open("configs/main.yml", 'r') as yml_file:
        cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
    os.environ["KERAS_BACKEND"] = cfg['keras_backend']


def tensor2numpy(tensor):
    # 如果是numpy数组，直接返回
    if isinstance(tensor, ndarray):
        return tensor
    # 检查device
    if tensor.device.type == "cuda":
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def reset():
    """
    This function works in an IPython environment (such as Jupyter Notebook).
    It combines `%reset -f` functionality to clear variables,

    - %reset -f clears all user-defined variables.

    Notes:
    This code only works within an IPython environment and won't function fully in plain Python scripts.
    However, the screen clear part will work in standard Python scripts using system commands.
    """
    ipython = get_ipython()  # Get the active IPython instance.

    if ipython is not None:
        # Run the reset magic command to clear all variables.
        ipython.run_line_magic('reset', '-f')
    else:
        print("This function only works in an IPython environment like Jupyter or IPython shell.")


# Example usage:
# reset_and_clear()


def autoload():
    """
    自动加载模块
    """
    ipython = get_ipython()  # Get the active IPython instance.
    if ipython is not None:
        ipython.run_line_magic('load_ext', 'autoreload')
        ipython.run_line_magic('autoreload', '2')
    else:
        print("This function only works in an IPython environment like Jupyter or IPython shell.")
