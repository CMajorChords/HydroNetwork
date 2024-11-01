import yaml
import os


def set_keras_backend():
    with open("configs/main.yml", 'r') as yml_file:
        cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
    os.environ["KERAS_BACKEND"] = cfg['keras_backend']


def tensor2numpy(tensor):
    # 检查device
    if tensor.device.type == "cuda":
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()
