import yaml
import os

with open("configs/main.yml", 'r') as yml_file:
    cfg = yaml.load(yml_file, Loader=yaml.FullLoader)
os.environ["KERAS_BACKEND"] = cfg['keras_backend']
