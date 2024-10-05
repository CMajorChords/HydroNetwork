import yaml
import os

with open("configs/main.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
os.environ["KERAS_BACKEND"] = cfg['keras_backend']