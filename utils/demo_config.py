import os
import torch
from easydict import EasyDict as edict
from utils.common_utils import ws, dir_check, Logger


def get_demo_config():
    config = edict()

    # outputs
    config.PATH_OUT = os.path.join(ws, "outputs")
    os.makedirs(config.PATH_OUT, exist_ok=True)

    # device (no GPU auto-dispatch)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # demo hyperparams (keep small)
    config.epoch = 1
    config.lr = 1e-4
    config.batch_size = 16
    config.T_h = 12

    # data/model placeholders
    config.data = edict()
    config.model = edict()
    config.model.T_h = 12
    config.model.T_p = 12

    # logger
    config.logger = Logger()
    log_path = os.path.join(config.PATH_OUT, "demo.log")
    dir_check(log_path)
    config.logger.open(log_path, mode="w")

    return config
