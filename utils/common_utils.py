# -*- coding: utf-8 -*-
import os
import numpy as np
import torch


def get_workspace():
    """
    Workspace path (project root): .../STDiff_Demo
    """
    cur_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(cur_path)
    root_dir = os.path.dirname(utils_dir)
    return root_dir


ws = get_workspace()


def dir_check(path: str):
    """
    Ensure the directory for 'path' exists.
    If 'path' is a directory, create it; if it's a file path, create its parent dir.
    """
    d = path if os.path.isdir(path) else os.path.split(path)[0]
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)
    return path


class Logger(object):
    def __init__(self):
        import sys
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode="w"):
        dir_check(file)
        self.file = open(file, mode, encoding="utf-8")

    def write(self, message, is_terminal=True, is_file=True):
        # Avoid logging carriage-return progress bars into file
        if "\r" in message:
            is_file = False

        if is_terminal:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file and (self.file is not None):
            self.file.write(message)
            self.file.flush()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
