import os
import time
from pathlib import Path


def get_log_folder(log_root: str):
    log_folder = Path(log_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder
