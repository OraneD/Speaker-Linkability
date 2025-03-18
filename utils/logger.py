import logging
import datetime
from datetime import datetime
import os

import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", name="SimMatrix"):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"{now}_{name}.log")

    logger = logging.getLogger(name)  
    if not logger.hasHandlers(): 
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        stream_handler = logging.StreamHandler()  # Affiche les logs dans la console
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logger.addHandler(handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

    return logger


    