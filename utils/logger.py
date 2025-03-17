import logging
import datetime
from datetime import datetime
import os

def setup_logger(log_dir, matrix_type):
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{now}.log")

        logging.basiConfig(level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
            ]
        )
        return logging.getLogger(matrix_type)

    