from loguru import logger
import os

class LoguruLogger:
    def __init__(self, log_folder="logs", log_file="training.log"):
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, log_file)

        logger.remove()  # Remove default logger
        logger.add(log_path, level="INFO", format="{time} - {name} - {level} - {message}")
        logger.add(lambda msg: print(msg, end=""), level="INFO", format="{time} - {name} - {level} - {message}")

    def get_logger(self):
        return logger

# Create a global logger instance
global_logger = LoguruLogger().get_logger()
