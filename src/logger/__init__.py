#logging: Python’s standard logging framework.
import logging 
# os: to build file paths and make folders.
import os 
# RotatingFileHandler: file handler that auto-rotates logs when they get big.
from logging.handlers import RotatingFileHandler
# datetime: to timestamp the log file name.
from datetime import datetime 
# sys: to send console logs to stdout.
import sys 

### Constants for log configuration
# All logs will live under a logs/ folder at the project root.
LOG_DIR = 'logs'
# Each run creates a timestamped file like 08_27_25_15_53_21.log.
LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
# When a log file hits 5 MB, it rolls over; at most 3 backups are kept (.1, .2, .3).
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
# # keep 3 rolled log files
BACKUP_COUNT = 3

### Build an absolute path to the log file
# root_dir: computes the project root by taking the directory of this file
# (src/logger/__init__.py), going up one (../), and absolutizing it.
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# log_dir_path: <project-root>/logs.
log_dir_path = os.path.join(root_dir, LOG_DIR)
# os.makedirs(..., exist_ok=True): create the folder if it doesn’t exist.
os.makedirs(log_dir_path, exist_ok=True)
# log_file_path: full path to the timestamped log file.
log_file_path = os.path.join(log_dir_path, LOG_FILE)

### Logger factory
def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Defines a function that returns a ready-to-use logger.
    # Gets the root logger.
    logger = logging.getLogger()
    # Sets the base threshold to DEBUG (handlers can still filter further).
    logger.setLevel(logging.DEBUG)
    # Log line format: time, logger name, level, message.
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Writes to the timestamped file, rotating at 5 MB and keeping 3 backups.
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    # File receives logs of INFO and above (no DEBUG noise on disk).
    file_handler.setLevel(logging.INFO)

    # Prints to the terminal (stdout).
    console_handler = logging.StreamHandler(sys.stdout)
    # Uses the same formatter for consistent log appearance.
    console_handler.setFormatter(formatter)
    # Only shows INFO level or higher logs in the console (so it won’t clutter with debug logs).
    console_handler.setLevel(logging.DEBUG)

    # Connects the two handlers (file + console) to the custom logger.
    logger.addHandler(file_handler)
    # Now, whenever you call logger.info(...), logger.error(...), etc.: The message will appear in the console.
    # The message will also be saved in a rotating log file
    logger.addHandler(console_handler)

#Runs the whole setup once, so logging works throughout your project.
configure_logger()



