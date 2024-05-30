import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Define the log file path with a timestamp
log_file_path = os.path.join(logs_path, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)