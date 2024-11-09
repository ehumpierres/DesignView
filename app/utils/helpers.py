import logging
from functools import wraps
from time import time
from typing import Callable
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(f: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        logger.info(f'Function {f.__name__} took {end-start:.2f} seconds')
        return result
    return wrap

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)