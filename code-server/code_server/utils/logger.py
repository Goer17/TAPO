import os
import logging
from typing import Dict
from datetime import datetime
from pathlib import Path

def __logger():
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    path = Path("logs") / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
	
    return logger

logger = __logger()

def format_exec_info(info: Dict[str, str | bool]):
    code_lines = info.get('code', '').split('\n')
    tabbed_code = "\n\t".join(code_lines)

    stdout_lines = info.get('stdout', '').split('\n')
    tabbed_stdout = "\n\t".join(stdout_lines)

    return (
        f"\ncode:\n\t{tabbed_code}\n"
        f"stdout:\n\t{tabbed_stdout}\n"
        f"finished:\t{info.get('finished', False)}\t\ttime:\t{info.get('time', '')}\n"
    )