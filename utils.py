# utils.py

import os
import requests
from log import logger


def create_directories(path):
    os.makedirs(path, exist_ok=True)


def remove_directory_contents(path):
    if not os.path.exists(path):
        return
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)


def download_manager(url, output_dir):
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path):
        return output_path

    logger.info(f"Downloading: {filename}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure directory exists

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    logger.info(f"Downloaded: {filename}")
    return output_path
