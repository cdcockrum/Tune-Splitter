# pipeline.py

from log import logger, initialize_models
initialize_models()

import os
import hashlib
import queue
import threading
import json
import shlex
import sys
import subprocess
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from utils import (
    remove_directory_contents,
    create_directories,
    download_manager,
)
import random
import spaces
import onnxruntime as ort
import warnings
import spaces
import gradio as gr
import logging
import time
import traceback
from pedalboard import Pedalboard, Reverb, Delay, Chorus, Compressor, Gain, HighpassFilter, LowpassFilter
from pedalboard.io import AudioFile
import numpy as np
import yt_dlp

# path fix to ensure consistent mdx_models dir usage
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
mdxnet_models_dir = BASE_DIR / "mdx_models"

# rest of your pipeline code continues...
