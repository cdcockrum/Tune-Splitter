# inference.py

import os
import gc
import json
import shlex
import sys
import torch
import librosa
import numpy as np
import subprocess
import soundfile as sf
import hashlib
import random
import time
import traceback
import onnxruntime as ort
from utils import logger, remove_directory_contents, create_directories
from mdx_core import MDX, MDXModel
from effects import add_vocal_effects, add_instrumental_effects


stem_naming = {
    "Vocals": "Instrumental",
    "Other": "Instruments",
    "Instrumental": "Vocals",
    "Drums": "Drumless",
    "Bass": "Bassless",
}


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False,
            suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2, device_base="cuda"):

    device = torch.device("cuda:0" if device_base == "cuda" else "cpu")
    processor_num = 0 if device_base == "cuda" else -1

    if device_base == "cuda":
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        m_threads = 1 if vram_gb < 8 else (8 if vram_gb > 32 else 2)
        logger.info(f"threads: {m_threads} vram: {vram_gb}")
    else:
        m_threads = 1

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)

    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak

    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)

    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    torch.cuda.empty_cache()
    return main_filepath, invert_filepath


def run_mdx_beta(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False,
                 suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=1, device_base=""):

    duration = librosa.get_duration(filename=filename)
    if duration >= 60 and duration <= 120:
        m_threads = 8
    elif duration > 120:
        m_threads = 16

    logger.info(f"threads: {m_threads}")

    device = torch.device("cpu")
    processor_num = -1

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)

    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"],
    )

    mdx_sess = MDX(model_path, model, processor=processor_num)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak

    if denoise:
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)

    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(main_filepath, wave_processed.T, sr)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(
            output_dir,
            f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav",
        )
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)

    if not keep_orig:
        os.remove(filename)

    del mdx_sess, wave_processed, wave
    gc.collect()
    torch.cuda.empty_cache()
    return main_filepath, invert_filepath


def convert_to_stereo_and_wav(audio_path, output_dir):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    if type(wave[0]) != np.ndarray or audio_path[-4:].lower() != ".wav":
        stereo_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_stereo.wav")
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}")
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return stereo_path
    return audio_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:18]


def random_sleep():
    time.sleep(round(random.uniform(5.2, 7.9), 1))
