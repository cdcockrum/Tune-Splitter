# log.py

import logging

logger = logging.getLogger("audio_splitter")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

# Auto model/data setup from anywhere
import os
import json
from pathlib import Path
from mdx_core import MDX

BASE_DIR = Path(__file__).resolve().parent
mdxnet_models_dir = BASE_DIR / "mdx_models"

MODEL_PRESETS = {
    "UVR-MDX-NET-Voc_FT.onnx": {
        "mdx_dim_f_set": 2048,
        "mdx_dim_t_set": 3,
        "mdx_n_fft_scale_set": 6144,
        "primary_stem": "Vocals",
        "compensate": 1.035
    },
    "UVR_MDXNET_KARA_2.onnx": {
        "mdx_dim_f_set": 1024,
        "mdx_dim_t_set": 3,
        "mdx_n_fft_scale_set": 4096,
        "primary_stem": "Main",
        "compensate": 1.035
    },
    "UVR-MDX-NET-Inst_HQ_4.onnx": {
        "mdx_dim_f_set": 2048,
        "mdx_dim_t_set": 3,
        "mdx_n_fft_scale_set": 6144,
        "primary_stem": "Instrumental",
        "compensate": 1.0
    },
    "Reverb_HQ_By_FoxJoy.onnx": {
        "mdx_dim_f_set": 2048,
        "mdx_dim_t_set": 3,
        "mdx_n_fft_scale_set": 6144,
        "primary_stem": "Vocals",
        "compensate": 1.035
    },
}


def initialize_models():
    mdxnet_models_dir.mkdir(parents=True, exist_ok=True)
    json_path = mdxnet_models_dir / "data.json"

    if json_path.exists():
        logger.info("✅ data.json already exists.")
        return

    logger.info("🔧 Generating data.json from local ONNX models")
    data = {}
    for filename, params in MODEL_PRESETS.items():
        model_path = mdxnet_models_dir / filename
        if model_path.exists():
            model_hash = MDX.get_hash(str(model_path))
            data[model_hash] = params

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"✅ Created data.json with {len(data)} entries.")
