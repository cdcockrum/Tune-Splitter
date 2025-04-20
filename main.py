# main.py

import os
from pathlib import Path
from gui import get_gui
from utils import download_manager

MDX_DOWNLOAD_LINK = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
UVR_MODELS = [
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR_MDXNET_KARA_2.onnx",
    "Reverb_HQ_By_FoxJoy.onnx",
    "UVR-MDX-NET-Inst_HQ_4.onnx",
]

BASE_DIR = Path(__file__).resolve().parent
mdxnet_models_dir = BASE_DIR / "mdx_models"

# Ensure mdx_models is a directory
if mdxnet_models_dir.exists() and not mdxnet_models_dir.is_dir():
    mdxnet_models_dir.unlink()
mdxnet_models_dir.mkdir(parents=True, exist_ok=True)

# Debug: confirm visibility of data.json
json_path = mdxnet_models_dir / "data.json"
print("🔍 Checking for data.json at:", json_path)
print("📁 mdx_models contents:", list(mdxnet_models_dir.iterdir()))
print("📂 Current working directory:", os.getcwd())

if not json_path.exists():
    print("❌ data.json NOT FOUND! Check if it's committed or misplaced.")
else:
    print("✅ data.json found.")


def download_models():
    for model in UVR_MODELS:
        url = os.path.join(MDX_DOWNLOAD_LINK, model)
        download_manager(url, str(mdxnet_models_dir))


if __name__ == "__main__":
    download_models()
    theme = "NoCrypt/miku"
    app = get_gui(theme)
    app.queue(default_concurrency_limit=40)
    app.launch(
        max_threads=40,
        share=False,
        show_error=True,
        quiet=False,
        debug=False,
    )
