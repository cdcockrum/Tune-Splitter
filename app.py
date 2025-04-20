import os
import tempfile
import subprocess
import gradio as gr
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demucs")

DEFAULT_MODEL = "htdemucs"

print("✅ App is starting...")
print(f"✅ Using Python at {sys.executable}")

def run_demucs(audio_path, selected_stems, model_name=DEFAULT_MODEL):
    try:
        logger.info(f"Running Demucs on {audio_path}")
        print("🔄 Starting demucs process...")
        output_dir = tempfile.mkdtemp()

        cmd = f"{sys.executable} -m demucs -n {model_name} -o {output_dir} \"{audio_path}\""
        logger.info(f"Executing command: {cmd}")
        print(f"🧪 Running command: {cmd}")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()
        print("✅ Process finished.")
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)

        if process.returncode != 0:
            logger.error(f"Demucs error: {stderr}")
            raise gr.Error(f"Demucs failed: {stderr}")

        track_name = Path(audio_path).stem
        stem_dir = Path(output_dir) / model_name / track_name

        output_files = []
        for stem in selected_stems:
            stem_path = stem_dir / f"{stem}.wav"
            if stem_path.exists():
                output_files.append(str(stem_path))

        if not output_files:
            raise gr.Error("No stems were generated")

        return output_files

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise gr.Error(f"Process failed: {str(e)}")


def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 🎚️ Demucs Stem Splitter")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")

        with gr.Row():
            vocals = gr.Checkbox(label="Vocals", value=True)
            drums = gr.Checkbox(label="Drums", value=True)
            bass = gr.Checkbox(label="Bass", value=True)
            other = gr.Checkbox(label="Other", value=True)

        with gr.Row():
            model_selector = gr.Dropdown(
                label="Model",
                choices=["htdemucs", "mdx_extra", "mdx_extra_q"],
                value="htdemucs"
            )

        with gr.Row():
            submit_btn = gr.Button("Split Stems")

        output = gr.File(label="Output Stems", file_count="multiple")

        def process(audio_file, vocals_enabled, drums_enabled, bass_enabled, other_enabled, model):
            selected = [stem for stem, enabled in [
                ("vocals", vocals_enabled),
                ("drums", drums_enabled),
                ("bass", bass_enabled),
                ("other", other_enabled),
            ] if enabled]

            if not selected:
                raise gr.Error("Please select at least one stem")

            return run_demucs(audio_file, selected, model)

        submit_btn.click(
            fn=process,
            inputs=[audio_input, vocals, drums, bass, other, model_selector],
            outputs=output
        )

    return interface


def main():
    print("🚀 Launching Gradio interface...")
    interface = create_interface()
    interface.launch()


if __name__ == '__main__':
    main()
