# gui.py

import gradio as gr
from downloader import downloader_conf, url_media_conf, url_button_conf, show_components_downloader
from pipeline import sound_separate
from log import logger  # ensure logger source is consistent

# Constants
TITLE = "<center><strong><font size='7'>Audio🔹separator</font></strong></center>"
DESCRIPTION = "This demo uses the MDX-Net models for vocal and background sound separation."


def get_gui(theme):
    with gr.Blocks(theme=theme) as app:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        # Downloader UI
        downloader_gui = downloader_conf()
        with gr.Row():
            with gr.Column(scale=2):
                url_media_gui = url_media_conf()
            with gr.Column(scale=1):
                url_button_gui = url_button_conf()

        downloader_gui.change(
            show_components_downloader,
            [downloader_gui],
            [url_media_gui, url_button_gui]
        )

        # Audio file upload input
        aud = gr.File(label="Audio file", type="filepath")

        url_button_gui.click(
            fn=audio_downloader,
            inputs=[url_media_gui],
            outputs=[aud]
        )

        # Stem choice
        stem_gui = gr.Radio(choices=["vocal", "background"], value="vocal", label="Stem")

        # Options for effects
        main_gui = gr.Checkbox(False, label="Main")
        dereverb_gui = gr.Checkbox(False, label="Dereverb", visible=True)
        vocal_effects_gui = gr.Checkbox(False, label="Vocal Effects", visible=True)
        background_effects_gui = gr.Checkbox(False, label="Background Effects", visible=False)

        stem_gui.change(
            lambda stem: (
                gr.update(visible=stem == "vocal"),
                gr.update(visible=stem == "vocal"),
                gr.update(visible=stem == "vocal"),
                gr.update(visible=stem == "background"),
            ),
            [stem_gui],
            [main_gui, dereverb_gui, vocal_effects_gui, background_effects_gui]
        )

        # Basic trigger
        button_base = gr.Button("Inference", variant="primary")
        output_base = gr.File(label="Result", file_count="multiple", interactive=False)

        button_base.click(
            fn=sound_separate,
            inputs=[
                aud, stem_gui, main_gui, dereverb_gui,
                vocal_effects_gui, background_effects_gui
            ],
            outputs=[output_base]
        )

    return app

