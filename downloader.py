# downloader.py

import os
import yt_dlp
import gradio as gr


def audio_downloader(url_media: str):
    url_media = url_media.strip()
    if not url_media:
        return None

    dir_output_downloads = "downloads"
    os.makedirs(dir_output_downloads, exist_ok=True)

    media_info = yt_dlp.YoutubeDL({
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True
    }).extract_info(url_media, download=False)

    download_path = f"{os.path.join(dir_output_downloads, media_info['title'])}.m4a"

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a'
        }],
        'force_overwrites': True,
        'noplaylist': True,
        'no_warnings': True,
        'quiet': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': download_path
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url_media])

    return download_path


# Gradio downloader-related UI components

def downloader_conf():
    return gr.Checkbox(False, label="URL-to-Audio", container=False)

def url_media_conf():
    return gr.Textbox("", label="Enter URL", placeholder="www.youtube.com/watch?v=abc123", visible=False, lines=1)

def url_button_conf():
    return gr.Button("Go", variant="secondary", visible=False)

def show_components_downloader(value_active):
    return gr.update(visible=value_active), gr.update(visible=value_active)
