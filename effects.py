# effects.py

from pedalboard import Pedalboard, Reverb, Delay, Chorus, Compressor, Gain, HighpassFilter, LowpassFilter
from pedalboard.io import AudioFile


def add_vocal_effects(input_file, output_file,
                      reverb_room_size=0.6, vocal_reverb_dryness=0.8, reverb_damping=0.6, reverb_wet_level=0.35,
                      delay_seconds=0.4, delay_mix=0.25,
                      compressor_threshold_db=-25, compressor_ratio=3.5,
                      compressor_attack_ms=10, compressor_release_ms=60,
                      gain_db=3):

    effects = [HighpassFilter()]

    effects.append(Reverb(
        room_size=reverb_room_size,
        damping=reverb_damping,
        wet_level=reverb_wet_level,
        dry_level=vocal_reverb_dryness,
    ))

    effects.append(Compressor(
        threshold_db=compressor_threshold_db,
        ratio=compressor_ratio,
        attack_ms=compressor_attack_ms,
        release_ms=compressor_release_ms,
    ))

    if delay_seconds > 0 or delay_mix > 0:
        effects.append(Delay(delay_seconds=delay_seconds, mix=delay_mix))

    if gain_db:
        effects.append(Gain(gain_db=gain_db))

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)


def add_instrumental_effects(input_file, output_file,
                              highpass_freq=100, lowpass_freq=12000,
                              reverb_room_size=0.5, reverb_damping=0.5, reverb_wet_level=0.25,
                              compressor_threshold_db=-20, compressor_ratio=2.5,
                              compressor_attack_ms=15, compressor_release_ms=80,
                              gain_db=2):

    effects = [
        HighpassFilter(cutoff_frequency_hz=highpass_freq),
        LowpassFilter(cutoff_frequency_hz=lowpass_freq),
    ]

    effects.append(Reverb(
        room_size=reverb_room_size,
        damping=reverb_damping,
        wet_level=reverb_wet_level,
    ))

    effects.append(Compressor(
        threshold_db=compressor_threshold_db,
        ratio=compressor_ratio,
        attack_ms=compressor_attack_ms,
        release_ms=compressor_release_ms,
    ))

    if gain_db:
        effects.append(Gain(gain_db=gain_db))

    board = Pedalboard(effects)

    with AudioFile(input_file) as f:
        with AudioFile(output_file, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)
