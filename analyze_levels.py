import numpy as np
import soundfile as sf
import os

DATA_DIR = "/home/esteve/dev/dsap_project/run/custom"

def load_wav(filename):
    filepath = os.path.join(DATA_DIR, filename)
    audio, fs = sf.read(filepath)
    if audio.ndim > 1: audio = audio[:, 0]
    return audio, fs

def db(x):
    return 10 * np.log10(np.mean(x**2))

def main():
    # Load Clean Signals (Performance)
    clean_v, fs = load_wav("test_clean_mic_violin.wav")
    clean_c, _ = load_wav("test_clean_mic_cello.wav")
    
    # Load Mix
    mix_v, _ = load_wav("test_mix_mic_violin.wav")
    
    # Calculate Bleed (Mix - Clean)
    # Note: This assumes perfect alignment and linearity, which pyroomacoustics provides.
    min_len = min(len(clean_v), len(mix_v))
    bleed = mix_v[:min_len] - clean_v[:min_len]
    
    print(f"Violin Signal Power: {db(clean_v):.2f} dB")
    print(f"Cello Bleed Power:   {db(bleed):.2f} dB")
    print(f"SIR (Signal to Interference): {db(clean_v) - db(bleed):.2f} dB")
    
    # Check Calibration File Levels
    calib_c_at_v, _ = load_wav("calib_cello_source_mic_violin.wav")
    calib_c_at_c, _ = load_wav("calib_cello_source_mic_cello.wav")
    
    print(f"\nCalibration Cello at Cello Mic: {db(calib_c_at_c):.2f} dB")
    print(f"Calibration Cello at Violin Mic: {db(calib_c_at_v):.2f} dB")
    print(f"Leakage Attenuation: {db(calib_c_at_c) - db(calib_c_at_v):.2f} dB")

if __name__ == "__main__":
    main()
