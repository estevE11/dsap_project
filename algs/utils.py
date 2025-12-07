import numpy as np
import librosa
import soundfile as sf
import os

def load_audio(file_path, sr=44100):
    """
    Load mono audio at specified sample rate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def save_audio(file_path, data, sr=44100):
    """
    Save audio to file.
    """
    sf.write(file_path, data, sr)
    print(f"-> Saved '{os.path.basename(file_path)}'")

def load_and_prepare_sources(data_path, inst1_filename, inst2_filename, sr=44100):
    """
    Loads two instrument files from the data path and prepares the source matrix.
    """
    p1 = os.path.join(data_path, inst1_filename)
    p2 = os.path.join(data_path, inst2_filename)
    
    print("Loading clean stems...")
    return prepare_source_matrix([p1, p2], sr=sr)

def prepare_source_matrix(file_paths, sr=44100):
    """
    Loads multiple audio files, trims them to the minimum length,
    and stacks them into a source matrix S.
    """
    signals = []
    for p in file_paths:
        signals.append(load_audio(p, sr=sr))
    
    # Trim to min length
    min_len = min(len(s) for s in signals)
    signals = [s[:min_len] for s in signals]
    
    # Stack
    S = np.vstack(signals)
    return S, min_len
