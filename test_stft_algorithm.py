"""
Test script for STFT-based microphone cross-talk cancellation.
Uses the frequency-domain implementation in algorithm_stft.py
"""

import numpy as np
import librosa
from algs.least_squares_calibration.algorithm_stft import calibrate_and_separate

# Configuration
CALIB_FILE = "test_calib.mp3"
REC_FILE = "test_rec.mp3"
SR = 44100
N_FFT = 2048
HOP_LENGTH = 512
LAMBDA = 1.0  # Trust region parameter (σ²_w / σ²_ν)


def load_stereo(file_path, sr=44100):
    """Load stereo audio file."""
    y, _ = librosa.load(file_path, sr=sr, mono=False)
    if y.ndim == 1:
        # Duplicate to stereo if mono
        y = np.vstack([y, y])
    return y


def detect_active_frames_stft(X_calib, n_fft=2048, hop_length=512, threshold_db=-40, margin_db=10):
    """
    Detect frames where each source is active using STFT energy.
    
    Returns:
        List of (n_frames,) boolean masks
    """
    from algs.least_squares_calibration.algorithm_stft import compute_stft
    
    X_stft = compute_stft(X_calib, n_fft=n_fft, hop_length=hop_length)
    n_mics, n_freqs, n_frames = X_stft.shape
    
    # Compute energy per frame per mic
    energy = np.sum(np.abs(X_stft) ** 2, axis=1)  # (n_mics, n_frames)
    energy_db = 10 * np.log10(energy + 1e-10)
    energy_db = energy_db - np.max(energy_db)  # Normalize
    
    masks = []
    for m in range(n_mics):
        # Source m is dominant when:
        # 1. Mic m energy > threshold
        # 2. Mic m energy > other mics + margin
        mask = energy_db[m] > threshold_db
        
        for n in range(n_mics):
            if n != m:
                mask = mask & (energy_db[m] > energy_db[n] + margin_db)
        
        masks.append(mask)
        print(f"  Source {m+1}: {np.sum(mask)} / {n_frames} frames ({np.sum(mask)/n_frames*100:.1f}%)")
    
    return masks


def main():
    print("="*70)
    print("STFT-BASED MICROPHONE CROSS-TALK CANCELLATION TEST")
    print("="*70)
    
    # Load audio
    print(f"\nLoading {CALIB_FILE}...")
    X_calib = load_stereo(CALIB_FILE, sr=SR)
    print(f"  Shape: {X_calib.shape}")
    
    print(f"\nLoading {REC_FILE}...")
    X_rec = load_stereo(REC_FILE, sr=SR)
    print(f"  Shape: {X_rec.shape}")
    
    # Detect active frames
    print(f"\nDetecting solo segments in calibration recording...")
    active_masks = detect_active_frames_stft(
        X_calib, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    # Check if we found enough solo segments
    for i, mask in enumerate(active_masks):
        if np.sum(mask) == 0:
            print(f"\nWARNING: No solo segments found for Source {i+1}!")
            print("The calibration may fail or produce poor results.")
    
    # Run calibration and separation
    S_separated = calibrate_and_separate(
        X_calib, X_rec, active_masks,
        sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        lambda_reg=LAMBDA, max_iter=10,
        save_files=True
    )
    
    print("\n" + "="*70)
    print("DONE! Check the output files:")
    print("  - Recovered_Source_1.wav")
    print("  - Recovered_Source_2.wav")
    print("="*70)


if __name__ == "__main__":
    main()
