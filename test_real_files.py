import numpy as np
import librosa
import soundfile as sf
from algs.least_squares_calibration.algorithm import perform_calibration, perform_separation

# 🎛️ CONTROL PANEL
CALIB_FILE = "test_calib.mp3"
REC_FILE = "test_rec.mp3"
SR = 44100

def load_audio_stereo(file_path, sr=44100):
    """Load stereo audio file."""
    y, _ = librosa.load(file_path, sr=sr, mono=False)
    return y

def detect_active_segments(X, sr=44100, frame_length=2048, hop_length=512, threshold_db=-40):
    """
    Detects segments where one source is significantly more active than the other.
    Returns a mask for each source.
    """
    # Calculate energy per channel
    energy = np.array([
        librosa.feature.rms(y=X[0], frame_length=frame_length, hop_length=hop_length)[0],
        librosa.feature.rms(y=X[1], frame_length=frame_length, hop_length=hop_length)[0]
    ])
    
    # Normalize energy
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    # Create masks
    # Source 1 dominant: Mic 1 energy > threshold AND Mic 1 > Mic 2 + margin
    margin = 10 # dB
    mask_s1 = (energy_db[0] > threshold_db) & (energy_db[0] > energy_db[1] + margin)
    
    # Source 2 dominant: Mic 2 energy > threshold AND Mic 2 > Mic 1 + margin
    mask_s2 = (energy_db[1] > threshold_db) & (energy_db[1] > energy_db[0] + margin)
    
    # Upsample masks to audio length
    mask_s1_full = np.repeat(mask_s1, hop_length)[:X.shape[1]]
    mask_s2_full = np.repeat(mask_s2, hop_length)[:X.shape[1]]
    
    # Handle length mismatch due to padding
    if len(mask_s1_full) < X.shape[1]:
        pad = X.shape[1] - len(mask_s1_full)
        mask_s1_full = np.pad(mask_s1_full, (0, pad))
        mask_s2_full = np.pad(mask_s2_full, (0, pad))
    elif len(mask_s1_full) > X.shape[1]:
        mask_s1_full = mask_s1_full[:X.shape[1]]
        mask_s2_full = mask_s2_full[:X.shape[1]]
        
    return mask_s1_full, mask_s2_full

def construct_calibration_sources(X_calib):
    """
    Constructs reference sources S from the calibration recording X using energy detection.
    """
    print("  - Detecting active segments...")
    mask_s1, mask_s2 = detect_active_segments(X_calib)
    
    s1_count = np.sum(mask_s1)
    s2_count = np.sum(mask_s2)
    print(f"  - Found {s1_count/44100:.2f}s of isolated Source 1")
    print(f"  - Found {s2_count/44100:.2f}s of isolated Source 2")
    
    if s1_count == 0 or s2_count == 0:
        print("WARNING: Could not find isolated segments for both sources!")
        print("Falling back to simple split (first half / second half)")
        n_samples = X_calib.shape[1]
        mid = n_samples // 2
        mask_s1 = np.zeros(n_samples, dtype=bool)
        mask_s2 = np.zeros(n_samples, dtype=bool)
        mask_s1[:mid] = True
        mask_s2[mid:] = True
        
    S_constructed = np.zeros_like(X_calib)
    
    # When Source 1 is dominant, we assume S1 = Mic1, S2 = 0
    S_constructed[0, mask_s1] = X_calib[0, mask_s1]
    
    # When Source 2 is dominant, we assume S2 = Mic2, S1 = 0
    S_constructed[1, mask_s2] = X_calib[1, mask_s2]
    
    return S_constructed

def main():
    print(f"--- Real File Calibration Test ---\n")
    print(f"Calibration File: {CALIB_FILE}")
    print(f"Recording File: {REC_FILE}")
    
    # 1. Load Calibration Data
    print(f"\nLoading {CALIB_FILE}...")
    try:
        X_calib = load_audio_stereo(CALIB_FILE, sr=SR)
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return

    print(f"  - Shape: {X_calib.shape}")
    
    # 2. Construct Reference Sources (S)
    # We assume the calibration track has isolated sources
    print("Constructing reference sources (assuming isolated sequence)...")
    S_calib = construct_calibration_sources(X_calib)
    
    # 3. Perform Calibration
    # We don't have a 'real_H' for validation, so we pass a dummy one or None
    # The algorithm expects real_H for validation metrics, let's pass Identity as placeholder
    dummy_H = np.eye(2) 
    
    # Note: perform_calibration expects (S, X, real_H, seconds)
    # But here we already have the full data.
    # We need to adapt or call estimate_mixing_matrix directly.
    # Let's call estimate_mixing_matrix directly to avoid the slicing logic in perform_calibration
    
    from algs.least_squares_calibration.algorithm import estimate_mixing_matrix, estimate_noise_statistics
    
    print("\nEstimating Mixing Matrix H...")
    # Initial estimate (Standard Ridge)
    H_est = estimate_mixing_matrix(S_calib, X_calib, lambda_reg=1e-4)
    print(f"Initial H:\n{H_est}")
    
    # Iterative Refinement (ML Estimator)
    from algs.least_squares_calibration.algorithm import iterative_refinement
    print("\nPerforming Iterative Refinement (ML Estimator)...")
    H_refined, _, converged, cost = iterative_refinement(S_calib, X_calib, H_est, max_iter=10, lambda_reg=1e-4)
    
    if converged:
        print(f"  - Converged (cost: {cost:.6f})")
    else:
        print(f"  - Max iterations reached (cost: {cost:.6f})")
        
    H_est = H_refined
    print(f"Refined H:\n{H_est}")
    
    # 4. Load Recording Data
    print(f"\nLoading {REC_FILE}...")
    try:
        X_rec = load_audio_stereo(REC_FILE, sr=SR)
    except Exception as e:
        print(f"Error loading recording file: {e}")
        return
        
    print(f"  - Shape: {X_rec.shape}")
    
    # 5. Separate
    print(f"\nSeparating {REC_FILE}...")
    perform_separation(X_rec, H_est, save_files=True)
    
    print("\nDone! Check 'Recovered_Violin.wav' and 'Recovered_Cello.wav'")

if __name__ == "__main__":
    main()
