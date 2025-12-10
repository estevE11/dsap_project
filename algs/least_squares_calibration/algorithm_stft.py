"""
Frequency-Domain Implementation of Microphone Cross-talk Cancellation
Based on Das et al. (2021) - Maximum Likelihood Estimation

This implementation operates in the STFT domain to properly handle 
room acoustics (delays, reverberation) as required by the paper.
"""

import numpy as np
import librosa
from algs.utils import save_audio


def compute_stft(signals, n_fft=2048, hop_length=512):
    """
    Compute STFT for multiple signals.
    
    Args:
        signals: (n_channels, n_samples) array
        n_fft: FFT window size
        hop_length: Hop size
        
    Returns:
        (n_channels, n_freqs, n_frames) complex array
    """
    n_channels = signals.shape[0]
    stfts = []
    
    for i in range(n_channels):
        S = librosa.stft(signals[i], n_fft=n_fft, hop_length=hop_length)
        stfts.append(S)
    
    return np.array(stfts)


def compute_istft(stfts, hop_length=512, length=None):
    """
    Compute ISTFT for multiple signals.
    
    Args:
        stfts: (n_channels, n_freqs, n_frames) complex array
        hop_length: Hop size
        length: Target length in samples
        
    Returns:
        (n_channels, n_samples) array
    """
    n_channels = stfts.shape[0]
    signals = []
    
    for i in range(n_channels):
        y = librosa.istft(stfts[i], hop_length=hop_length, length=length)
        signals.append(y)
    
    return np.array(signals)


def estimate_transfer_function_calibration(X_stft, active_masks, energy_threshold=0.01):
    """
    Estimate initial transfer function H̃ using spectral ratios from solo segments.
    
    Implements Equation 4 from the paper:
    H̃_nm(ω) = (1/N_sig) Σ X_n(τ,ω) / X_m(τ,ω)  for n ≠ m
    H̃_nn(ω) = 1.0  (diagonal forced to unity)
    
    Args:
        X_stft: (n_mics, n_freqs, n_frames) - Microphone STFT
        active_masks: List of (n_frames,) boolean masks for each source
        energy_threshold: Minimum energy to consider a frame valid
        
    Returns:
        H_tilde: (n_mics, n_mics, n_freqs) - Initial transfer function per frequency
    """
    n_mics, n_freqs, n_frames = X_stft.shape
    H_tilde = np.zeros((n_mics, n_mics, n_freqs), dtype=complex)
    
    # Set diagonal to 1.0 for all frequencies (CRITICAL)
    for f in range(n_freqs):
        H_tilde[:, :, f] = np.eye(n_mics, dtype=complex)
    
    # Compute off-diagonal elements from spectral ratios
    for m in range(n_mics):  # Source mic
        if m >= len(active_masks):
            continue
            
        mask = active_masks[m]
        if np.sum(mask) == 0:
            continue
        
        for n in range(n_mics):  # Distant mic
            if n == m:
                continue  # Skip diagonal
            
            # Compute spectral ratio per frequency bin
            for f in range(n_freqs):
                X_n = X_stft[n, f, mask]
                X_m = X_stft[m, f, mask]
                
                # Filter out low-energy frames
                energy = np.abs(X_m) ** 2
                valid = energy > energy_threshold * np.max(energy)
                
                if np.sum(valid) > 0:
                    # Average spectral ratio (complex division)
                    ratio = X_n[valid] / (X_m[valid] + 1e-10)
                    H_tilde[n, m, f] = np.mean(ratio)
    return H_tilde


def mle_optimization_per_bin(X_f, S_init_f, H_tilde_f, lambda_reg=1.0, max_iter=10, tol=1e-6):
    """
    Perform MLE optimization for a single frequency bin.
    
    Solves the alternating optimization problem (Equation 7-9):
    min_H,S  ||X - HS||² + (σ²_w/σ²_ν) * ||H - H̃||²
    
    Args:
        X_f: (n_mics, n_frames) - Observations at frequency f
        S_init_f: (n_sources, n_frames) - Initial source estimate
        H_tilde_f: (n_mics, n_sources) - Prior transfer function
        lambda_reg: σ²_w/σ²_ν regularization parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        H_opt: (n_mics, n_sources) - Optimized transfer function
        S_opt: (n_sources, n_frames) - Optimized sources
    """
    H_current = H_tilde_f.copy()
    S_current = S_init_f.copy()
    
    n_mics, n_sources = H_current.shape
    n_frames = X_f.shape[1]
    
    prev_cost = np.inf
    
    for iteration in range(max_iter):
        # Step 1: Update S given H
        # S = (H^H * H + λI)^(-1) * H^H * X
        HH_H = H_current.conj().T @ H_current  # Hermitian transpose!
        regularized = HH_H + lambda_reg * np.eye(n_sources)
        S_current = np.linalg.solve(regularized, H_current.conj().T @ X_f)
        
        # Step 2: Update H given S
        # Implements the solution from Equation 9:
        # H* = H̃ + σ²_ν/σ²_w * XS^H * (I + σ²_ν/σ²_w * SS^H)^(-1)
        # Rearranged: H = (XS^H + λH̃) * (SS^H + λI)^(-1)
        
        SS_H = S_current @ S_current.conj().T  # Hermitian!
        regularized_S = SS_H + lambda_reg * np.eye(n_sources)
        
        XS_H = X_f @ S_current.conj().T
        numerator = XS_H + lambda_reg * H_tilde_f
        
        H_current = numerator @ np.linalg.inv(regularized_S)
        
        # Compute cost
        residual = X_f - H_current @ S_current
        reconstruction_error = np.sum(np.abs(residual) ** 2)
        regularization_term = lambda_reg * np.sum(np.abs(H_current - H_tilde_f) ** 2)
        cost = reconstruction_error + regularization_term
        
        # Check convergence
        if abs(prev_cost - cost) < tol * abs(prev_cost):
            break
        
        prev_cost = cost
    
    return H_current, S_current


def perform_calibration_stft(X_stft, active_masks, lambda_reg=1.0, max_iter=10, n_fft=2048, hop_length=512):
    """
    Perform frequency-domain calibration.
    
    Args:
        X_stft: (n_mics, n_freqs, n_frames) - Microphone observations (STFT)
        active_masks: List of (n_frames,) boolean masks indicating when each source is solo
        lambda_reg: Regularization parameter
        max_iter: Maximum iterations for MLE
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        H_estimated: (n_mics, n_mics, n_freqs) - Estimated transfer function
    """
    n_mics, n_freqs, n_frames = X_stft.shape
    
    print(f"\n[Calibration] Estimating H̃ from solo segments...")
    H_tilde = estimate_transfer_function_calibration(X_stft, active_masks)
    
    print(f"[Calibration] Performing MLE optimization per frequency bin...")
    H_estimated = np.zeros_like(H_tilde)
    
    for f in range(n_freqs):
        if f % 100 == 0:
            print(f"  Processing bin {f}/{n_freqs}...")
        
        X_f = X_stft[:, f, :]  # (n_mics, n_frames)
        H_tilde_f = H_tilde[:, :, f]  # (n_mics, n_mics)
        
        # Initial source estimate: S ≈ inv(H̃) * X
        try:
            S_init_f = np.linalg.solve(H_tilde_f, X_f)
        except:
            S_init_f = np.linalg.lstsq(H_tilde_f, X_f, rcond=None)[0]
        
        # MLE optimization
        H_opt, _ = mle_optimization_per_bin(
            X_f, S_init_f, H_tilde_f,
            lambda_reg=lambda_reg,
            max_iter=max_iter
        )
        
        H_estimated[:, :, f] = H_opt
    
    print(f"[Calibration] Done.")
    return H_estimated


def perform_separation_stft(X_stft, H_estimated):
    """
    Perform source separation using estimated transfer function.
    
    Args:
        X_stft: (n_mics, n_freqs, n_frames) - Microphone observations
        H_estimated: (n_mics, n_mics, n_freqs) - Estimated transfer function
        
    Returns:
        S_estimated: (n_mics, n_freqs, n_frames) - Separated sources
    """
    n_mics, n_freqs, n_frames = X_stft.shape
    S_estimated = np.zeros_like(X_stft)
    
    print(f"\n[Separation] Inverting H per frequency bin...")
    
    for f in range(n_freqs):
        H_f = H_estimated[:, :, f]
        X_f = X_stft[:, f, :]
        
        # S = inv(H) * X
        try:
            S_estimated[:, f, :] = np.linalg.solve(H_f, X_f)
        except:
            S_estimated[:, f, :] = np.linalg.lstsq(H_f, X_f, rcond=None)[0]
    
    print(f"[Separation] Done.")
    return S_estimated


def calibrate_and_separate(X_calib, X_rec, active_masks, sr=44100, n_fft=2048, 
                           hop_length=512, lambda_reg=1.0, max_iter=10, save_files=True):
    """
    Full pipeline: Calibration + Separation.
    
    Args:
        X_calib: (n_mics, n_samples) - Calibration recording
        X_rec: (n_mics, n_samples) - Recording to separate
        active_masks: List of (n_frames,) boolean masks for solo segments
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        lambda_reg: Regularization parameter
        max_iter: Maximum MLE iterations
        save_files: Whether to save output
        
    Returns:
        S_separated: (n_mics, n_samples) - Separated sources in time domain
    """
    print("="*70)
    print("STFT-BASED MICROPHONE CROSS-TALK CANCELLATION")
    print("="*70)
    
    # 1. Compute STFT
    print("\n[Step 1] Computing STFT...")
    X_calib_stft = compute_stft(X_calib, n_fft=n_fft, hop_length=hop_length)
    X_rec_stft = compute_stft(X_rec, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Calibration
    print("\n[Step 2] Calibration...")
    H_estimated = perform_calibration_stft(
        X_calib_stft, active_masks,
        lambda_reg=lambda_reg, max_iter=max_iter,
        n_fft=n_fft, hop_length=hop_length
    )
    
    # 3. Separation
    print("\n[Step 3] Separation...")
    S_stft = perform_separation_stft(X_rec_stft, H_estimated)
    
    # 4. Inverse STFT
    print("\n[Step 4] Inverse STFT...")
    S_separated = compute_istft(S_stft, hop_length=hop_length, length=X_rec.shape[1])
    
    # 5. Save
    if save_files:
        print("\n[Step 5] Saving separated sources...")
        save_audio("Recovered_Source_1.wav", S_separated[0], sr)
        save_audio("Recovered_Source_2.wav", S_separated[1], sr)
        print("Done!")
    
    return S_separated
