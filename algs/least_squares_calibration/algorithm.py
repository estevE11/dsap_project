import numpy as np
from algs.utils import save_audio

def estimate_noise_statistics(S_ref, X_mic, H_est):
    """
    Estimates noise covariance from the residual error.
    Assumes noise model: X = H*S + noise
    
    Args:
        S_ref (np.ndarray): Reference source matrix (sources x samples)
        X_mic (np.ndarray): Microphone observation matrix (mics x samples)
        H_est (np.ndarray): Estimated mixing matrix
        
    Returns:
        float: Estimated noise variance
    """
    # Compute residual: noise_est = X - H*S
    X_predicted = H_est @ S_ref
    residual = X_mic - X_predicted
    
    # Estimate noise variance (assuming i.i.d. Gaussian noise)
    noise_var = np.mean(residual ** 2)
    
    return noise_var

def estimate_mixing_matrix(S_ref, X_mic, lambda_reg=1e-4, H_prior=None):
    """
    Estimates the mixing matrix H using Regularized Least Squares (Ridge Regression).
    
    If H_prior is provided, solves: min ||X - HS||^2 + lambda ||H - H_prior||^2
    Solution: H = (X S^T + lambda H_prior) (S S^T + lambda I)^(-1)
    
    If H_prior is None, solves: min ||X - HS||^2 + lambda ||H||^2
    Solution: H = (X S^T) (S S^T + lambda I)^(-1)
    
    Args:
        S_ref (np.ndarray): Reference source matrix (sources x samples)
        X_mic (np.ndarray): Microphone observation matrix (mics x samples)
        lambda_reg (float): Regularization parameter
        H_prior (np.ndarray): Prior estimate of H (optional)
        
    Returns:
        np.ndarray: Estimated mixing matrix H
    """
    n_sources = S_ref.shape[0]
    
    # Regularized least squares: add lambda*I to S*S^T for stability
    gram_matrix = S_ref @ S_ref.T
    regularized_gram = gram_matrix + lambda_reg * np.eye(n_sources)
    
    # Numerator: X * S^T (+ lambda * H_prior if available)
    numerator = X_mic @ S_ref.T
    
    if H_prior is not None:
        numerator = numerator + lambda_reg * H_prior
    
    # H_est = Numerator * (S*S^T + lambda*I)^(-1)
    H_estimated = numerator @ np.linalg.inv(regularized_gram)
    
    return H_estimated

def iterative_refinement(S_init, X_mic, H_init, max_iter=5, tol=1e-5, lambda_reg=1e-4):
    """
    Performs iterative refinement using alternating least squares.
    Inspired by the MLE approach in Das et al. (2021), this alternates between:
    1. Estimating sources S given current H
    2. Re-estimating H given updated S
    
    Args:
        S_init (np.ndarray): Initial source estimate (sources x samples)
        X_mic (np.ndarray): Microphone observations (mics x samples)
        H_init (np.ndarray): Initial mixing matrix estimate
        max_iter (int): Maximum number of iterations
        tol (float): Convergence tolerance
        lambda_reg (float): Regularization parameter
        
    Returns:
        tuple: (refined_H, refined_S, converged, final_cost)
    """
    H_current = H_init.copy()
    S_current = S_init.copy()
    n_mics, n_sources = H_current.shape
    
    prev_cost = np.inf
    
    for iteration in range(max_iter):
        # Step 1: Update S given H using least squares: S = (H^T*H + lambda*I)^(-1) * H^T * X
        HTH = H_current.T @ H_current
        regularized_HTH = HTH + lambda_reg * np.eye(n_sources)
        S_current = np.linalg.inv(regularized_HTH) @ H_current.T @ X_mic
        
        # Step 2: Update H given S using regularized least squares with prior
        # We use the initial H (from calibration) as the prior to prevent drifting
        H_current = estimate_mixing_matrix(S_current, X_mic, lambda_reg, H_prior=H_init)
        
        # Compute cost (reconstruction error)
        residual = X_mic - H_current @ S_current
        cost = np.mean(residual ** 2)
        
        # Check convergence
        cost_change = abs(prev_cost - cost)
        if cost_change < tol:
            return H_current, S_current, True, cost
            
        prev_cost = cost
    
    # Max iterations reached
    return H_current, S_current, False, cost

def validate_calibration(H_est, real_H, noise_var):
    """
    Validates the calibration quality with statistical metrics.
    
    Args:
        H_est (np.ndarray): Estimated mixing matrix
        real_H (np.ndarray): Ground truth mixing matrix (if available)
        noise_var (float): Estimated noise variance
        
    Returns:
        dict: Dictionary containing validation metrics
    """
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((real_H - H_est)**2)
    metrics['mse'] = mse
    
    # Normalized Mean Squared Error
    nmse = mse / np.mean(real_H ** 2)
    metrics['nmse'] = nmse
    
    # Relative error per element
    rel_error = np.abs(real_H - H_est) / (np.abs(real_H) + 1e-10)
    metrics['mean_rel_error'] = np.mean(rel_error)
    metrics['max_rel_error'] = np.max(rel_error)
    
    # Condition number (matrix invertibility indicator)
    metrics['condition_number'] = np.linalg.cond(H_est)
    
    # Signal-to-noise ratio estimate
    signal_power = np.mean(H_est ** 2)
    metrics['snr_db'] = 10 * np.log10(signal_power / (noise_var + 1e-10))
    
    return metrics

def separate_sources(X_mic, H_est):
    """
    Separates sources by inverting the estimated mixing matrix.
    s_hat = inv(H_est) * x
    
    Args:
        X_mic (np.ndarray): Microphone observation matrix (mics x samples)
        H_est (np.ndarray): Estimated mixing matrix
        
    Returns:
        np.ndarray: Recovered source matrix
    """
    try:
        H_inverse = np.linalg.inv(H_est)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular. The bleed is too strong to separate!")
        
    S_recovered = H_inverse @ X_mic
    return S_recovered

def perform_calibration(S, X, real_H, calibration_seconds, sr=44100, 
                       use_iterative=True, lambda_reg=1e-4, max_iter=5):
    """
    Performs enhanced calibration with regularization and iterative refinement.
    Based on Das et al. (2021) approach for microphone cross-talk cancellation.
    
    Args:
        S (np.ndarray): Source signals (sources x samples)
        X (np.ndarray): Microphone observations (mics x samples)
        real_H (np.ndarray): Ground truth mixing matrix (for validation)
        calibration_seconds (float): Duration of calibration period in seconds
        sr (int): Sample rate
        use_iterative (bool): Whether to use iterative refinement
        lambda_reg (float): Regularization parameter
        max_iter (int): Maximum iterations for refinement
        
    Returns:
        np.ndarray: Estimated mixing matrix H
    """
    print(f"\n[Step 2] Enhanced Calibration (Learning H from first {calibration_seconds}s)...")
    print(f"  - Using regularized least squares (lambda={lambda_reg})")
    if use_iterative:
        print(f"  - Iterative refinement enabled (max_iter={max_iter})")
    
    calib_samples = int(calibration_seconds * sr)
    
    # Slice the data for calibration
    S_calib = S[:, :calib_samples] 
    X_calib = X[:, :calib_samples] 
    
    # Initial estimate with regularized least squares
    H_estimated = estimate_mixing_matrix(S_calib, X_calib, lambda_reg)
    
    # Estimate noise statistics
    noise_var = estimate_noise_statistics(S_calib, X_calib, H_estimated)
    print(f"  - Estimated noise variance: {noise_var:.6f}")
    
    # Iterative refinement (inspired by MLE alternating optimization)
    if use_iterative:
        print(f"  - Performing iterative refinement...")
        H_refined, S_refined, converged, final_cost = iterative_refinement(
            S_calib, X_calib, H_estimated, max_iter, lambda_reg=lambda_reg
        )
        
        if converged:
            print(f"  - Converged after refinement (cost: {final_cost:.6f})")
        else:
            print(f"  - Max iterations reached (final cost: {final_cost:.6f})")
        
        H_estimated = H_refined
    
    print(f"\nEstimated H:\n{H_estimated}")
    
    # Validate calibration
    metrics = validate_calibration(H_estimated, real_H, noise_var)
    
    print(f"\n--- Calibration Validation Metrics ---")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Normalized MSE: {metrics['nmse']:.6f}")
    print(f"  Mean Relative Error: {metrics['mean_rel_error']:.4f} ({metrics['mean_rel_error']*100:.2f}%)")
    print(f"  Max Relative Error: {metrics['max_rel_error']:.4f} ({metrics['max_rel_error']*100:.2f}%)")
    print(f"  Condition Number: {metrics['condition_number']:.2f}")
    print(f"  Estimated SNR: {metrics['snr_db']:.2f} dB")
    
    # Success criteria
    if metrics['mse'] < 0.001:
        print("\n✓ SUCCESS: The algorithm learned the leakage accurately.")
    elif metrics['mse'] < 0.01:
        print("\n⚠ WARNING: Moderate calibration error detected.")
    else:
        print("\n✗ WARNING: High calibration error - separation may be poor.")
        
    return H_estimated

def perform_separation(X, H_est, save_files=False, sr=44100):
    """
    Performs the separation stage and optionally saves results.
    """
    print(f"\n[Step 3] Separating (Inverting H)...")
    
    try:
        S_recovered = separate_sources(X, H_est)
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
        
    if save_files:
        save_audio("Recovered_Violin.wav", S_recovered[0], sr)
        save_audio("Recovered_Cello.wav", S_recovered[1], sr)
        print("\n-> DONE! Saved 'Recovered_Violin.wav' and 'Recovered_Cello.wav'")
        
    return S_recovered
