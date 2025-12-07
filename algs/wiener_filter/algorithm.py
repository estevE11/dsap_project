import numpy as np
from algs.utils import save_audio

def blind_interference_cancellation(X_mic):
    """
    Separates sources using a Blind Wiener Filter / Interference Canceller approach.
    It assumes that for each mic, the other mics serve as reference for the noise (bleed).
    
    For 2 channels:
    s1_hat = x1 - w12 * x2
    s2_hat = x2 - w21 * x1
    
    Weights w are calculated to minimize the energy of the output (MMSE).
    w12 = E[x1 * x2] / E[x2^2]
    
    This assumes:
    1. The bleed is "subtle" (signal leakage into reference is low).
    2. Sources are uncorrelated.
    3. Mixing is instantaneous (no delays).
    """
    num_mics, num_samples = X_mic.shape
    S_recovered = np.zeros_like(X_mic)
    
    # Calculate covariance matrix of the mixture
    # R_xx = (X * X.T) / N
    R_xx = (X_mic @ X_mic.T) / num_samples
    
    print(f"Mixture Covariance:\n{R_xx}")
    
    # For each channel i, treat all other channels j as noise references
    # For the simple 2-channel case:
    if num_mics == 2:
        # Clean Channel 1 (Violin) using Channel 2 (Cello) as noise reference
        # w12 = E[x1 x2] / E[x2^2]
        w12 = R_xx[0, 1] / R_xx[1, 1]
        S_recovered[0] = X_mic[0] - w12 * X_mic[1]
        print(f"Computed weight w12 (removing Mic 2 from Mic 1): {w12:.4f}")
        
        # Clean Channel 2 (Cello) using Channel 1 (Violin) as noise reference
        # w21 = E[x2 x1] / E[x1^2]
        w21 = R_xx[1, 0] / R_xx[0, 0]
        S_recovered[1] = X_mic[1] - w21 * X_mic[0]
        print(f"Computed weight w21 (removing Mic 1 from Mic 2): {w21:.4f}")
        
    else:
        # General case (not implemented for simplicity, but follows same logic)
        raise NotImplementedError("Blind cancellation currently implemented for 2 channels only.")
        
    return S_recovered

def perform_separation(X, save_files=False, sr=44100):
    """
    Performs the separation stage using Blind Wiener Filter and optionally saves results.
    """
    print(f"\n[Step 2] Separating (Blind Wiener / Interference Cancellation)...")
    
    try:
        S_recovered = blind_interference_cancellation(X)
    except Exception as e:
        print(f"ERROR: {e}")
        return None
        
    if save_files:
        save_audio("Recovered_Violin_Wiener.wav", S_recovered[0], sr)
        save_audio("Recovered_Cello_Wiener.wav", S_recovered[1], sr)
        print("\n-> DONE! Saved 'Recovered_Violin_Wiener.wav' and 'Recovered_Cello_Wiener.wav'")
        
    return S_recovered
