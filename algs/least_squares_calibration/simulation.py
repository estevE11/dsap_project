from algs.utils import save_audio
import numpy as np

def simulate_mixing(S, H, save_files=False, sr=44100, snr_db=None):
    """
    Simulates the leakage/mixing process.
    X = H @ S
    
    Args:
        S (np.ndarray): Source matrix (sources x samples)
        H (np.ndarray): Mixing matrix (mics x sources)
        save_files (bool): Whether to save the simulated microphone audio
        sr (int): Sample rate for saving audio
        
    Returns:
        np.ndarray: Microphone observation matrix X
    """
    print(f"\n[Step 1] Simulating Leakage using Real H:\n{H}")
    X = H @ S
    
    # Add Gaussian Noise
    if snr_db is not None:
        print(f"  - Adding Gaussian noise (SNR: {snr_db}dB)")
        # Calculate signal power per mic
        sig_power = np.mean(X**2, axis=1, keepdims=True)
        # Calculate noise power
        noise_power = sig_power / (10**(snr_db/10))
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), X.shape)
        X = X + noise
    
    if save_files:
        save_audio("Simulated_Mic_Violin.wav", X[0], sr)
        save_audio("Simulated_Mic_Cello.wav", X[1], sr)
        print("(Listen to this: it has bleed!)")
        
    return X
