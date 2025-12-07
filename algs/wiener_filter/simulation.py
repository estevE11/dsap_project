from algs.utils import save_audio

def simulate_mixing(S, H, save_files=False, sr=44100):
    """
    Simulates the leakage/mixing process.
    X = H @ S
    """
    print(f"\n[Step 1] Simulating Leakage using Real H:\n{H}")
    X = H @ S
    
    if save_files:
        save_audio("Simulated_Mic_Violin.wav", X[0], sr)
        save_audio("Simulated_Mic_Cello.wav", X[1], sr)
        print("(Listen to this: it has bleed!)")
        
    return X
