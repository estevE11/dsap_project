import numpy as np
from algs.utils import load_and_prepare_sources
from algs.wiener_filter.simulation import simulate_mixing
from algs.wiener_filter.algorithm import perform_separation

# 🎛️ CONTROL PANEL
INST_1 = "Violin_1.flac"
INST_2 = "Cello.flac"
REAL_H = np.array([[1.0, 0.4], [0.3, 1.0]])
DATA_PATH = "/Users/esteve/dev/dsap_project/My_Ensemble_Dataset/BBCSO_Ensembles/MiseroPargoletto/Close"

def main():
    print(f"--- Wiener Filter Simulation Lab (Blind) ---")
    
    # 1. Load Sources
    S, _ = load_and_prepare_sources(DATA_PATH, INST_1, INST_2)
    
    # 2. Simulate Leakage
    X = simulate_mixing(S, REAL_H, save_files=True)
    
    # 3. Separate (Blind Wiener / Interference Canceller)
    # No calibration needed!
    perform_separation(X, save_files=True)

if __name__ == "__main__":
    main()
