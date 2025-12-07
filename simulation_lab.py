import numpy as np
from algs.utils import load_and_prepare_sources
from algs.least_squares_calibration.simulation import simulate_mixing
from algs.least_squares_calibration.algorithm import perform_calibration, perform_separation

# 🎛️ CONTROL PANEL
INST_1 = "Violin_1.flac"
INST_2 = "Cello.flac"
REAL_H = np.array([[1.0, 0.4], [0.3, 1.0]])
CALIBRATION_SECONDS = 5 
DATA_PATH = "/Users/esteve/dev/dsap_project/My_Ensemble_Dataset/BBCSO_Ensembles/MiseroPargoletto/Close"

def main():
    print(f"--- Algorithm 3 Simulation Lab ---")
    
    # 1. Load Sources
    S, _ = load_and_prepare_sources(DATA_PATH, INST_1, INST_2)
    
    # 2. Simulate Leakage
    X = simulate_mixing(S, REAL_H, save_files=True)
    
    # 3. Calibrate
    H_est = perform_calibration(S, X, REAL_H, CALIBRATION_SECONDS)
    
    # 4. Separate
    perform_separation(X, H_est, save_files=True)

if __name__ == "__main__":
    main()