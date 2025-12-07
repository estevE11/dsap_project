import numpy as np
import soundfile as sf
import os
import sys
import argparse
import librosa
import mir_eval
import csv
from datetime import datetime

# Add the project root to sys.path to allow imports from algs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyass.peass import PEASS_ObjectiveMeasure

from algs.least_squares_calibration.algorithm_stft import calibrate_and_separate

# Configuration - Must match generate_crosstalk_dataset.py settings
BASE_DATA_DIR = "/home/esteve/dev/dsap_project/run/custom"

# Fixed parameters (must match simulation - Das et al. 2021)
SOURCE_SOURCE_DISTANCE = 1.0  # 1m between sources (matching paper)

# Algorithm versioning - update ALGORITHM_VERSION when making changes to the algorithm
ALGORITHM_NAME = "stft"
ALGORITHM_VERSION = "v1"

# CSV Results file path
RESULTS_CSV_PATH = "/home/esteve/dev/dsap_project/run/results.csv"

def get_experiment_folder_name(source_mic_distance):
    """Generate folder name matching generate_crosstalk_dataset.py format."""
    return f"d_mic_{source_mic_distance}m_d_src_{SOURCE_SOURCE_DISTANCE}m"

def load_wav(data_dir, filename):
    filepath = os.path.join(data_dir, filename)
    audio, fs = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, fs

def compute_metrics(estimated, reference):
    """Computes basic audio quality metrics."""
    min_len = min(len(estimated), len(reference))
    est = estimated[:min_len]
    ref = reference[:min_len]
    
    # MSE
    mse = np.mean((est - ref)**2)
    
    # SNR
    noise_power = np.mean((est - ref)**2)
    signal_power = np.mean(ref**2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return mse, snr

def compute_bss_eval(estimated, reference):
    """Computes BSS Eval metrics (SDR, SIR, SAR, ISR)."""
    # mir_eval expects (n_sources, n_samples)
    # We wrap them in 2D arrays
    ref = reference[np.newaxis, :]
    est = estimated[np.newaxis, :]
    
    # bss_eval_sources returns (sdr, sir, sar, perm)
    # We use compute_permutation=False because we know the mapping
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref, est, compute_permutation=False)
    
    return sdr[0], sir[0], sar[0]

def save_results_to_csv(csv_path, results_dict):
    """
    Save results to a CSV file. Creates the file with headers if it doesn't exist.
    Checks for existing results with the same (alg, mic_d, src_d) and prompts user
    before overwriting.
    
    Returns True if results were saved, False if user declined to override.
    """
    columns = ['calc', 'alg', 'mic_d', 'src_d', 'ops', 'tps', 'ips', 'aps', 
               'room_config', 'sdr', 'sir', 'sar', 'timestamp']
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    file_exists = os.path.exists(csv_path)
    existing_rows = []
    duplicate_found = False
    duplicate_idx = -1
    
    # Check for existing results with same config
    if file_exists:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for idx, row in enumerate(reader):
                existing_rows.append(row)
                # Check for duplicate: same alg, mic_d, src_d
                if (row['alg'] == results_dict['alg'] and 
                    float(row['mic_d']) == float(results_dict['mic_d']) and
                    float(row['src_d']) == float(results_dict['src_d'])):
                    duplicate_found = True
                    duplicate_idx = idx
    
    if duplicate_found:
        print(f"\n[CSV] Found existing result for config:")
        print(f"      alg={results_dict['alg']}, mic_d={results_dict['mic_d']}m, src_d={results_dict['src_d']}m")
        print(f"      Existing: OPS={existing_rows[duplicate_idx]['ops']}, TPS={existing_rows[duplicate_idx]['tps']}")
        print(f"      New:      OPS={results_dict['ops']}, TPS={results_dict['tps']}")
        
        response = input("Do you want to override the existing result? [y/N]: ").strip().lower()
        if response != 'y':
            print("[CSV] Keeping existing result, skipping save.")
            return False
        
        # Override: replace the row
        existing_rows[duplicate_idx] = results_dict
        
        # Rewrite entire file
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
        
        print(f"\n[CSV] Result overridden in {csv_path}")
        return True
    
    # No duplicate - append new row
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results_dict)
    
    print(f"\n[CSV] Results saved to {csv_path}")
    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run calibration and separation')
    parser.add_argument('--dist-mic', type=float, default=0.1,
                        help='Source-to-microphone distance in meters (default: 0.1)')
    args = parser.parse_args()
    
    source_mic_distance = args.dist_mic
    
    # Compute data directory based on distance
    experiment_folder = get_experiment_folder_name(source_mic_distance)
    data_dir = os.path.join(BASE_DATA_DIR, experiment_folder)
    output_dir = os.path.join(data_dir, "output")
    
    print(f"=" * 60)
    print(f"Running calibration for experiment: {experiment_folder}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"=" * 60)
    
    print("\nLoading calibration files...")
    # Load Calibration Files
    v_src_mic_v, fs = load_wav(data_dir, "calib_violin_source_mic_violin.wav")
    v_src_mic_c, _ = load_wav(data_dir, "calib_violin_source_mic_cello.wav")
    
    c_src_mic_c, _ = load_wav(data_dir, "calib_cello_source_mic_cello.wav")
    c_src_mic_v, _ = load_wav(data_dir, "calib_cello_source_mic_violin.wav")
    
    # Ensure all have same length
    min_len = min(len(v_src_mic_v), len(v_src_mic_c), len(c_src_mic_c), len(c_src_mic_v))
    v_src_mic_v = v_src_mic_v[:min_len]
    v_src_mic_c = v_src_mic_c[:min_len]
    c_src_mic_c = c_src_mic_c[:min_len]
    c_src_mic_v = c_src_mic_v[:min_len]
    
    # Construct Calibration Matrices (Concatenated)
    # X_calib: [Mic_Violin, Mic_Cello]
    # Part 1: Violin Solo
    X_part1 = np.vstack([v_src_mic_v, v_src_mic_c])
    # Part 2: Cello Solo
    X_part2 = np.vstack([c_src_mic_v, c_src_mic_c])
    
    X_calib = np.hstack([X_part1, X_part2])
    
    # Create Active Masks (in STFT frames)
    # We need to calculate how many frames corresponds to the split point
    n_fft = 2048
    hop_length = 512
    
    n_samples_part1 = X_part1.shape[1]
    n_samples_total = X_calib.shape[1]
    
    # Calculate number of frames
    # librosa.stft produces 1 + n_samples // hop_length frames (roughly, depends on padding)
    # Let's compute STFT of a dummy signal to get exact frame count
    dummy_stft = librosa.stft(np.zeros(n_samples_total), n_fft=n_fft, hop_length=hop_length)
    n_frames_total = dummy_stft.shape[1]
    
    # Frame index for split
    split_frame = int(n_samples_part1 / hop_length) 
    # Adjust for centering if needed, but rough split is usually fine for masks
    
    mask_v = np.zeros(n_frames_total, dtype=bool)
    mask_v[:split_frame] = True # Violin active in first part
    
    mask_c = np.zeros(n_frames_total, dtype=bool)
    mask_c[split_frame:] = True # Cello active in second part
    
    active_masks = [mask_v, mask_c]
    
    print("\nLoading test files...")
    # Load Test Files
    test_mic_v, _ = load_wav(data_dir, "test_mix_mic_violin.wav")
    test_mic_c, _ = load_wav(data_dir, "test_mix_mic_cello.wav")
    
    # Load Ground Truth
    clean_mic_v, _ = load_wav(data_dir, "test_clean_mic_violin.wav")
    
    # Truncate
    min_test_len = min(len(test_mic_v), len(test_mic_c), len(clean_mic_v))
    test_mic_v = test_mic_v[:min_test_len]
    test_mic_c = test_mic_c[:min_test_len]
    clean_mic_v = clean_mic_v[:min_test_len]
    
    X_test = np.vstack([test_mic_v, test_mic_c])
    
    # Run Calibration and Separation (STFT)
    S_recovered = calibrate_and_separate(
        X_calib, 
        X_test, 
        active_masks, 
        sr=fs, 
        n_fft=n_fft, 
        hop_length=hop_length,
        lambda_reg=0.01,
        save_files=False
    )
    
    if S_recovered is not None:
        recovered_violin = S_recovered[0]
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Outputs
        out_v_path = os.path.join(output_dir, "recovered_violin.wav")
        sf.write(out_v_path, recovered_violin, fs)
        
        print(f"\nSaved recovered violin to {out_v_path}")
        
        # Compare with Ground Truth
        print("\n--- Verification Metrics (Recovered vs Ground Truth) ---")
        mse, snr = compute_metrics(recovered_violin, clean_mic_v)
        print(f"MSE: {mse:.6f}")
        print(f"SNR: {snr:.2f} dB")
        
        # Also compare with the raw mix
        mse_raw, snr_raw = compute_metrics(test_mic_v, clean_mic_v)
        print(f"Original Mix SNR: {snr_raw:.2f} dB")
        print(f"Improvement: {snr - snr_raw:.2f} dB")
        
        # Compute BSS Eval Metrics
        print("\n--- BSS Eval Metrics (Standard Industry Metrics) ---")
        sdr, sir, sar = compute_bss_eval(recovered_violin, clean_mic_v)
        print(f"SDR (Source to Distortion Ratio): {sdr:.2f} dB")
        print(f"SIR (Source to Interference Ratio): {sir:.2f} dB")
        print(f"SAR (Sources to Artifacts Ratio):   {sar:.2f} dB")
        
        # Calculate for Original Mix for comparison
        sdr_raw, sir_raw, sar_raw = compute_bss_eval(test_mic_v[:len(clean_mic_v)], clean_mic_v)
        print(f"\nOriginal Mix SDR: {sdr_raw:.2f} dB")
        print(f"SDR Improvement: {sdr - sdr_raw:.2f} dB")
        
        # Save 30s metrics files
        print("\n[Metrics] Saving 30s truncated files for evaluation...")
        limit_samples = 30 * fs
        
        # Truncate
        recovered_violin_30s = recovered_violin[:limit_samples]
        clean_mic_v_30s = clean_mic_v[:limit_samples]
        
        # Save Outputs (Separated)
        out_v_30s_path = os.path.join(output_dir, "recovered_violin_30s.wav")
        sf.write(out_v_30s_path, recovered_violin_30s, fs)
        
        # Save Targets (Clean Reference)
        target_v_30s_path = os.path.join(output_dir, "target_violin_30s.wav")
        sf.write(target_v_30s_path, clean_mic_v_30s, fs)
        
        print(f"Saved {out_v_30s_path}")
        print(f"Saved {target_v_30s_path}")

        # Save 1s metrics files
        print("\n[Metrics] Saving 1s truncated files for evaluation...")
        limit_samples_1s = 1 * fs
        
        # Truncate
        recovered_violin_1s = recovered_violin[:limit_samples_1s]
        clean_mic_v_1s = clean_mic_v[:limit_samples_1s]
        
        # Save Outputs (Separated)
        out_v_1s_path = os.path.join(output_dir, "recovered_violin_1s.wav")
        sf.write(out_v_1s_path, recovered_violin_1s, fs)
        
        # Save Targets (Clean Reference)
        target_v_1s_path = os.path.join(output_dir, "target_violin_1s.wav")
        sf.write(target_v_1s_path, clean_mic_v_1s, fs)
        
        print(f"Saved {out_v_1s_path}")
        print(f"Saved {target_v_1s_path}")

        # Compute PEASS metrics
        print("\n--- PEASS Metrics ---")
        peass_options = {
            'destDir': output_dir,
        }
        try:
            peass_result = PEASS_ObjectiveMeasure(
                originalFiles=[target_v_1s_path],
                estimateFile=out_v_1s_path,
                options=peass_options
            )
            print(f"OPS (Overall Perceptual Score):    {peass_result['OPS']:.2f}")
            print(f"TPS (Target Perceptual Score):     {peass_result['TPS']:.2f}")
            print(f"IPS (Interference Perceptual Score): {peass_result['IPS']:.2f}")
            print(f"APS (Artifacts Perceptual Score):  {peass_result['APS']:.2f}")
            
            # Save results to CSV
            results_dict = {
                'calc': 1,  # Our calculated results
                'alg': f"{ALGORITHM_NAME}_{ALGORITHM_VERSION}",
                'mic_d': source_mic_distance,
                'src_d': SOURCE_SOURCE_DISTANCE,
                'ops': round(peass_result['OPS'], 2),
                'tps': round(peass_result['TPS'], 2),
                'ips': round(peass_result['IPS'], 2),
                'aps': round(peass_result['APS'], 2),
                'room_config': experiment_folder,
                'sdr': round(sdr, 2),
                'sir': round(sir, 2),
                'sar': round(sar, 2),
                'timestamp': datetime.now().isoformat()
            }
            save_results_to_csv(RESULTS_CSV_PATH, results_dict)
            
        except Exception as e:
            print(f"Error computing PEASS metrics: {e}")

if __name__ == "__main__":
    main()
