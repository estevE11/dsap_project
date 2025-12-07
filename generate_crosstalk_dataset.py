import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import os
import argparse

# Configuration
BASE_OUTPUT_DIR = "/home/esteve/dev/dsap_project/run/custom"
INPUT_FILES = {
    "violin": "/home/esteve/dev/dsap_project/My_Ensemble_Dataset/BBCSO_Ensembles/MiseroPargoletto/Close/Violin_1.flac",
    "cello": "/home/esteve/dev/dsap_project/My_Ensemble_Dataset/BBCSO_Ensembles/MiseroPargoletto/Close/Cello.flac"
}
MIN_DURATION = 20  # seconds
CALIB_DURATION = 10 # seconds

# =============================================================================
# Room Configuration - Matching Das et al. (2021) AES Paper
# =============================================================================
# Paper specifications:
# - Room: 5m x 5m simulation environment
# - Source-mic distances tested: 0.1m to 0.5m
# - Near-anechoic conditions (very low reverberation)
# - Two sound sources with close microphones
# - Best results at d = 0.1m with MLE hyp = 0 or 1

# Room Geometry - 5m x 5m x 3m (matching paper)
ROOM_DIM = [5, 5, 3]

# For anechoic simulation (like the paper), use max_order=0 (no reflections)
# This is more reliable than trying to achieve very low RT60 via absorption
ANECHOIC_MODE = True
if ANECHOIC_MODE:
    ABSORPTION = 1.0  # Full absorption (not used when max_order=0)
    MAX_ORDER = 0     # No reflections = anechoic
else:
    RT60 = 0.3  # If you want reverberant room, use this
    ABSORPTION, MAX_ORDER = pra.inverse_sabine(RT60, ROOM_DIM)

# Fixed parameters
SOURCE_SOURCE_DISTANCE = 1.0  # 1m between sources (typical close ensemble)
ROOM_CENTER_X = ROOM_DIM[0] / 2.0  # 2.5m
ROOM_CENTER_Y = ROOM_DIM[1] / 2.0  # 2.5m
SOURCE_HEIGHT = 1.2  # Typical instrument height

def get_experiment_folder_name(source_mic_distance):
    """Generate a folder name that encodes the key simulation parameters."""
    return f"d_mic_{source_mic_distance}m_d_src_{SOURCE_SOURCE_DISTANCE}m"

def compute_positions(source_mic_distance):
    """Compute source and mic positions based on configuration."""
    pos_src_violin = [ROOM_CENTER_X - SOURCE_SOURCE_DISTANCE / 2, ROOM_CENTER_Y, SOURCE_HEIGHT]
    pos_src_cello = [ROOM_CENTER_X + SOURCE_SOURCE_DISTANCE / 2, ROOM_CENTER_Y, SOURCE_HEIGHT]
    pos_mic_violin = [pos_src_violin[0], pos_src_violin[1] - source_mic_distance, SOURCE_HEIGHT]
    pos_mic_cello = [pos_src_cello[0], pos_src_cello[1] - source_mic_distance, SOURCE_HEIGHT]
    return pos_src_violin, pos_src_cello, pos_mic_violin, pos_mic_cello

def ensure_inputs_exist():
    """Generates dummy files if they don't exist for testing purposes."""
    fs = 16000
    duration = 30
    t = np.linspace(0, duration, int(fs * duration))
    
    for name, filename in INPUT_FILES.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found. Generating dummy sine wave.")
            freq = 440 if name == "violin" else 220
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            sf.write(filename, audio, fs)

def load_audio(filename):
    audio, fs = sf.read(filename)
    if audio.ndim > 1:
        audio = audio[:, 0] # Take first channel if stereo
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return fs, audio

def create_room(fs, pos_mic_violin, pos_mic_cello):
    """Creates the room object with defined geometry."""
    room = pra.ShoeBox(
        ROOM_DIM,
        fs=fs,
        absorption=ABSORPTION,
        max_order=10
    )
    # Add Microphones
    # We add them as a single array to get all channels at once
    # Channel 0: Violin Mic, Channel 1: Cello Mic
    mics = np.c_[pos_mic_violin, pos_mic_cello]
    room.add_microphone_array(mics)
    return room

def run_simulation(fs, source_signals, pos_mic_violin, pos_mic_cello):
    """
    Runs the simulation.
    source_signals: list of (position, signal) tuples. 
                    If signal is None, that source is silent/not added.
    """
    room = create_room(fs, pos_mic_violin, pos_mic_cello)
    
    for pos, signal in source_signals:
        if signal is not None:
            room.add_source(pos, signal=signal)
        else:
            # Add a silent source or just don't add it. 
            # Not adding it is safer to avoid any computation, 
            # but to be physically identical "presence" wise, pyroomacoustics 
            # sources are point sources without scattering, so omitting is fine.
            pass
            
    room.simulate()
    return room.mic_array.signals

def save_wav(output_dir, filename, fs, data):
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, data, fs) 
    print(f"Saved {filename}: Duration {len(data)/fs:.2f}s")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate crosstalk simulation dataset')
    parser.add_argument('--dist-mic', type=float, default=0.1,
                        help='Source-to-microphone distance in meters (default: 0.1)')
    args = parser.parse_args()
    
    source_mic_distance = args.dist_mic
    
    # Compute positions and output directory
    pos_src_violin, pos_src_cello, pos_mic_violin, pos_mic_cello = compute_positions(source_mic_distance)
    output_dir = os.path.join(BASE_OUTPUT_DIR, get_experiment_folder_name(source_mic_distance))
    
    os.makedirs(output_dir, exist_ok=True)
    ensure_inputs_exist()
    
    # Print configuration summary
    print("=" * 60)
    print("Room Simulation Configuration (Das et al. 2021)")
    print("=" * 60)
    print(f"  Room dimensions:       {ROOM_DIM[0]}m x {ROOM_DIM[1]}m x {ROOM_DIM[2]}m")
    print(f"  Mode:                  {'Anechoic (max_order=0)' if ANECHOIC_MODE else f'RT60={RT60}s'}")
    print(f"  Source-Mic distance:   {source_mic_distance}m")
    print(f"  Source-Source distance: {SOURCE_SOURCE_DISTANCE}m")
    print(f"  Violin position:       {pos_src_violin}")
    print(f"  Cello position:        {pos_src_cello}")
    print(f"  Violin mic position:   {pos_mic_violin}")
    print(f"  Cello mic position:    {pos_mic_cello}")
    print(f"  Output directory:      {output_dir}")
    print("=" * 60)
    
    
    # 1. Load Audio
    fs_v, audio_v = load_audio(INPUT_FILES["violin"])
    fs_c, audio_c = load_audio(INPUT_FILES["cello"])
    
    if fs_v != fs_c:
        raise ValueError("Sampling rates must match!")
    fs = fs_v
    
    # Ensure minimum length
    min_len = int(MIN_DURATION * fs)
    if len(audio_v) < min_len or len(audio_c) < min_len:
        print("Warning: Input files are shorter than 20s. Looping to extend.")
        # Simple loop for testing
        while len(audio_v) < min_len: audio_v = np.concatenate((audio_v, audio_v))
        while len(audio_c) < min_len: audio_c = np.concatenate((audio_c, audio_c))

    # 2. Split Audio
    split_idx = int(CALIB_DURATION * fs)
    
    # Calibration Segments (0 -> 10s)
    calib_v = audio_v[:split_idx]
    calib_c = audio_c[:split_idx]
    
    # Performance Segments (10s -> End)
    perf_v = audio_v[split_idx:]
    perf_c = audio_c[split_idx:]
    
    print(f"Calibration Length: {len(calib_v)/fs:.2f}s")
    print(f"Performance Length: {len(perf_v)/fs:.2f}s")

    # 3. Generate Calibration Files (Group B)
    print("\n--- Generating Calibration Files ---")
    
    # Simulation 1: Violin Solo
    # Sources: Violin=Active, Cello=Silent (Not added)
    sigs_v_solo = run_simulation(fs, [(pos_src_violin, calib_v)], pos_mic_violin, pos_mic_cello)
    # Channel 0 is Violin Mic, Channel 1 is Cello Mic
    save_wav(output_dir, "calib_violin_source_mic_violin.wav", fs, sigs_v_solo[0])
    save_wav(output_dir, "calib_violin_source_mic_cello.wav", fs, sigs_v_solo[1])
    
    # Simulation 2: Cello Solo
    # Sources: Violin=Silent, Cello=Active
    sigs_c_solo = run_simulation(fs, [(pos_src_cello, calib_c)], pos_mic_violin, pos_mic_cello)
    save_wav(output_dir, "calib_cello_source_mic_violin.wav", fs, sigs_c_solo[0])
    save_wav(output_dir, "calib_cello_source_mic_cello.wav", fs, sigs_c_solo[1])

    # 4. Generate Performance Files (Group A)
    print("\n--- Generating Performance Files ---")
    
    # Simulation 3: Full Mix
    # Sources: Violin=Active, Cello=Active
    sigs_mix = run_simulation(fs, [(pos_src_violin, perf_v), (pos_src_cello, perf_c)], pos_mic_violin, pos_mic_cello)
    save_wav(output_dir, "test_mix_mic_violin.wav", fs, sigs_mix[0])
    save_wav(output_dir, "test_mix_mic_cello.wav", fs, sigs_mix[1])

    # Simulation 4: Clean Performance (Ground Truth)
    # Violin Solo (Performance Segment)
    sigs_v_perf = run_simulation(fs, [(pos_src_violin, perf_v)], pos_mic_violin, pos_mic_cello)
    save_wav(output_dir, "test_clean_mic_violin.wav", fs, sigs_v_perf[0])
    
    # Cello Solo (Performance Segment)
    sigs_c_perf = run_simulation(fs, [(pos_src_cello, perf_c)], pos_mic_violin, pos_mic_cello)
    save_wav(output_dir, "test_clean_mic_cello.wav", fs, sigs_c_perf[1])

if __name__ == "__main__":
    main()

