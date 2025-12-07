import os
import numpy as np
import soundfile as sf
from ensemble_loader import EnsembleDataLoader

# =============================================================================
# 🎵 EnsembleLoader Tutorial
# =============================================================================
# This script demonstrates how to use the EnsembleDataLoader class to:
# 1. Initialize the loader
# 2. List available songs
# 3. Load a specific song
# 4. Access the stereo mixture ("Mics")
# 5. Access individual instrument tracks ("Sources")
# =============================================================================

# 1. Initialize the Loader
# ------------------------
# Point it to the root directory containing your song folders.
# Example: "My_Ensemble_Dataset/BBCSO_Ensembles"
DATASET_ROOT = "My_Ensemble_Dataset/BBCSO_Ensembles"
loader = EnsembleDataLoader(DATASET_ROOT, sample_rate=44100)

print(f"Initialized loader for: {DATASET_ROOT}")

# 2. List Available Songs
# -----------------------
songs = loader.get_songs()
print(f"\nAvailable Songs: {songs}")

if not songs:
    print("❌ No songs found! Check your DATASET_ROOT.")
    exit()

# 3. Load a Song
# --------------
# Let's load the first song found.
song_name = songs[0]
print(f"\nLoading song: '{song_name}'...")

# This returns a dictionary with 'mixture', 'sources', and 'sample_rate'
data = loader.load_song(song_name)

# 4. Access the Mixture (The "Mics")
# ----------------------------------
# The 'mixture' is a stereo signal (2 channels x Time).
# You can think of these as your two main microphones (Left and Right).
mixture = data['mixture']
sr = data['sample_rate']

print(f"\n[Mixture Info]")
print(f"Shape: {mixture.shape} (Channels x Samples)")
print(f"Duration: {mixture.shape[1] / sr:.2f} seconds")

# Accessing Left and Right channels separately
mic_left = mixture[0, :]  # First row
mic_right = mixture[1, :] # Second row

print(f"Left Channel (Mic 1) samples: {mic_left.shape[0]}")
print(f"Right Channel (Mic 2) samples: {mic_right.shape[0]}")

# (Optional) Save the mix to hear it
# sf.write("tutorial_mix.wav", mixture.T, sr) # Transpose for soundfile (Time x Channels)

# 5. Access Individual Tracks (The "Sources")
# -------------------------------------------
# The 'sources' is a dictionary where keys are instrument names and values are audio arrays.
sources = data['sources']

print(f"\n[Sources Info]")
print(f"Instruments found: {list(sources.keys())}")

# Example: Accessing the Cello track
if 'Cello' in sources:
    cello_audio = sources['Cello']
    print(f"Cello audio shape: {cello_audio.shape}")
    
    # Note: Sources might be mono (1 x N) or stereo (2 x N) depending on the dataset.
    # If you need to process it as mono, you can average the channels:
    if cello_audio.shape[0] == 2:
        cello_mono = np.mean(cello_audio, axis=0)
        print("Converted Cello to mono.")
    else:
        cello_mono = cello_audio[0]

    # (Optional) Save the cello track
    # sf.write("tutorial_cello.wav", cello_mono, sr)

# Example: Iterating through all tracks
print("\nAll tracks details:")
for name, audio in sources.items():
    print(f" - {name}: {audio.shape} (Duration: {audio.shape[1]/sr:.2f}s)")

print("\n✅ Tutorial Complete!")
