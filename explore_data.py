import os
import glob
import librosa
import numpy as np

dataset_path = "My_Ensemble_Dataset/BBCSO_Ensembles/MiseroPargoletto"

print(f"Exploring {dataset_path}...")

# List all files
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav") or file.endswith(".flac"):
            # print(os.path.join(root, file))
            pass

# Check Mix_1 content
mix_1_path = os.path.join(dataset_path, "Mix_1")
if os.path.exists(mix_1_path):
    print("\nMix_1 files:")
    files = os.listdir(mix_1_path)
    for f in files:
        if f.endswith(".flac"):
            path = os.path.join(mix_1_path, f)
            y, sr = librosa.load(path, sr=None)
            print(f"  {f}: {librosa.get_duration(y=y, sr=sr):.2f}s, {y.shape} shape, {sr} Hz")

# Check Close content
close_path = os.path.join(dataset_path, "Close")
if os.path.exists(close_path):
    print("\nClose files:")
    files = os.listdir(close_path)
    for f in files:
        if f.endswith(".flac"):
            path = os.path.join(close_path, f)
            y, sr = librosa.load(path, sr=None)
            print(f"  {f}: {librosa.get_duration(y=y, sr=sr):.2f}s, {y.shape} shape, {sr} Hz")

# Check if Mix_1 files are identical to Close files
print("\nComparing Mix_1 and Close files...")
common_files = set(os.listdir(mix_1_path)) & set(os.listdir(close_path))
for f in common_files:
    if f.endswith(".flac"):
        p1 = os.path.join(mix_1_path, f)
        p2 = os.path.join(close_path, f)
        d1, sr1 = librosa.load(p1, sr=None)
        d2, sr2 = librosa.load(p2, sr=None)
        if np.array_equal(d1, d2):
            print(f"  {f} is IDENTICAL")
        else:
            print(f"  {f} is DIFFERENT")
            # Check correlation or something?
            diff = np.mean(np.abs(d1 - d2))
            print(f"    Mean abs diff: {diff}")

