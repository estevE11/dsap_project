import os
import glob
import numpy as np
import librosa
import soundfile as sf

class EnsembleDataLoader:
    def __init__(self, dataset_root, sample_rate=44100):
        """
        Initialize the EnsembleDataLoader.
        
        Args:
            dataset_root (str): Path to the root directory of the dataset (e.g., 'My_Ensemble_Dataset/BBCSO_Ensembles').
            sample_rate (int): Target sample rate for audio loading.
        """
        self.dataset_root = dataset_root
        self.sample_rate = sample_rate
        
    def get_songs(self):
        """
        Returns a list of available song names in the dataset.
        """
        if not os.path.exists(self.dataset_root):
            return []
            
        # List directories in the dataset root
        songs = [d for d in os.listdir(self.dataset_root) 
                 if os.path.isdir(os.path.join(self.dataset_root, d)) and not d.startswith('.')]
        return sorted(songs)

    def load_song(self, song_name):
        """
        Loads a specific song's mixture and ground truth sources.
        
        Args:
            song_name (str): Name of the song to load.
            
        Returns:
            dict: A dictionary containing:
                - 'mixture': (2, N) numpy array (stereo mixture).
                - 'sources': Dictionary mapping instrument names to (1, N) or (2, N) numpy arrays.
                - 'sample_rate': The sample rate used.
        """
        song_path = os.path.join(self.dataset_root, song_name)
        if not os.path.exists(song_path):
            raise ValueError(f"Song '{song_name}' not found in {self.dataset_root}")
            
        mix_path = os.path.join(song_path, 'Mix_1')
        close_path = os.path.join(song_path, 'Close')
        
        if not os.path.exists(mix_path):
             raise ValueError(f"Mix_1 folder not found for song '{song_name}'")
        if not os.path.exists(close_path):
             raise ValueError(f"Close folder not found for song '{song_name}'")

        # Load Sources (Ground Truth)
        sources = {}
        close_files = glob.glob(os.path.join(close_path, "*.flac"))
        
        # We need to determine the length of the audio first to initialize the mixture
        # We'll load the first source to get the length
        if not close_files:
            raise ValueError(f"No FLAC files found in Close folder for '{song_name}'")
            
        # Load all sources
        for file_path in close_files:
            instrument_name = os.path.splitext(os.path.basename(file_path))[0]
            # Load audio. librosa.load returns (y, sr). y is (N,) or (2, N)
            # We force mono=False to keep stereo if it is stereo, but usually these are mono or stereo.
            # The context says "Close/ filenames represent the clean, dry signal".
            # Let's assume they can be stereo or mono.
            y, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # Ensure shape is (channels, samples)
            if y.ndim == 1:
                y = y[np.newaxis, :] # Make it (1, N)
                
            sources[instrument_name] = y

        # Create Mixture
        # "Mix_1" contains individual instrument files that sum up to the mix?
        # Or does it contain the mix itself?
        # The context says: "Mix_1/ -> INPUT (X): The professional stereo mix."
        # BUT my exploration showed Mix_1 contains "Cello.flac", "Viola.flac", etc.
        # And they were NOT identical to Close files (different content).
        # So Mix_1 likely contains the "wet" or "panned" stems that make up the mix.
        # So we should sum them up.
        
        mix_files = glob.glob(os.path.join(mix_path, "*.flac"))
        if not mix_files:
             raise ValueError(f"No FLAC files found in Mix_1 folder for '{song_name}'")

        mixture = None
        
        for file_path in mix_files:
            y, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            if y.ndim == 1:
                y = y[np.newaxis, :]
            
            if mixture is None:
                mixture = y
            else:
                # Handle potential length mismatches (though unlikely in this dataset)
                min_len = min(mixture.shape[1], y.shape[1])
                mixture = mixture[:, :min_len] + y[:, :min_len]
                
        return {
            'mixture': mixture,
            'sources': sources,
            'sample_rate': self.sample_rate
        }

if __name__ == "__main__":
    # Simple test
    loader = EnsembleDataLoader("My_Ensemble_Dataset/BBCSO_Ensembles")
    songs = loader.get_songs()
    print(f"Available songs: {songs}")
    
    if songs:
        data = loader.load_song(songs[0])
        print(f"Loaded {songs[0]}")
        print(f"Mixture shape: {data['mixture'].shape}")
        print(f"Sources: {list(data['sources'].keys())}")
