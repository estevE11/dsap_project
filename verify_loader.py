import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from ensemble_loader import EnsembleDataLoader

def verify():
    print("Initializing EnsembleDataLoader...")
    loader = EnsembleDataLoader("My_Ensemble_Dataset/BBCSO_Ensembles")
    
    print("Getting songs...")
    songs = loader.get_songs()
    print(f"Found songs: {songs}")
    
    if not songs:
        print("No songs found. Verification failed (or no data downloaded).")
        return
        
    song_name = songs[0]
    print(f"Loading song: {song_name}")
    
    try:
        data = loader.load_song(song_name)
        mixture = data['mixture']
        sources = data['sources']
        sr = data['sample_rate']
        
        print(f"Sample Rate: {sr}")
        print(f"Mixture Shape: {mixture.shape}")
        print(f"Sources: {list(sources.keys())}")
        
        for name, audio in sources.items():
            print(f"  {name}: {audio.shape}")
            
        # Basic checks
        if mixture.ndim != 2:
            print("ERROR: Mixture should be 2D (channels, samples)")
        if mixture.shape[0] not in [1, 2]:
            print("ERROR: Mixture channels should be 1 or 2")
            
        print("Verification SUCCESS!")
        
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
