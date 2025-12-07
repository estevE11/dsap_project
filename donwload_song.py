import os
import remotezip
import sys

# --- CONFIGURATION ---
# The direct link to the massive 58GB file
ZIP_URL = "https://zenodo.org/record/6519024/files/BBCSO_Ensembles.zip?download=1"

# The specific song you want (Change this to other song names if needed)
TARGET_SONG = "MiseroPargoletto"

# The folder on your computer where the music will be saved
OUTPUT_FOLDER = "My_Ensemble_Dataset"

def main():
    print(f"--- EnsembleSet Downloader ---")
    print(f"Target: {TARGET_SONG}")
    print(f"Source: Zenodo (Streaming without full download)")
    print(f"Saving to: {os.path.abspath(OUTPUT_FOLDER)}\n")

    print("Connecting to the server... (this takes a few seconds)")

    try:
        with remotezip.RemoteZip(ZIP_URL) as zip_file:
            # 1. Search the zip file index for our target song
            print("Scanning file list...")
            all_files = zip_file.namelist()
            files_to_download = [f for f in all_files if TARGET_SONG in f]

            if not files_to_download:
                print(f"Error: Could not find '{TARGET_SONG}' inside the zip file.")
                print("Check the spelling or try a different song name.")
                return

            print(f"Found {len(files_to_download)} files related to this song.")
            print("Starting download... (Press Ctrl+C to stop)\n")

            # 2. Download files one by one
            count = 0
            for filename in files_to_download:
                # Skip folder definitions (entries ending in /)
                if filename.endswith('/'):
                    continue

                # Clean up the path: Remove "EnsembleSet/" from the start
                clean_path = filename.replace("EnsembleSet/", "")
                local_filepath = os.path.join(OUTPUT_FOLDER, clean_path)

                # Create the folder structure locally
                os.makedirs(os.path.dirname(local_filepath), exist_ok=True)

                # Only download if we don't have it yet
                if not os.path.exists(local_filepath):
                    print(f"Downloading [{count+1}/{len(files_to_download)}]: {os.path.basename(filename)}")
                    
                    # Extract specific file to the destination
                    # We read the bytes and write them manually to control the path
                    with zip_file.open(filename) as source, open(local_filepath, "wb") as target:
                        target.write(source.read())
                else:
                    print(f"Skipping [{count+1}/{len(files_to_download)}]: {os.path.basename(filename)} (Already exists)")
                
                count += 1

            print(f"\nSUCCESS! Download complete.")
            print(f"Go to the folder '{OUTPUT_FOLDER}' to see your files.")

    except KeyboardInterrupt:
        print("\n\nDownload stopped by user.")
        sys.exit()
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("Check your internet connection or try again later.")

if __name__ == "__main__":
    main()
