# AI Context: EnsembleSet Source Separation Project

## 1. Project Overview
This project focuses on **Music Source Separation (MSS)** using the **EnsembleSet** dataset. 
The goal is to separate a full chamber music mix (String Quartets, etc.) into individual instrument stems (Violin, Cello, Viola).

## 2. Dataset Architecture
The dataset is a "Virtual Recording Studio" simulation.
* **Source:** Zenodo (BBCSO_Ensembles).
* **Uniqueness:** Every song contains ~20 subfolders representing different microphone positions.
* **Constraint:** We do **not** download the full 58GB dataset. We stream/download specific songs and specific folders programmatically.

## 3. Directory Structure & Semantics
The project relies on a strict directory structure. AI agents must respect the distinction between "Input" and "Ground Truth."

### Active Folders (Keep)
* `Mix_1/` -> **INPUT (X)**: The professional stereo mix. This is the audio the model listens to.
* `Close/` -> **TARGETS (Y)**: Spot microphones for individual instruments. These are the ground truth stems.
    * *Note:* Filenames inside `Close/` (e.g., `Cello.flac`) represent the clean, dry signal of that instrument.

### Optional Folders (Grouped Stems)
* `SpStr/` (Spot Strings), `SpWW/` (Spot Woodwinds), `SpBr/` (Spot Brass).
* *Usage:* Use these only if training for section separation, not instrument separation.

### Ignored Folders (Noise/Reverb)
**DO NOT USE** the following folders for training. They contain excessive room reverb or redundant data:
* `Amb` (Ambient), `Tree` (Decca Tree), `Balcony`, `Floor`, `Out`, `Outr`, `Sides`, `Mids`, `Mono`, `AtmosF`, `AtmosR`.

## 4. Variable Mapping
When writing data loading scripts, map the files as follows:

| Variable | Folder Path | Description |
| :--- | :--- | :--- |
| `input_audio` | `.../Mix_1/mix.wav` | The full song mixture. |
| `target_stems` | `.../Close/*.flac` | Dictionary of individual instruments (e.g., {'Violin_1': data, 'Cello': data}). |

## 5. Known Issues & Edge Cases
1.  **Duplicate Filenames:** The file `Cello.flac` exists in *every* subfolder (`Close/Cello.flac`, `Amb/Cello.flac`). 
    * *Rule:* Always check the **parent folder** before processing a file. Never process based on filename alone.
2.  **Download Script:** We use `remotezip` to fetch partial data from Zenodo. Do not suggest downloading the full zip via `wget`.

## 6. Tech Stack
* **Language:** Python 3.x
* **Audio Loading:** `librosa` or `torchaudio`
* **Data Handling:** `remotezip`, `pandas`

## 7. Dependency Management
* **Requirements File:** Always keep `requirements.txt` updated when adding new dependencies.