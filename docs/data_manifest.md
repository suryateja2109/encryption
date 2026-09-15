# Data Manifest: Labeled Faces in the Wild (LFW-Deepfunneled)

## 1. Environment & Hardware Audit

- **Operating System**: Windows 11 (AMD64)
- **Python Runtime**: Python 3.12.3 (64-bit AMD64, MSC v.1938)
- **Host GPU Availability**: No discrete NVIDIA GPU detected (`nvidia-smi` not found). All computation executed via optimized NumPy / SciPy with AVX2 multi-threading and Numba JIT acceleration where appropriate.
- **Core Dependencies Installed**:
  - `numpy`: 1.26.0
  - `scipy`: 1.12.0
  - `matplotlib`: 3.8.3
  - `pandas`: 2.2.1
  - `scikit-image`: 0.26.0
  - `scikit-learn`: 1.4.1.post1
  - `opencv-python`: 4.11.0.86
  - `deepface`: 0.0.100
  - `mealpy`: 3.0.3
  - `nistrng`: 1.2.3
  - `streamlit`: 1.63.0
  - `pytest`: 9.1.1
  - `pillow`: 12.3.0

## 2. Dataset Overview (`archive/`)

The facial dataset provided in `archive/` corresponds to the official **Labeled Faces in the Wild (LFW) - Deepfunneled** aligned face dataset, supplemented by benchmark verification splits.

| Characteristic | Measured Value |
|---|---|
| **Root Location** | `archive/` |
| **Image Subdirectory** | `archive/lfw-deepfunneled/lfw-deepfunneled/` |
| **Total Images** | 13,233 images |
| **Total Individuals (Identities)** | 5,749 individuals |
| **Image Format** | JPEG (`.jpg`), 3-channel RGB |
| **Resolution Distribution** | $250 \times 250$ pixels (uniform across all 13,233 images) |
| **Images per Person (Min)** | 1 image |
| **Images per Person (Max)** | 530 images (George_W_Bush) |
| **Images per Person (Mean)** | 2.30 images |
| **Images per Person (Median)** | 1.0 image |

## 3. Metadata and Split Analysis

The root of `archive/` contains 10 standardized CSV metadata files:

1. `lfw_allnames.csv` (5,749 rows): Lists each identity and total image count.
2. `people.csv` (5,749 rows): Identical census mapping of identities to image count.
3. `pairs.csv` (6,000 rows): Standard 10-fold cross-validation pairs for face verification benchmarking (3,000 matched pairs, 3,000 mismatched pairs).
4. `matchpairsDevTrain.csv` (1,100 pairs): Training matched identity pairs.
5. `mismatchpairsDevTrain.csv` (1,100 pairs): Training mismatched identity pairs.
6. `matchpairsDevTest.csv` (500 pairs): Testing matched identity pairs (`name`, `imagenum1`, `imagenum2`).
7. `mismatchpairsDevTest.csv` (500 pairs): Testing mismatched identity pairs (`name1`, `imagenum1`, `name2`, `imagenum2`).
8. `peopleDevTrain.csv` (4,038 individuals): Development train split identities.
9. `peopleDevTest.csv` (1,711 individuals): Development test split identities.
10. `lfw_readme.csv`: Dataset release notes.

## 4. Experimental Facial Database vs. Probe Image Architecture

In Section IV-B and Section V-A of Ding et al. (*IEEE TCSVT* 2025), the authors construct a facial database of known individuals from bulk images and then evaluate selective face encryption against probe images (reporting match vs. mismatch decisions and Euclidean distances against threshold 0.5 in Table II).

To mirror this experimental architecture faithfully:

1. **Gallery Database (Known Individuals)**:
   - Built from identities with multiple images available in the dataset (using `matchpairsDevTest.csv` and `peopleDevTrain.csv`).
   - For each enrolled individual, the primary image (e.g., `_0001.jpg`) is extracted, facial ROI detected via `deepface`, and its feature embedding computed and indexed in `src/face_processing/database.py`.
2. **Probe Query Set (Test Evaluation)**:
   - Matched Probes: Secondary images of enrolled individuals (e.g., `_0002.jpg` from `matchpairsDevTest.csv`). These must successfully match (distance $< 0.5$), triggering selective encryption.
   - Mismatched Probes: Images of individuals absent from the gallery (drawn from `mismatchpairsDevTest.csv`). These must be rejected (distance $\ge 0.5$), leaving the image unencrypted or encrypted under fallback protocol.
   - This matches the paper's Table II validation protocol.
