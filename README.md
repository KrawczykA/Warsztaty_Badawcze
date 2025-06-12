# Warsztaty Badawcze 2
# Training SSL Model and Evaluating its Performance Against Various Representations
***Agata Krawczyk, Jakub Kaproń, Antoni Kingston, Miłosz Kita***

## Struktura Projektu

### 📁 `ssl_training/`
Folder zawierający implementację i trening modeli Self-Supervised Learning:

- **`datasets.py`** - funkcje pomocnicze do ładowania i pobierania zbiorów danych
- **`models.py`** - implementacje modeli SSL (BYOL, SimCLR, MAE) i baselinowego
- **`{byol/simclr/mae}_cifar100.ipynb`** - notebooki, gdzie trenowano poszczególne modele SSL na zbiorze CIFAR-100
- **`imagenet_and_baselines.ipynb`** - trening modeli SSL na ImageNet oraz trenowanie modeli baseline (supervised) na obu zbiorach danych

### 📁 `evaluation/`
Folder zawierający kod do ewaluacji wytrenowanych modeli:

- **`CKNNA.ipynb`** - notebook do ewaluacji metodą k-NN (k-Nearest Neighbors)
- **`datasets.py`** - funkcje pomocnicze do ładowania zbiorów danych do ewaluacji
- **`eval.py`** - główny plik ewaluacji:
  - Klasa `Config` - konfiguracja parametrów ewaluacji
  - `get_dataset()` - ładowanie różnych zbiorów danych (CIFAR-10/100, ImageNet, Flowers, Pets, Aircraft)
  - `load_model()` - wczytywanie modeli SSL z checkpointów
  - `extract_features()` - ekstraktowanie cech z modeli
  - `linear_probing()` - ewaluacja przez linear probing
  - `fine_tuning()` - ewaluacja przez fine-tuning
  - `knn_evaluation()` - klasyfikacja k-NN na reprezentacjach
  - `baseline_representations()` - redukcja wymiarów (PCA, t-SNE, UMAP)
  - Klasa `SSLEvaluator` - orkiestracja pełnej ewaluacji
- **`evaluating_baselines.ipynb`** - notebook do ewaluacji modeli baseline:
- **`evaluation_whole.py`** - skrypty do uruchamiania ewaluacji:
  - `test_all_100()` - ewaluacja wszystkich modeli na CIFAR-100
  - `test_mae_imgnet()`, `test_byol_imgnet()`, `test_simclr_imgnet()` - ewaluacja modeli trenowanych na ImageNet
  - `test_add_mae_*()` - ewaluacja modeli MAE na dodatkowych zbiorach (Flowers, Pets, Aircraft)
- **`models.py`** - implementacje modeli SSL (BYOL, SimCLR, MAE) i baselinowego
- **`tsneumap.ipynb`** - notebook do wizualizacji reprezentacji z użyciem t-SNE i UMAP
