# Neural Networks Project

Projekt dotyczy analizy obrazow USG piersi na danych BUSI. Repozytorium zawiera notebooki i pliki wynikowe dla glownych zadan:

- klasyfikacji obrazow,
- detekcji zmian metodami YOLO,
- interpretowalnosci modeli z uzyciem Grad-CAM.

Glowne zbiory danych w projekcie:

- `Dataset_BUSI_with_GT/` - oryginalny zbior BUSI,
- `Dataset_BUSI_noisy/` - wersja rozszerzona o dodatkowe obrazy z szumem Gaussa,
- `Dataset_BUSI_YOLO/` - dane przygotowane do detekcji w formacie YOLO (`images/`, `labels/`, `data.yaml`).

## Struktura projektu

### Przygotowanie danych

- `prepare_data.ipynb` - przygotowanie danych BUSI do detekcji YOLO: augmentacja szumem Gaussa, konwersja masek do bounding boxow, podzial train/val/test oraz zapis `data.yaml`.
- `augment_busi_noise.py` - skrypt Python do tworzenia zaszumionych kopii obrazow BUSI w osobnym folderze.

### Klasyfikacja

- `Classification.ipynb` - trening i porownanie modeli `ResNet50`, `DenseNet121` oraz `EfficientNetB7` dla klasyfikacji klas `benign`, `malignant`, `normal` na zbiorze `Dataset_BUSI_noisy`.
- `resnet50_results.csv` - przebieg treningu i metryki dla modelu ResNet50.
- `densenet121_results.csv` - przebieg treningu i metryki dla modelu DenseNet121.
- `efficientnetb7_results.csv` - przebieg treningu i metryki dla modelu EfficientNetB7.
- `classification_summary.csv` - zbiorcze porownanie wynikow modeli klasyfikacyjnych.
- `classification_comparison.png` - wykres porownujacy wyniki klasyfikacji.

### Detekcja

- `detection.ipynb` - trening, walidacja i predykcje dla modelu YOLO na danych z `Dataset_BUSI_YOLO`.
- `yolo11n.pt` - wagi startowe modelu YOLO11n uzywane w notebooku detekcji.
- `yolov8_result.csv` - zapis metryk treningu modelu YOLOv8.
- `yolov11_result.csv` - zapis metryk treningu modelu YOLOv11.
- `comparision.ipynb` - porownanie wynikow detekcji YOLOv8 i YOLOv11 na podstawie plikow CSV.
- `runs/` - katalog z wynikami uruchomien Ultralytics, w tym treningami i predykcjami.

### Grad-CAM i interpretowalnosc

- `gradcam_option_A_pretrained_demo.ipynb` - demonstracja Grad-CAM dla wstepnie wytrenowanego `ResNet50` bez dodatkowego trenowania na BUSI.
- `gradcam_option_C_quick_finetune.ipynb` - szybki fine-tuning klasyfikatora (`resnet18` lub `resnet50`) na BUSI i interpretacja Grad-CAM dla dotrenowanego modelu.
- `gradcam_comparison_pretrained_vs_finetuned.ipynb` - porownanie map Grad-CAM dla modelu pretrenowanego i modelu po fine-tuningu na tych samych obrazach.

### Dokumentacja i materialy pomocnicze

- `requirements.txt` - lista bibliotek Python potrzebnych do uruchomienia projektu.
- `README.md` - opis projektu, plikow i sposobu uruchomienia.
- `source/Gleboki-projekt-2.pdf` - material zrodlowy lub raport projektowy.

## Sugerowana kolejnosc pracy

1. Uruchom `prepare_data.ipynb`, aby przygotowac dane do detekcji oraz wygenerowac zbior rozszerzony.
2. Uruchom `Classification.ipynb`, jesli chcesz odtworzyc wyniki klasyfikacji.
3. Uruchom `detection.ipynb`, jesli chcesz trenowac lub ewaluowac modele YOLO.
4. Uruchom notebooki `gradcam_*.ipynb`, jesli chcesz analizowac interpretowalnosc modeli.
5. Uzyj `comparision.ipynb`, aby porownac zapisane wyniki detekcji YOLOv8 i YOLOv11.

## Jak utworzyc srodowisko venv

Projekt zaklada uzycie Python `3.10.x`.

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Sprawdzenie wersji Pythona

Po aktywacji srodowiska mozesz sprawdzic wersje:

```bash
python --version
```

Oczekiwany wynik:

```bash
Python 3.10.x
```

## Uwagi

- Notebook `detection.ipynb` wymaga, aby wczesniej istnial plik `Dataset_BUSI_YOLO/data.yaml` wygenerowany przez `prepare_data.ipynb`.
- Czesc notebookow zapisuje wyniki do plikow CSV i katalogu `runs/`.
- Notebooki Grad-CAM korzystaja z biblioteki `grad-cam`, ktora jest juz uwzgledniona w `requirements.txt`.
