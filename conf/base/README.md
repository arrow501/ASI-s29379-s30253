# conf/base — wspólna konfiguracja Kedro

Kedro ładuje ten katalog jako podstawę dla wszystkich środowisk. Domyślne uruchomienie używa `base` i ewentualnych nadpisań z `conf/local/`.

## Katalog danych

[`catalog.yml`](catalog.yml) definiuje wejście `raw_data` dla `data_processing`. W środowisku lokalnym Kedro wczytuje `data/01_raw/bank_marketing/bank-full.csv` z separatorem `;`.

Aby pobrać pełne dane i sprawdzić pipeline przygotowania, uruchom z katalogu głównego projektu:

```bash
python scripts/download_data.py
kedro run --pipeline data_processing
```

S2-3 uzupełnia katalog o ścieżki zapisu modelu i metryk.

## Parametry

[`parameters.yml`](parameters.yml) zawiera wspólne nadpisania parametrów. Domyślne ustawienia grupy `project`, w tym listę cech i parametry splitu, Kedro pobiera z [`project.yaml`](../../project.yaml) i kontraktu zbioru w [`datasets.yaml`](../../src/asi_project/datasets.yaml).

W Sprincie 3 ta konfiguracja zostanie rozszerzona o parametry eksperymentów, w tym limit czasu AutoGluona.
