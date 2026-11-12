# data_processing — przygotowanie danych

Pipeline przygotowuje dane Bank Marketing do treningu. Wykorzystuje reguły ze Sprintu 1 zapisane w [`data.py`](../../data.py).

## Przebieg

1. `validate_data` sprawdza wymagane kolumny i klasy targetu.
2. `normalize_data` przycina tekst i zamienia `pdays=-1` na brak wartości.
3. `filter_data` usuwa rekordy bez targetu `y`, zachowując kolejność pozostałych wierszy.

Data Catalog dostarcza `raw_data`, a `params:project` wskazuje kontrakt zbioru. Wynik `prepared_data` zachowuje target i kolejność rekordów. W pełnym przebiegu Kedro przekazuje go w pamięci do `data_science`, gdzie odbywają się podział danych, trening i ewaluacja.

## Źródła danych

| Środowisko | Plik wejściowy | Separator |
| --- | --- | --- |
| `local` (domyślne) | `data/01_raw/bank_marketing/bank-full.csv` | `;` |
| `ci` | `data/01_raw/sample.csv` | `,` |

## Uruchomienie i testy

Polecenia uruchamiaj z katalogu głównego projektu w środowisku `asi-ml`. Próbka CI jest dostępna w repozytorium:

```bash
kedro run --pipeline data_processing --env ci
```

Pełne dane pobierz przed lokalnym uruchomieniem:

```bash
python scripts/download_data.py
kedro run --pipeline data_processing
```

Testy sprawdzają obsługę braków, normalizację `pdays`, kolejność rekordów i zachowanie wierszy z poprawnym targetem:

```bash
python -m pytest -q tests/pipelines/test_data_processing.py tests/pipelines/test_data_processing_pipeline.py
```
