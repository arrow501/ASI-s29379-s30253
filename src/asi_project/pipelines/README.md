# Pipeline’y Kedro

Kedro łączy nody w pipeline’y i wczytuje konfigurację z katalogu `conf/`. Nody korzystają z istniejącej logiki przygotowania danych i treningu w `data.py` oraz `model.py`.

| Pipeline | Sprint | Wejście i wynik |
| --- | --- | --- |
| [`data_processing`](data_processing/README.md) | S2-2 | Waliduje i normalizuje `raw_data`, usuwa rekordy bez targetu i zwraca `prepared_data`. |
| [`data_science`](data_science/README.md) | S2-3, rozwijany w S3 | Dzieli `prepared_data`, trenuje modele i zapisuje model oraz metryki. |

`prepared_data` zawiera target `y` i zachowuje kolejność rekordów. Po połączeniu obu części `kedro run` przekazuje ten DataFrame w pamięci z `data_processing` do `data_science`. Domyślne uruchomienie korzysta z pełnych danych, a `--env ci` wybiera próbkę z repozytorium.

Polecenia uruchomienia i testów S2-2 są w [README `data_processing`](data_processing/README.md). Testy obu pipeline’ów są w [`tests/pipelines`](../../../tests/pipelines).

W Sprincie 3 `data_science` zostanie rozszerzony o rejestrowanie eksperymentów w W&B, AutoGluon i pakiet modelu.
