# Pipeline’y Kedro

Kedro łączy nody w pipeline’y i wczytuje konfigurację z katalogu `conf/`. Nody korzystają z istniejącej logiki przygotowania danych i treningu w `data.py` oraz `model.py`.

| Pipeline | Sprint | Wejście i wynik |
| --- | --- | --- |
| [`data_processing`](data_processing/README.md) | S2-2 | Waliduje i normalizuje `raw_data`, usuwa rekordy bez targetu i zwraca `prepared_data`. |
| [`data_science`](data_science/README.md) | S2-3 | Baseline 80/20: split, trening i metryki; domyślne uruchomienie S2. |
| [`experiments`](../../../docs/notes/sprint-3-local-experiments.md) | S3-1 | Holdout, bagging i foldy chronologiczne; jawny wybór przed testem. |

`prepared_data` zawiera target `y` i zachowuje kolejność rekordów. Po połączeniu obu części `kedro run` przekazuje ten DataFrame w pamięci z `data_processing` do `data_science`. Domyślne uruchomienie korzysta z pełnych danych, a `--env ci` wybiera próbkę z repozytorium.

Polecenia uruchomienia i testów S2-2 są w [README `data_processing`](data_processing/README.md). Testy obu pipeline’ów są w [`tests/pipelines`](../../../tests/pipelines).

Eksperymenty S3 ponownie używają `data_processing` i fabryki baseline’u. Komendy, kontrakt
plików i wyniki są w notatce powyżej. Integracja W&B i docelowy pakiet modelu pozostają
kolejnymi pracami Sprintu 3.
