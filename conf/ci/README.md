# conf/ci — konfiguracja Kedro dla CI

Opcja `--env ci` ładuje konfigurację bazową i nakłada na nią ustawienia z tego katalogu.

## Próbka danych

[`catalog.yml`](catalog.yml) nadpisuje wpis `raw_data`. Kedro wczytuje `data/01_raw/sample.csv` z separatorem `,`. Próbka jest w repozytorium, więc można uruchomić pipeline bez pobierania pełnych danych:

```bash
kedro run --pipeline data_processing --env ci
```

Nazwa wejścia `raw_data` pozostaje taka sama. Dzięki temu `data_processing` wykonuje tę samą walidację, normalizację i filtrowanie na obu źródłach.

## Konfiguracja kolejnych etapów

S2-3 dodaje osobne ścieżki dla modelu i metryk z próbki CI. Od Sprintu 3 konfiguracja CI wyłączy wysyłanie danych do W&B przez `WANDB_MODE=disabled` i pozwoli wyłączyć AutoGluon lub ograniczyć czas jego działania.
