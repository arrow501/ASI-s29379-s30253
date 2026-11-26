# conf/ci — konfiguracja Kedro dla CI

Opcja `--env ci` ładuje konfigurację bazową i nakłada na nią ustawienia z tego katalogu.

## Próbka danych

[`catalog.yml`](catalog.yml) nadpisuje wpis `raw_data`. Kedro wczytuje `data/01_raw/sample.csv` z separatorem `,`. Próbka jest w repozytorium, więc można uruchomić pipeline bez pobierania pełnych danych:

```bash
kedro run --pipeline data_processing --env ci
```

Nazwa wejścia `raw_data` pozostaje taka sama. Dzięki temu `data_processing` wykonuje tę samą walidację, normalizację i filtrowanie na obu źródłach.

## Konfiguracja kolejnych etapów

Catalog ma osobne ścieżki dla modelu, metryk, raportów eksperymentów i pakietów z próbki CI.
`parameters_experiments.yml` wyłącza AutoGluona; zwykły workflow CI ustawia
`WANDB_MODE=disabled`. Po `experiments_test` uruchamia lokalny smoke pakietu baseline’u.
Osobny workflow `Production model smoke` pobiera model z W&B online:
[komendy i odbiór](../../docs/notes/sprint-3-smoke-test.md).
