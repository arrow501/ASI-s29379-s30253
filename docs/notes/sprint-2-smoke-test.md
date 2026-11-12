## Smoke test Sprintu 2

- Data: 2026-11-12. Windows, świeży klon commita `2168f88` i nowe środowisko Conda.
- Wersje: Python 3.11.18, Kedro 1.7.0, `kedro-datasets` 10.0.0.
- `kedro run --env ci` — model i metryki zapisane w podkatalogach `ci/`. Split 480 / 120.
- `python scripts/download_data.py` i `kedro run` — model i metryki zapisane dla pełnych danych. Split 36 168 / 9 043.
- Pełny raport metryk jest identyczny z referencją Sprintu 1. F1 regresji logistycznej: 0,2238069116840373.
- `pytest` — 229 testów przeszło.
- `pre-commit run --all-files` — wszystkie 9 kontroli przeszło.
