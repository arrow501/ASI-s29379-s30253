# conf/ — konfiguracja Kedro

Wspólna konfiguracja działających pipeline’ów Kedro oraz lokalnych eksperymentów S3.

| Katalog | Sprint | Rola |
|---|---|---|
| `conf/base/` | S2 (rozszerzany w S3) | konfiguracja wspólna: katalog danych i parametry |
| `conf/local/` | S2 | ustawienia lokalne i poświadczenia — poza gitem (oprócz README) |
| `conf/ci/` | S2 | nadpisania dla CI: te same pipeline'y na próbce `data/01_raw/sample.csv` |

Kontrakt (z planu Sprintu 2):

- `kedro run` — pełne dane (`base` + `local`);
- `kedro run --env ci` — próbka CI (`base` + `ci`), uruchamiane w GitHub Actions;
- kod pipeline'ów nie zna ścieżek do plików — ścieżki i parametry są w konfiguracji.

Parametry modeli i limity: `base/parameters_experiments.yml`; liczby foldów:
`base/parameters_validation_protocols.yml`. `ci` wyłącza AutoGluona. Proporcje splitu,
cechy i seed nadal pochodzą z `project.yaml`. Przypięta wersja Kedro:
`requirements/preflight-s2.txt`.
