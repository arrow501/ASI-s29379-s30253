# scripts/

## Aktywne w Sprincie 1

| Skrypt | Rola |
|---|---|
| `download_data.py` | pobranie pełnych danych wybranego datasetu i odcisk SHA-256 |
| `make_ci_sample.py` | próbka CI `data/01_raw/sample.csv` |
| `check_env.py` | kontrola środowiska; `--autogluon` to pre-flight gotowości do Sprintu 3 |
| `check_dod.py` | raport Definition of Done (`--sprint 1`; kolejne sprinty dopiszą swoje punkty) |
| `check_metrics.py`, `project_status.py`, `run_notebook.py` | kroki CI |
| `build_issues.py` | generowanie `github_issues/` z `docs/sprints/` |
| `smoke_datasets.py` | narzędzie prowadzącej: smoke prawdziwych źródeł wszystkich datasetów |

## Sprint 2 — komponent treningowy

`check_data_science.py` uruchamia pipeline treningowy na danych przygotowanych
istniejącymi helperami S1, sprawdza zapis/odczyt modelu i zgodność predykcji z
metrykami. Nie zastępuje integracji wejściowego pipeline'u Antona.

```bash
python scripts/check_data_science.py --data-mode sample
python scripts/check_data_science.py --data-mode full --compare-baseline
```

Pełne dane muszą być pobrane; wyniki trafiają do ignorowanych ścieżek catalogu.
Opcja `--compare-baseline` wymaga pełnych danych i porównuje cały rekord z
niezmienionym `metrics.json` ze Sprintu 1.

Komenda `python scripts/check_pipeline_integration.py --env ci` sprawdza
rzeczywisty catalog wejściowy, kontrakt `prepared_data`, cały pipeline domyślny
oraz ponowny odczyt artefaktów przez nowy katalog. `--env local` sprawdza pełne
dane. Oba checki zapisują standardowe artefakty wybranego środowiska.
Szczegóły: `docs/notes/sprint-2-handoff.md`.

## Sprint 3 — pakiety modelu

| Skrypt | Rola |
|---|---|
| `manage_model_package.py export` | zapis pakietu wskazanego kandydata z istniejącego eksperymentu, bez treningu |
| `manage_model_package.py publish` | publikacja pakietu jako W&B Artifact; `--production` po zatwierdzeniu wyboru S3-2 |
| `manage_model_package.py download` | pobranie wersji lub aliasu i zapis identyfikatora konkretnej wersji |
| `predict_model_package.py` | wczytanie pakietu i predykcja CSV przez wspólny interfejs, bez treningu i W&B |
| `smoke_model.py` | pięć predykcji w nowym procesie; lokalny pakiet dla CI albo rzeczywiste pobranie W&B `production` z zapisem konkretnej wersji |

[Kompletne komendy PowerShell](../src/asi_project/README.md#komendy-obsługi-pakietu)
opisują środowisko, separatory, wersje artefaktów i lokalizację potwierdzeń.
Komendy odbioru S3-4 i rozróżnienie lokalnego CI od W&B online:
[smoke test produkcyjnego modelu](../docs/notes/sprint-3-smoke-test.md).

## Future scaffold (Sprinty 4–6)

> Scaffold for Sprints 4–6 — do not implement before these sprints.

| Skrypt | Sprint | Kontrakt |
|---|---|---|
| wspólny smoke API | S4 → S6 | `/healthz` → tryb persistence → `/predict` z tokenem → kontrakt odpowiedzi → w trybie `sql` zapisany rekord, w trybie `disabled` brak wymogu zapisu; ten sam skrypt na kontenerze, na compose i na Cloud Run |

Skrypt wdrożenia (`deploy.sh`, Sprint 6) znajdzie się w katalogu `deploy/`.
