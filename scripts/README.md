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

## Future scaffold (Sprinty 3–6)

> Scaffold for Sprints 3–6 — do not implement before these sprints.

| Skrypt | Sprint | Kontrakt |
|---|---|---|
| smoke modelu | S3 | pobranie pakietu `production` z W&B → wczytanie w czystym procesie → predykcja 5 wierszy → sprawdzenie schematu odpowiedzi |
| wspólny smoke API | S4 → S6 | `/healthz` → tryb persistence → `/predict` z tokenem → kontrakt odpowiedzi → w trybie `sql` zapisany rekord, w trybie `disabled` brak wymogu zapisu; ten sam skrypt na kontenerze, na compose i na Cloud Run |

Skrypt wdrożenia (`deploy.sh`, Sprint 6) znajdzie się w katalogu `deploy/`.
