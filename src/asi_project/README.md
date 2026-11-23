# Pakiet `asi_project`

Wspólny kod projektu. Jeden pakiet gwarantuje, że trening i serwowanie korzystają z tej samej
logiki (np. `normalize_input` działa identycznie w notebooku, w pipeline Kedro i w API).

## Aktywne w Sprincie 1

| Moduł | Rola |
|---|---|
| `datasets.yaml` | kontrakt zatwierdzonych datasetów (prowadząca; zespół nie edytuje) |
| `config.py` | odczyt `project.yaml` i rejestru datasetów |
| `data.py` | kontrakt danych, input normalization, training-only filtering, X/y, split |
| `model.py` | DummyClassifier i gotowy sklearn Pipeline z model preprocessing |
| `metrics.py` | metryki i kontrakt `metrics.json` |
| `download.py` | pobieranie danych i odcisk SHA-256 |

## Pipeline’y S2/S3 i granica z serwowaniem

| Element | Sprint | Kontrakt |
|---|---|---|
| `pipelines/data_processing/` | S2 | nody: walidacja → input normalization → training-only filtering (funkcje z `data.py`) |
| `pipelines/data_science/` | S2 | baseline: split → trening Pipeline → ewaluacja |
| `pipelines/experiments/` | S3 | podział/foldy → trening → zamrożenie decyzji → opcjonalny test |
| `provenance.py`, `experiment_store.py` | S3 | fingerprint/SHA oraz lokalizacja wyników z catalogu |
| `pipeline_registry.py`, `settings.py` | S2 | pliki projektu Kedro (rejestracja pipeline'ów, ustawienia) |
| `model_package.py` (do realizacji w #20) | S3 | model i `metadata.json` (cechy, target, klasa pozytywna, próg, metryki, SHA, fingerprint, wersje); wspólny zapis/odczyt dla Kedro, smoke modelu i API |
| `api/` | S4 | usługa predykcji — patrz `api/README.md` |

Zasada: moduły aktywne w Sprincie 1 nie importują Kedro, W&B, AutoGluona, FastAPI ani innych
bibliotek przyszłych sprintów. Pilnuje tego test w `tests/test_repo_hygiene.py`.
