# tests/pipelines

Testy nodów i kontraktu połączenia S2-2/S2-3:

```bash
pytest -q tests/pipelines
python scripts/check_pipeline_integration.py --env ci
```

`assert_prepared_data` w `asi_project.pipelines.contracts` jest wspólną asercją
do testów nodów Antona. `test_integration_contract.py` sprawdza kontrakt i harness
na jawnie oznaczonym stand-inie; nie zastępuje implementacji `data_processing`.

`test_catalog_integration.py` kopiuje konfigurację base/ci do tymczasowego projektu,
wczytuje różne pliki CSV dla obu środowisk i sprawdza rzeczywiste fabryki oraz
zapis/odczyt artefaktów. Nie dziedziczy lokalnych nadpisań ani nie zapisuje poza
tymczasowym projektem. Krok `kedro run --env ci` w workflow należy do S2-4.

## Sprint 3 — eksperymenty lokalne

`test_experiment_runner.py` uruchamia rzeczywiste przygotowanie danych i trzy protokoły
Kedro, z końcowym testem i bez niego. Używa katalogów tymczasowych i działa bez SDK W&B
oraz AutoGluona. Pozostałe testy pokrywają progi, wyrównanie klas/indeksów, natywny
zapis/odczyt, osobny proces bez treningu, błędy I/O i niezmienność zapisanej selekcji.

`test_experiment_adapters.py` i test adaptera holdout sprawdzają kontrakt wywołań SDK
na jawnych atrapach; nie potwierdzają treningu AutoGluona. Rzeczywiste przebiegi na
próbce/pełnych danych i komendy są w
[notatce S3](../../docs/notes/sprint-3-local-experiments.md).
