# tests/pipelines

Testy nodów i kontraktu połączenia S2-2/S2-3:

```bash
pytest -q tests/pipelines
python scripts/check_pipeline_integration.py --env ci
```

`assert_prepared_data` w `asi_project.pipelines.contracts` jest wspólną asercją
do testów nodów Antona. `test_integration_contract.py` sprawdza kontrakt i harness
na jawnie oznaczonym stand-inie; nie zastępuje implementacji `data_processing`.

Komenda integracyjna używa prawdziwych fabryk i kończy się kodem 2, gdy #11
jeszcze nie jest zaimplementowane. Jej wynik musi być zielony po integracji.
Krok `kedro run --env ci` w workflow należy do S2-4.
